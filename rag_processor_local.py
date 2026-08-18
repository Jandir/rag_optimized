#!/usr/bin/env python3
"""
Script: rag_processor_local.py
Descrição: Automatiza a adaptação de transcrições de vídeo para RAG usando NLP heurístico local.
           Usa spaCy para NER, YAKE para palavras-chave e TextTiling para segmentação.
           Mantém as regras de terminologia (Sete Montes, Ekklezia, etc.).
"""

import os
import sys
import re
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import spacy
import yake
from nltk.tokenize.texttiling import TextTilingTokenizer
from dotenv import load_dotenv

# --- EDUCATIVO: O QUE ESTE SCRIPT FAZ? ---
# Este script pega transcrições de vídeo (em formato .txt ou .srt) e as prepara para sistemas de RAG.
# RAG é uma técnica onde a IA "lê" seus documentos para responder perguntas.
# O script limpa o texto, extrai palavras-chave, identifica tópicos e organiza tudo em Markdown.

# --- Constants & Configuration ---
MONTHS_PT_DICT: Dict[str, str] = {
    "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
    "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
    "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro",
    "1": "Janeiro", "2": "Fevereiro", "3": "Março", "4": "Abril",
    "5": "Maio", "6": "Junho", "7": "Julho", "8": "Agosto",
    "9": "Setembro", "10": "Outubro", "11": "Novembro", "12": "Dezembro"
}

EXCLUDED_FILES_SET: set[str] = {
    "historico.txt", "cookies.txt", "requirements.txt", "rules.txt",
    "LICENSE", "README.md", "rag_processor.py", "rag_processor_local.py"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR_PATH: str = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR_PATH, '.env'))

# --- Helper Functions ---

HTML_TAG_PATTERN: re.Pattern = re.compile(r'<[^>]*>')

def _parse_srt_blocks(content_str: str) -> List[str]:
    """Extracts text blocks from SRT content, removing tags."""
    blocks_list: List[str] = []
    raw_blocks_list: List[str] = content_str.split('\n\n')
    for block_str in raw_blocks_list:
        arrow_idx_int: int = block_str.find('-->')
        if arrow_idx_int != -1:
            text_start_idx_int: int = block_str.find('\n', arrow_idx_int)
            if text_start_idx_int != -1:
                text_block_str: str = block_str[text_start_idx_int:].strip()
                text_block_str = HTML_TAG_PATTERN.sub('', text_block_str)
                if text_block_str:
                    blocks_list.append(text_block_str)
    return blocks_list

def _handle_simple_repetition(prev_text_str: str, curr_text_str: str) -> Optional[str]:
    """Retorna a nova parte do texto se for uma repetição simples, caso contrário None."""
    if curr_text_str.startswith(prev_text_str):
        return curr_text_str[len(prev_text_str):].strip()
    return None

def _handle_partial_overlap(prev_text_str: str, curr_text_str: str) -> List[str]:
    """Identifica sobreposições parciais linha a linha e retorna as linhas únicas."""
    prev_lines_list: List[str] = [line.strip() for line in prev_text_str.split('\n') if line.strip()]
    curr_lines_list: List[str] = [line.strip() for line in curr_text_str.split('\n') if line.strip()]
    start_idx_int: int = 0

    if prev_lines_list and curr_lines_list:
        if curr_lines_list[0] == prev_lines_list[-1]:
            start_idx_int = 1
        elif len(prev_lines_list) < len(curr_lines_list) and curr_lines_list[:len(prev_lines_list)] == prev_lines_list:
            start_idx_int = len(prev_lines_list)

    return curr_lines_list[start_idx_int:]

def _deduplicate_srt_lines(blocks_list: List[str]) -> List[str]:
    """Lógica modular para remover repetições em legendas do tipo 'rollup'."""
    if not blocks_list:
        return []

    cleaned_lines_list: List[str] = [blocks_list[0]]
    for i_int in range(1, len(blocks_list)):
        prev_text_str: str = blocks_list[i_int - 1]
        curr_text_str: str = blocks_list[i_int]

        new_part_str: Optional[str] = _handle_simple_repetition(prev_text_str, curr_text_str)
        if new_part_str is not None:
            if new_part_str:
                cleaned_lines_list.append(new_part_str)
            continue

        unique_lines_list: List[str] = _handle_partial_overlap(prev_text_str, curr_text_str)
        cleaned_lines_list.extend(unique_lines_list)

    return cleaned_lines_list

def clean_srt_content(content_str: str) -> str:
    """Limpa arquivos .srt removendo tempos e deduplicando conteúdo rollup."""
    content_str = content_str.replace('\r\n', '\n')
    blocks_list: List[str] = _parse_srt_blocks(content_str)
    cleaned_lines_list: List[str] = _deduplicate_srt_lines(blocks_list)
    return ' '.join(cleaned_lines_list)

def _parse_rule_line(line_str: str) -> Optional[Dict[str, Any]]:
    """Analisa uma única linha do arquivo de regras."""
    line_str = line_str.strip()
    if not line_str or line_str.startswith('#'):
        return None

    is_regex_bool: bool = line_str.startswith('REGEX:')
    if is_regex_bool:
        line_str = line_str[6:].strip()

    if '->' in line_str:
        parts_list: List[str] = line_str.split('->', 1)
        return {
            "original": parts_list[0].strip(),
            "replacement": parts_list[1].strip(),
            "is_regex": is_regex_bool
        }
    return None

def load_rules(rules_path_str: str = "rules.txt") -> List[Dict[str, Any]]:
    """Carrega regras de terminologia de forma modular."""
    absolute_path_str: str = os.path.join(SCRIPT_DIR_PATH, rules_path_str)
    rules_list: List[Dict[str, Any]] = []

    if not os.path.exists(absolute_path_str):
        logger.warning(f"Rules file not found: {absolute_path_str}")
        return rules_list

    try:
        with open(absolute_path_str, 'r', encoding='utf-8') as f_obj:
            for line_str in f_obj:
                rule_dict = _parse_rule_line(line_str)
                if rule_dict:
                    rules_list.append(rule_dict)
        return rules_list
    except Exception as error_obj:
        logger.error(f"Error loading rules: {error_obj}")
        return []

def enforce_terminology(text_str: str, rules_list: List[Dict[str, Any]]) -> str:
    """Aplica substituições de termos baseadas nas regras."""
    for rule_dict in rules_list:
        if rule_dict["is_regex"]:
            text_str = re.sub(rule_dict["original"], rule_dict["replacement"], text_str)
        else:
            text_str = text_str.replace(rule_dict["original"], rule_dict["replacement"])
    return text_str

def _extract_event_date(clean_name_str: str) -> str:
    """Extrai e formata a data do evento."""
    date_match_obj: Optional[re.Match] = re.search(
        r'(Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)\s+(\d{4})',
        clean_name_str,
        re.I
    )
    if not date_match_obj:
        return "N/A"

    month_abbr_str: str = date_match_obj.group(1).capitalize()[:3]
    year_str: str = date_match_obj.group(2)
    full_month_str: str = MONTHS_PT_DICT.get(month_abbr_str, month_abbr_str)
    return f"{full_month_str} de {year_str}"

def _extract_video_id(clean_name_str: str) -> str:
    """Extrai o ID do vídeo do YouTube."""
    video_id_match_obj: Optional[re.Match] = re.search(
        r'(?:\[|[-_])([a-zA-Z0-9_-]{11})(?:\])?$',
        clean_name_str
    )
    return video_id_match_obj.group(1) if video_id_match_obj else "N/A"

def extract_metadata_from_filename(filename_str: str) -> Dict[str, str]:
    """Extrai título, data e ID do vídeo de forma modular."""
    clean_name_str: str = (
        filename_str.replace(" Transcrição.txt", "")
        .replace(".txt", "")
        .replace(" Transcrição.srt", "")
        .replace(".srt", "")
        .strip()
    )
    event_date_str: str = _extract_event_date(clean_name_str)
    video_id_str: str = _extract_video_id(clean_name_str)
    title_str: str = clean_name_str
    
    if "MasterMind" in clean_name_str and event_date_str != "N/A":
        title_str = f"MasterMind {event_date_str.replace(' de ', ' ')}"
            
    return {"title": title_str, "event_date": event_date_str, "video_id": video_id_str}

# --- Heuristic Engine ---

class HeuristicProcessor:
    """
    O 'Cérebro' do script. Usa Processamento de Linguagem Natural (NLP) local
    para analisar o texto sem precisar de APIs externas pagas.
    """
    
    def __init__(self):
        # Carregamos o modelo de Português do spaCy (NER e Sentenças)
        logger.info("Loading spaCy model...")
        self.nlp_obj = spacy.load("pt_core_news_sm")
        # Configuramos o extrator de palavras-chave YAKE
        self.kw_extractor_obj = yake.KeywordExtractor(lan="pt", n=3, dedupLim=0.9, top=10)
        self.sec_kw_extractor_obj = yake.KeywordExtractor(lan="pt", n=2, top=3)
        # O TextTiling ajuda a dividir o texto em seções baseadas em mudança de tópico
        self.tt_tokenizer_obj = TextTilingTokenizer()

    def clean_filler_words(self, text_str: str) -> str:
        """Removes common Portuguese filler words via regex."""
        fillers_list: List[str] = [
            r'\bne\b', r'\bentão\b', r'\btipo\b', r'\bsabe\b', r'\bpra\b', r'\btá\b', r'\bgente\b'
        ]
        pattern_obj: re.Pattern = re.compile('|'.join(fillers_list), re.IGNORECASE)
        return pattern_obj.sub('', text_str).replace('  ', ' ').strip()

    def _extract_keywords(self, text_str: str) -> List[str]:
        """Extracts top keywords using YAKE."""
        return [kw[0] for kw in self.kw_extractor_obj.extract_keywords(text_str)]

    def _extract_entities(self, doc_obj: Any) -> List[str]:
        """Extracts top entities (ORG, PER, LOC)."""
        entities_list: List[str] = [
            ent.text for ent in doc_obj.ents if ent.label_ in ["ORG", "PER", "LOC"]
        ]
        return list(dict.fromkeys(entities_list))[:5]

    def _segment_content(self, doc_obj: Any) -> List[str]:
        """Segments text using TextTiling with a sentence-to-paragraph fallback."""
        sentences_list: List[str] = [sent.text.strip() for sent in doc_obj.sents]
        paragraphs_list: List[str] = [
            " ".join(sentences_list[i:i+5]) for i in range(0, len(sentences_list), 5)
        ]
        segmented_text_str: str = "\n\n".join(paragraphs_list)
        
        try:
            return self.tt_tokenizer_obj.tokenize(segmented_text_str)
        except Exception:
            return [segmented_text_str]

    def _generate_markdown_header(self, meta_dict: Dict[str, str], entities_list: List[str], keywords_list: List[str]) -> str:
        """Gera o cabeçalho e metadados do documento Markdown."""
        now_obj: datetime = datetime.now()
        current_date_str: str = f"{now_obj.day} de {MONTHS_PT_DICT[str(now_obj.month)]} de {now_obj.year}"

        header_str: str = f"# Fonte RAG: {meta_dict['title']}\n\n"
        header_str += "## Metadados do Documento\n"
        header_str += f"- **ID:** {meta_dict['video_id']}\n"
        header_str += f"- **Data da Transcrição:** {current_date_str}\n"
        header_str += f"- **Data do Evento:** {meta_dict['event_date']}\n"
        header_str += f"- **Assunto Principal:** {', '.join(entities_list) if entities_list else 'Conteúdo Geral'}\n"
        header_str += "- **Público-Alvo:** Líderes, Ekklezia, Mesa do Conselho.\n"
        header_str += f"- **Terminologia Chave:** {', '.join(keywords_list)}\n\n"
        return header_str

    def _format_section(self, i_int: int, section_str: str) -> str:
        """Formata uma única seção com tags e título heurístico."""
        sec_keywords_list: List[str] = [kw[0] for kw in self.sec_kw_extractor_obj.extract_keywords(section_str)]
        sec_title_str: str = f"Seção {i_int}: " + (sec_keywords_list[0].capitalize() if sec_keywords_list else "Desenvolvimento")
        
        output_str: str = f"### {sec_title_str}\n"
        output_str += f"**Tags:** {' '.join(['#'+kw.replace(' ', '') for kw in sec_keywords_list])}\n\n"
        output_str += f"{section_str.strip()}\n\n"
        return output_str

    def _generate_markdown_output(self, meta_dict: Dict[str, str], entities_list: List[str], keywords_list: List[str], sections_list: List[str]) -> str:
        """Monta o documento Markdown final unificando cabeçalho e seções."""
        output_str: str = self._generate_markdown_header(meta_dict, entities_list, keywords_list)
        output_str += "## Seções Temáticas\n"
        for i_int, section_str in enumerate(sections_list, 1):
            output_str += self._format_section(i_int, section_str)
        return output_str

    def process(self, text_str: str, meta_dict: Dict[str, str]) -> str:
        """
        Executa o pipeline completo:
        1. Limpa palavras irrelevantes (vícios de linguagem).
        2. Analisa o texto com spaCy.
        3. Extrai palavras-chave e entidades (pessoas, lugares, organizações).
        4. Divide o texto em blocos lógicos.
        5. Monta o documento Markdown final.
        """
        text_str = self.clean_filler_words(text_str)
        doc_obj: Any = self.nlp_obj(text_str)
        
        keywords_list: List[str] = self._extract_keywords(text_str)
        entities_list: List[str] = self._extract_entities(doc_obj)
        sections_list: List[str] = self._segment_content(doc_obj)

        return self._generate_markdown_output(meta_dict, entities_list, keywords_list, sections_list)

def get_files_to_process(input_dir_path: str, specific_files_list: Optional[List[str]] = None) -> List[str]:
    """Filters files in the directory based on extension and exclusion list."""
    all_files_list: List[str] = [
        f for f in os.listdir(input_dir_path)
        if (f.endswith('.txt') or f.endswith('.srt'))
        and "_rag" not in f
        and f not in EXCLUDED_FILES_SET
        and not f.startswith(".")
    ]
    
    if specific_files_list:
        return [f for f in all_files_list if any(p in f for p in specific_files_list)]
    return all_files_list

def _read_file_content(file_path_str: str) -> str:
    """Lê e realiza limpeza inicial do conteúdo do arquivo."""
    filename_str: str = os.path.basename(file_path_str)
    with open(file_path_str, 'r', encoding='utf-8') as f_obj:
        content_str: str = f_obj.read()

    if filename_str.lower().endswith('.srt'):
        content_str = clean_srt_content(content_str)
    return content_str

def _save_rag_result(output_path_str: str, final_text_str: str, original_content_str: str) -> None:
    """Salva o resultado final e a transcrição original."""
    with open(output_path_str, 'w', encoding='utf-8') as f_obj:
        f_obj.write(final_text_str)
        f_obj.write("\n\n---\n\n## Transcrição Completa Original\n\n")
        f_obj.write(original_content_str)

def process_single_file(
    filename_str: str,
    input_dir_path: str,
    output_dir_path: str,
    rules_list: List[Dict[str, Any]],
    processor_obj: HeuristicProcessor
) -> None:
    """Processa um único arquivo de transcrição e salva a versão RAG."""
    try:
        file_path_str: str = os.path.join(input_dir_path, filename_str)
        name_str, _ = os.path.splitext(filename_str)
        output_path_str: str = os.path.join(output_dir_path, f"{name_str}_rag.txt")

        logger.info(f"Processing: {filename_str}")
        
        content_str: str = _read_file_content(file_path_str)
        meta_dict: Dict[str, str] = extract_metadata_from_filename(filename_str)
        
        processed_text_str: str = processor_obj.process(content_str, meta_dict)
        final_text_str: str = enforce_terminology(processed_text_str, rules_list)
        
        _save_rag_result(output_path_str, final_text_str, content_str)
        logger.info(f"Saved: {output_path_str}")
    except Exception as error_obj:
        logger.error(f"Error processing {filename_str}: {error_obj}")

def _parse_arguments() -> argparse.Namespace:
    """Configura e analisa argumentos de linha de comando."""
    parser_obj: argparse.ArgumentParser = argparse.ArgumentParser()
    parser_obj.add_argument("--dir", default=".", help="Input directory")
    parser_obj.add_argument("--output", help="Output directory")
    parser_obj.add_argument("--rules", default="rules.txt", help="Rules file")
    parser_obj.add_argument("--files", nargs='+', help="Specific files to process")
    return parser_obj.parse_args()

def main() -> None:
    """Ponto de entrada coordenador do script heurístico local."""
    args_obj: argparse.Namespace = _parse_arguments()

    input_dir_path: str = args_obj.dir
    output_dir_path: str = args_obj.output if args_obj.output else input_dir_path
    os.makedirs(output_dir_path, exist_ok=True)

    rules_list: List[Dict[str, Any]] = load_rules(args_obj.rules)
    processor_obj: HeuristicProcessor = HeuristicProcessor()
    
    files_to_process_list: List[str] = get_files_to_process(input_dir_path, args_obj.files)
    if not files_to_process_list:
        logger.info("No files found to process.")
        return

    logger.info(f"Processing {len(files_to_process_list)} files heuristically...")
    for filename_str in files_to_process_list:
        process_single_file(filename_str, input_dir_path, output_dir_path, rules_list, processor_obj)

if __name__ == "__main__":
    main()
