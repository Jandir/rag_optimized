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

# ⚡ BOLT OPTIMIZATION: Pre-compile regexes at module level to avoid repeated compilation in loops
SRT_BLOCK_PATTERN = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*?)(?=\n\n|$)', re.DOTALL)
HTML_TAG_PATTERN = re.compile(r'<[^>]*>')
DATE_EXTRACT_PATTERN = re.compile(r'(Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)\s+(\d{4})', re.I)
VIDEO_ID_PATTERN = re.compile(r'(?:\[|[-_])([a-zA-Z0-9_-]{11})(?:\])?$')
FILLER_WORDS_PATTERN = re.compile(r'\bne\b|\bentão\b|\btipo\b|\bsabe\b|\bpra\b|\btá\b|\bgente\b', re.IGNORECASE)

def _parse_srt_blocks(content_str: str) -> List[str]:
    """Extracts text blocks from SRT content, removing tags."""
    blocks_list: List[str] = []
    for match_obj in SRT_BLOCK_PATTERN.finditer(content_str):
        text_block_str: str = match_obj.group(4).strip()
        text_block_str = HTML_TAG_PATTERN.sub('', text_block_str)
        if text_block_str:
            blocks_list.append(text_block_str)
    return blocks_list

def _deduplicate_srt_lines(blocks_list: List[str]) -> List[str]:
    """Handles the complex deduplication logic for rollup subtitles."""
    cleaned_lines_list: List[str] = []
    if not blocks_list:
        return cleaned_lines_list

    cleaned_lines_list.append(blocks_list[0])
    for i_int in range(1, len(blocks_list)):
        prev_text_str: str = blocks_list[i_int - 1]
        curr_text_str: str = blocks_list[i_int]

        if curr_text_str.startswith(prev_text_str):
            new_part_str: str = curr_text_str[len(prev_text_str):].strip()
            if new_part_str:
                cleaned_lines_list.append(new_part_str)
            continue

        prev_lines_list: List[str] = [l.strip() for l in prev_text_str.split('\n') if l.strip()]
        curr_lines_list: List[str] = [l.strip() for l in curr_text_str.split('\n') if l.strip()]
        start_idx_int: int = 0

        if prev_lines_list and curr_lines_list:
            if curr_lines_list[0] == prev_lines_list[-1]:
                start_idx_int = 1
            elif len(prev_lines_list) < len(curr_lines_list) and curr_lines_list[:len(prev_lines_list)] == prev_lines_list:
                start_idx_int = len(prev_lines_list)

        for j_int in range(start_idx_int, len(curr_lines_list)):
            cleaned_lines_list.append(curr_lines_list[j_int])

    return cleaned_lines_list

def clean_srt_content(content_str: str) -> str:
    """
    Remove timestamps e limpa a repetição de linhas comum em legendas do tipo 'rollup'.
    Muitas legendas geradas automaticamente repetem o texto anterior à medida que novas palavras aparecem.
    Esta função garante que tenhamos um texto corrido e limpo.
    """
    content_str = content_str.replace('\r\n', '\n')
    # Primeiro, extraímos apenas os blocos de texto, ignorando os números e tempos.
    blocks_list: List[str] = _parse_srt_blocks(content_str)
    # Depois, comparamos os blocos para remover o que está repetido.
    cleaned_lines_list: List[str] = _deduplicate_srt_lines(blocks_list)
    return ' '.join(cleaned_lines_list)

def load_rules(rules_path_str: str = "rules.txt") -> List[Dict[str, Any]]:
    """Loads terminology rules from a text file."""
    absolute_path_str: str = os.path.join(SCRIPT_DIR_PATH, rules_path_str)
    rules_list: List[Dict[str, Any]] = []
    if not os.path.exists(absolute_path_str):
        logger.warning(f"Rules file not found: {absolute_path_str}")
        return rules_list
    try:
        with open(absolute_path_str, 'r', encoding='utf-8') as f_obj:
            for line_str in f_obj:
                line_str = line_str.strip()
                if not line_str or line_str.startswith('#'):
                    continue
                is_regex_bool: bool = line_str.startswith('REGEX:')
                if is_regex_bool:
                    line_str = line_str[6:].strip()
                if '->' in line_str:
                    parts_list: List[str] = line_str.split('->', 1)
                    original_str = parts_list[0].strip()
                    replacement_str = parts_list[1].strip()

                    pattern_obj = None
                    if is_regex_bool:
                        try:
                            pattern_obj = re.compile(original_str)
                        except Exception as e:
                            logger.error(f"Erro ao compilar regex '{original_str}': {e}")
                            continue

                    rules_list.append({
                        "original": original_str,
                        "replacement": replacement_str,
                        "is_regex": is_regex_bool,
                        "pattern": pattern_obj
                    })
        return rules_list
    except Exception as error_obj:
        logger.error(f"Error loading rules: {error_obj}")
        return []

def enforce_terminology(text_str: str, rules_list: List[Dict[str, Any]]) -> str:
    """
    Garante que termos específicos sejam escritos corretamente.
    Lê as regras do arquivo 'rules.txt' e faz substituições (simples ou via Regex).
    Isso é vital para manter a consistência de termos como 'Ekklezia' ou 'Sete Montes'.
    """
    for rule_dict in rules_list:
        if rule_dict["is_regex"]:
            text_str = rule_dict["pattern"].sub(rule_dict["replacement"], text_str)
        else:
            text_str = text_str.replace(rule_dict["original"], rule_dict["replacement"])
    return text_str

def extract_metadata_from_filename(filename_str: str) -> Dict[str, str]:
    """Extracts title, date and video ID from the filename."""
    clean_name_str: str = (
        filename_str.replace(" Transcrição.txt", "")
        .replace(".txt", "")
        .replace(" Transcrição.srt", "")
        .replace(".srt", "")
        .strip()
    )
    date_match_obj: Optional[re.Match] = DATE_EXTRACT_PATTERN.search(clean_name_str)
    
    title_str: str = clean_name_str
    event_date_str: str = "N/A"
    
    if date_match_obj:
        month_abbr_str: str = date_match_obj.group(1).capitalize()[:3]
        year_str: str = date_match_obj.group(2)
        full_month_str: str = MONTHS_PT_DICT.get(month_abbr_str, month_abbr_str)
        event_date_str = f"{full_month_str} de {year_str}"
        if "MasterMind" in clean_name_str:
            title_str = f"MasterMind {full_month_str} {year_str}"
            
    video_id_match_obj: Optional[re.Match] = VIDEO_ID_PATTERN.search(clean_name_str)
    video_id_str: str = video_id_match_obj.group(1) if video_id_match_obj else "N/A"
    
    return {
        "title": title_str,
        "event_date": event_date_str,
        "video_id": video_id_str
    }

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
        # ⚡ BOLT OPTIMIZATION: Initialize section keyword extractor once to prevent repetitive slow instantiations in loops
        self.sec_kw_extractor_obj = yake.KeywordExtractor(lan="pt", n=2, top=3)
        # O TextTiling ajuda a dividir o texto em seções baseadas em mudança de tópico
        self.tt_tokenizer_obj = TextTilingTokenizer()

    def clean_filler_words(self, text_str: str) -> str:
        """Removes common Portuguese filler words via regex."""
        return FILLER_WORDS_PATTERN.sub('', text_str).replace('  ', ' ').strip()

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

    def _generate_markdown_output(self, meta_dict: Dict[str, str], entities_list: List[str], keywords_list: List[str], sections_list: List[str]) -> str:
        """Formats the final RAG-ready markdown."""
        now_obj: datetime = datetime.now()
        current_date_str: str = f"{now_obj.day} de {MONTHS_PT_DICT[str(now_obj.month)]} de {now_obj.year}"

        output_str: str = f"# Fonte RAG: {meta_dict['title']}\n\n"
        output_str += "## Metadados do Documento\n"
        output_str += f"- **ID:** {meta_dict['video_id']}\n"
        output_str += f"- **Data da Transcrição:** {current_date_str}\n"
        output_str += f"- **Data do Evento:** {meta_dict['event_date']}\n"
        output_str += f"- **Assunto Principal:** {', '.join(entities_list) if entities_list else 'Conteúdo Geral'}\n"
        output_str += "- **Público-Alvo:** Líderes, Ekklezia, Mesa do Conselho.\n"
        output_str += f"- **Terminologia Chave:** {', '.join(keywords_list)}\n\n"
        
        output_str += "## Seções Temáticas\n"
        for i_int, section_str in enumerate(sections_list, 1):
            # ⚡ BOLT OPTIMIZATION: Reuse self.sec_kw_extractor_obj instead of re-instantiating YAKE per section
            sec_keywords_list: List[str] = [kw[0] for kw in self.sec_kw_extractor_obj.extract_keywords(section_str)]
            sec_title_str: str = f"Seção {i_int}: " + (sec_keywords_list[0].capitalize() if sec_keywords_list else "Desenvolvimento")
            
            output_str += f"### {sec_title_str}\n"
            output_str += f"**Tags:** {' '.join(['#'+kw.replace(' ', '') for kw in sec_keywords_list])}\n\n"
            output_str += f"{section_str.strip()}\n\n"

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

def process_single_file(
    filename_str: str,
    input_dir_path: str,
    output_dir_path: str,
    rules_list: List[Dict[str, Any]],
    processor_obj: HeuristicProcessor
) -> None:
    """Processes a single transcript file and saves the RAG version."""
    try:
        file_path_str: str = os.path.join(input_dir_path, filename_str)
        logger.info(f"Processing: {filename_str}")
        
        with open(file_path_str, 'r', encoding='utf-8') as f_obj:
            content_str: str = f_obj.read()
        
        if filename_str.lower().endswith('.srt'):
            content_str = clean_srt_content(content_str)
        
        meta_dict: Dict[str, str] = extract_metadata_from_filename(filename_str)
        processed_text_str: str = processor_obj.process(content_str, meta_dict)
        final_text_str: str = enforce_terminology(processed_text_str, rules_list)
        
        name_str: str
        name_str, _ = os.path.splitext(filename_str)
        output_path_str: str = os.path.join(output_dir_path, f"{name_str}_rag.txt")
        
        with open(output_path_str, 'w', encoding='utf-8') as f_obj:
            f_obj.write(final_text_str)
            f_obj.write("\n\n---\n\n## Transcrição Completa Original\n\n")
            f_obj.write(content_str)
            
        logger.info(f"Saved: {output_path_str}")
    except Exception as error_obj:
        logger.error(f"Error processing {filename_str}: {error_obj}")

def main() -> None:
    """
    Ponto de entrada do script. Gerencia os argumentos de linha de comando,
    lista os arquivos e coordena o processamento de cada um.
    """
    parser_obj: argparse.ArgumentParser = argparse.ArgumentParser()
    parser_obj.add_argument("--dir", default=".", help="Input directory")
    parser_obj.add_argument("--output", help="Output directory")
    parser_obj.add_argument("--rules", default="rules.txt", help="Rules file")
    parser_obj.add_argument("--files", nargs='+', help="Specific files to process")
    args_obj: argparse.Namespace = parser_obj.parse_args()

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
