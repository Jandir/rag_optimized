#!/usr/bin/env python3
"""
RAG Processor - Video Transcript Transformer
============================================
An automated tool designed to transform raw video transcripts (YouTube/SRT) into 
high-quality, structured Markdown files optimized for RAG (Retrieval-Augmented Generation).

Key Features:
- Structured Enhancement: Leverages Gemini Flash to add metadata, themes, and actionable tags.
- Data Sanitation: Cleans YouTube "rollup" subtitling and handles SRT timestamps.
- Terminology Compliance: Enforces project-specific nomenclature (e.g., Sete Montes, Ekklezia).
- Performance Optimized: Multi-threading for batch processing and efficient API usage.
- Intelligent Metadata: Extracts titles, event dates, and unique video IDs from filenames.
"""

import os
import sys
import re
import argparse
import logging
import time
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

from google import genai
from dotenv import load_dotenv

# --- EDUCATIVO: O QUE ESTE SCRIPT FAZ? ---
# Este script automatiza a transformação de transcrições de vídeo brutas (do YouTube ou arquivos .srt)
# em documentos Markdown estruturados e otimizados para sistemas RAG (Retrieval-Augmented Generation).
# Ele utiliza a API do Gemini Flash da Google para o processamento inteligente do texto.

# --- Constantes e Configuração ---
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

# Silencia logs internos da biblioteca do Google para evitar poluição visual no terminal
logging.getLogger("google.genai").setLevel(logging.WARNING)

SCRIPT_DIR_PATH: str = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR_PATH, '.env'))
load_dotenv(os.path.join(SCRIPT_DIR_PATH, 'to-notion', '.env'))

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("API KEY não encontrada! Verifique o arquivo .env.")
    sys.exit(1)
# --- Funções Auxiliares (Helper Functions) ---

HTML_TAG_PATTERN: re.Pattern = re.compile(r'<[^>]*>')

def _parse_srt_blocks(content_str: str) -> List[str]:
    """Extrai blocos de texto de conteúdo SRT, removendo tags HTML."""
    blocks_list: List[str] = []
    for block_str in content_str.split('\n\n'):
        arrow_idx_int: int = block_str.find('-->')
        if arrow_idx_int != -1:
            eol_idx_int: int = block_str.find('\n', arrow_idx_int)
            if eol_idx_int != -1:
                text_block_str: str = block_str[eol_idx_int + 1:].strip()
                # Remove tags HTML simples como <i> ou <b> que podem vir no .srt
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

        # Caso 1: Repetição simples
        new_part_str: Optional[str] = _handle_simple_repetition(prev_text_str, curr_text_str)
        if new_part_str is not None:
            if new_part_str:
                cleaned_lines_list.append(new_part_str)
            continue

        # Caso 2: Sobreposições parciais
        unique_lines_list: List[str] = _handle_partial_overlap(prev_text_str, curr_text_str)
        cleaned_lines_list.extend(unique_lines_list)

    return cleaned_lines_list

def clean_srt_content(content_str: str) -> str:
    """Limpa arquivos .srt removendo tempos e deduplicando conteúdo rollup."""
    content_str = content_str.replace('\r\n', '\n')
    blocks_list: List[str] = _parse_srt_blocks(content_str)
    cleaned_lines_list: List[str] = _deduplicate_srt_lines(blocks_list)
    return ' '.join(cleaned_lines_list)

def format_duration(seconds_float: float) -> str:
    """Formata segundos em uma string legível (ex: 1h 2m 3s)."""
    if seconds_float < 60:
        return f"{seconds_float:.2f} segundos"
    
    minutes_int = int(seconds_float // 60)
    remaining_seconds_int = int(seconds_float % 60)
    
    if minutes_int < 60:
        return f"{minutes_int}m {remaining_seconds_int}s"
        
    hours_int = int(minutes_int // 60)
    remaining_minutes_int = int(minutes_int % 60)
    return f"{hours_int}h {remaining_minutes_int}m {remaining_seconds_int}s"

def _parse_rule_line(line_str: str) -> Optional[Dict[str, Any]]:
    """Analisa uma única linha do arquivo de regras e retorna um dicionário de regra."""
    line_str = line_str.strip()
    if not line_str or line_str.startswith('#'):
        return None

    is_regex_bool: bool = line_str.startswith('REGEX:')
    if is_regex_bool:
        line_str = line_str[6:].strip()

    if '->' in line_str:
        parts_list: List[str] = line_str.split('->', 1)
        original_str = parts_list[0].strip()
        replacement_str = parts_list[1].strip()

        rule_dict: Dict[str, Any] = {
            "original": original_str,
            "replacement": replacement_str,
            "is_regex": is_regex_bool
        }

        if is_regex_bool:
            # ⚡ Bolt Optimization: Pre-compile user-defined dynamic regex rules during initialization.
            # This avoids the overhead of compiling or relying on Python's re internal cache
            # on every loop iteration inside `enforce_terminology`.
            try:
                rule_dict["compiled_pattern"] = re.compile(original_str)
            except Exception as error_obj:
                logger.error(f"Erro ao compilar regex '{original_str}': {error_obj}")
                return None

        return rule_dict
    return None

def load_rules(rules_path_str: str = "rules.txt") -> List[Dict[str, Any]]:
    """Carrega regras de terminologia de um arquivo de texto de forma modular."""
    absolute_path_str: str = os.path.join(SCRIPT_DIR_PATH, rules_path_str)
    rules_list: List[Dict[str, Any]] = []

    if not os.path.exists(absolute_path_str):
        logger.warning(f"Arquivo de regras não encontrado: {absolute_path_str}")
        return rules_list

    try:
        with open(absolute_path_str, 'r', encoding='utf-8') as f_obj:
            for line_str in f_obj:
                rule_dict = _parse_rule_line(line_str)
                if rule_dict:
                    rules_list.append(rule_dict)
        return rules_list
    except Exception as error_obj:
        logger.error(f"Erro ao carregar regras: {error_obj}")
        return []

def enforce_terminology(text_str: str, rules_list: List[Dict[str, Any]]) -> str:
    """Aplica substituições de termos baseadas nas regras carregadas."""
    for rule_dict in rules_list:
        if rule_dict["is_regex"]:
            if "compiled_pattern" in rule_dict:
                try:
                    text_str = rule_dict["compiled_pattern"].sub(rule_dict["replacement"], text_str)
                except Exception as error_obj:
                    logger.error(f"Erro na substituição Regex '{rule_dict['original']}': {error_obj}")
        else:
            text_str = text_str.replace(rule_dict["original"], rule_dict["replacement"])
    return text_str

def _extract_event_date(clean_name_str: str) -> str:
    """Extrai e formata a data do evento a partir do nome limpo."""
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
    """Extrai o ID do vídeo do YouTube a partir do nome limpo."""
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
        # Extrai mês e ano da string formatada para o título MasterMind
        title_str = f"MasterMind {event_date_str.replace(' de ', ' ')}"

    return {"title": title_str, "event_date": event_date_str, "video_id": video_id_str}

# --- Lógica de IA (Gemini Integration) ---


class GeminiProcessor:
    """
    Gerencia as interações com a API do Google Gemini.
    Responsável por formatar os prompts, enviar o texto e tratar erros/limites de cota.
    """
    
    def __init__(self, api_key_str: str):
        self.client_obj = genai.Client(api_key=api_key_str)
        self.model_name_str = "gemini-2.0-flash" # Atualizado para a versão flash mais recente

    def _get_rag_prompt(self, text_str: str, filename_str: str, title_str: str, current_date_str: str, event_date_str: str, video_id_str: str) -> str:
        """Constrói o prompt detalhado para a IA."""
        return f"""
        Sua missão é adaptar esta transcrição de vídeo para ser uma fonte RAG (Retrieval-Augmented Generation) de alta qualidade.
        
        ESTRUTURA REQUERIDA (Markdown):
        
        1. # Fonte RAG: {title_str}
        
        2. ## Metadados do Documento
        - **ID:** {video_id_str if video_id_str != "N/A" else "[Crie um ID curto, ex: LIVE-00X]"}
        - **Data da Transcrição:** {current_date_str}
        - **Data do Evento:** {event_date_str}
        - **Assunto Principal:** [2-3 temas centrais]
        - **Público-Alvo:** Líderes, Ekklezia, Mesa do Conselho.
        - **Terminologia Chave:** [5-7 palavras-chave separadas por vírgula]
        
        3. ## Seções Temáticas
        Divida o texto em seções lógicas usando:
        ### [Título da Seção]
        **Tags:** #[Tag1] #[Tag2]
        [Conteúdo estruturado, limpo de vícios de linguagem, focado em princípios e estratégias]
        
        REGRAS CRÍTICAS:
        - Mantenha o conteúdo profundo (não resuma demais).
        - Remova redundâncias de fala (saudações repetitivas, ruídos).
        - Use Markdown rigoroso.
        - Mantenha os termos "Sete Montes" e "Ekklezia" sempre que o conteúdo se referir a governo ou igreja.
        
        ARQUIVO ORIGINAL: {filename_str}
        CONTEÚDO:
        {text_str}
        """

    def _prepare_processing_context(self, filename_str: str, meta_dict: Dict[str, str]) -> Dict[str, str]:
        """Prepara as strings de data e prompt para o processamento."""
        now_obj = datetime.now()
        current_date_str: str = f"{now_obj.day} de {MONTHS_PT_DICT[str(now_obj.month)]} de {now_obj.year}"
        return {
            "current_date": current_date_str,
            "event_date": meta_dict['event_date'],
            "title": meta_dict['title'],
            "video_id": meta_dict['video_id']
        }

    def _call_gemini_api(self, prompt_str: str, filename_str: str, max_retries_int: int) -> str:
        """Realiza a chamada à API Gemini com lógica de retentativa para limites de cota."""
        for attempt_int in range(max_retries_int):
            try:
                response_obj = self.client_obj.models.generate_content(
                    model=self.model_name_str,
                    contents=prompt_str
                )
                return response_obj.text
            except Exception as error_obj:
                error_msg_str: str = str(error_obj)
                if "429" in error_msg_str or "quota" in error_msg_str.lower():
                    wait_time_int: int = (attempt_int + 1) * 5
                    logger.warning(f"Limite de API atingido para {filename_str}. Esperando {wait_time_int}s...")
                    time.sleep(wait_time_int)
                else:
                    logger.error(f"Erro na API Gemini para {filename_str}: {error_obj}")
                    if attempt_int == max_retries_int - 1:
                        return ""
                    time.sleep(2)
        return ""

    def process(self, text_str: str, filename_str: str, meta_dict: Dict[str, str], max_retries_int: int = 3) -> str:
        """Orquestra o processamento do texto via Gemini de forma modular."""
        ctx_dict = self._prepare_processing_context(filename_str, meta_dict)
        
        prompt_str: str = self._get_rag_prompt(
            text_str, filename_str, ctx_dict['title'], 
            ctx_dict['current_date'], ctx_dict['event_date'], ctx_dict['video_id']
        )
        
        return self._call_gemini_api(prompt_str, filename_str, max_retries_int)

def get_files_to_process(input_dir_path: str, specific_files_list: Optional[List[str]] = None) -> List[str]:
    """Filtra os arquivos no diretório com base na extensão e na lista de exclusão."""
    all_files_list: List[str] = [
        f for f in os.listdir(input_dir_path)
        if (f.endswith('.txt') or f.endswith('.srt'))
        and "_rag" not in f
        and f not in EXCLUDED_FILES_SET
        and not f.startswith(".")
    ]
    
    if specific_files_list:
        expanded_patterns_set = set()
        for pattern_str in specific_files_list:
            glob_path_str = os.path.join(input_dir_path, pattern_str)
            matches_list = glob.glob(glob_path_str)
            if matches_list:
                 for match_str in matches_list:
                      expanded_patterns_set.add(os.path.basename(match_str))
            else:
                 expanded_patterns_set.add(pattern_str)
        return [f for f in all_files_list if f in expanded_patterns_set]
        
    return all_files_list

def _read_file_content(file_path_str: str) -> str:
    """Lê e realiza limpeza inicial do conteúdo do arquivo."""
    filename_str: str = os.path.basename(file_path_str)
    with open(file_path_str, 'r', encoding='utf-8') as f_obj:
        content_str: str = f_obj.read()

    if filename_str.lower().endswith('.srt'):
        logger.info(f"Limpando SRT: {filename_str}")
        content_str = clean_srt_content(content_str)
    return content_str

def _get_output_path(file_path_str: str, output_dir_path: str) -> str:
    """Gera o caminho do arquivo de saída."""
    filename_str: str = os.path.basename(file_path_str)
    name_str, _ = os.path.splitext(filename_str)
    return os.path.join(output_dir_path, f"{name_str}_rag.txt")

def _save_rag_result(output_path_str: str, final_text_str: str, original_content_str: str) -> None:
    """Salva o resultado final e a transcrição original."""
    with open(output_path_str, 'w', encoding='utf-8') as f_obj:
        f_obj.write(final_text_str)
        f_obj.write("\n\n---\n\n## Transcrição Completa Original\n\n")
        f_obj.write(original_content_str)

def process_single_file(
    file_path_str: str,
    output_dir_path: str,
    rules_list: List[Dict[str, Any]],
    processor_obj: GeminiProcessor
) -> None:
    """Processa um único arquivo (.txt ou .srt) de forma modular."""
    filename_str: str = os.path.basename(file_path_str)
    output_path_str: str = _get_output_path(file_path_str, output_dir_path)

    # Idempotência: pula se já processado
    if os.path.exists(output_path_str):
        logger.info(f"Pulando: {filename_str} (Já processado)")
        return

    try:
        content_str: str = _read_file_content(file_path_str)
        if not content_str.strip():
            logger.warning(f"Arquivo vazio: {filename_str}")
            return

        meta_dict: Dict[str, str] = extract_metadata_from_filename(filename_str)

        # Processamento com IA
        optimized_text_str: str = processor_obj.process(content_str, filename_str, meta_dict)
        if not optimized_text_str:
            logger.error(f"IA falhou em {filename_str}")
            return

        # Aplica regras de Terminologia e salva
        final_text_str: str = enforce_terminology(optimized_text_str, rules_list)
        _save_rag_result(output_path_str, final_text_str, content_str)

        logger.info(f"Salvo: {output_path_str}")

    except Exception as error_obj:
        logger.error(f"Erro ao processar {filename_str}: {error_obj}")

# --- Ponto de Entrada (Main) ---

def _parse_arguments() -> argparse.Namespace:
    """Configura e analisa os argumentos de linha de comando."""
    parser_obj = argparse.ArgumentParser(description="Processador de Transcrições para RAG (Gemini).")
    parser_obj.add_argument("--dir", default=".", help="Diretório de entrada")
    parser_obj.add_argument("--output", help="Diretório de saída")
    parser_obj.add_argument("--workers", type=int, default=3, help="Número de threads simultâneas")
    parser_obj.add_argument("--rules", default="rules.txt", help="Arquivo de regras")
    parser_obj.add_argument("--files", nargs='+', help="Filtros de arquivos específicos")
    return parser_obj.parse_args()

def _orchestrate_parallel_processing(
    input_dir_path: str,
    output_dir_path: str,
    files_list: List[str],
    workers_int: int,
    rules_list: List[Dict[str, Any]],
    processor_obj: GeminiProcessor
) -> None:
    """Gerencia o pool de threads para processamento paralelo."""
    total_files_int: int = len(files_list)
    with ThreadPoolExecutor(max_workers=workers_int) as executor:
        futures_dict: Dict[Any, str] = {
            executor.submit(
                process_single_file,
                os.path.join(input_dir_path, f),
                output_dir_path,
                rules_list,
                processor_obj
            ): f for f in files_list
        }

        for i_int, future_obj in enumerate(as_completed(futures_dict), 1):
            filename_str: str = futures_dict[future_obj]
            try:
                future_obj.result()
                logger.info(f"[{i_int}/{total_files_int}] Concluído: {filename_str}")
            except Exception as error_obj:
                logger.error(f"[{i_int}/{total_files_int}] Falha crítica para {filename_str}: {error_obj}")

def main() -> None:
    """Ponto de entrada coordenador do script."""
    args_obj: argparse.Namespace = _parse_arguments()

    input_dir_path: str = args_obj.dir
    if not os.path.exists(input_dir_path):
        logger.error(f"Diretório não encontrado: {input_dir_path}")
        return

    output_dir_path: str = args_obj.output if args_obj.output else input_dir_path
    os.makedirs(output_dir_path, exist_ok=True)

    files_to_process_list: List[str] = get_files_to_process(input_dir_path, args_obj.files)
    if not files_to_process_list:
        logger.info("Nenhuma transcrição nova encontrada.")
        return

    logger.info(f"Encontrados {len(files_to_process_list)} arquivos. Processando com {args_obj.workers} workers...")

    rules_list: List[Dict[str, Any]] = load_rules(args_obj.rules)
    processor_obj: GeminiProcessor = GeminiProcessor(GEMINI_API_KEY)

    start_time_float: float = time.time()
    _orchestrate_parallel_processing(
        input_dir_path, output_dir_path, files_to_process_list,
        args_obj.workers, rules_list, processor_obj
    )

    elapsed_float: float = time.time() - start_time_float
    logger.info(f"Lote concluído em {format_duration(elapsed_float)}.")

if __name__ == "__main__":
    main()
