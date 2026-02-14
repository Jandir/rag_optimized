#!/usr/bin/env python3
"""
Script: rag_processor.py
Description: Automates the adaptation of video transcripts for RAG (Retrieval-Augmented Generation).
             Uses Gemini Flash to structure content, add metadata, and enforce terminology rules.
             Optimized with parallel processing, API client reuse, and external rules configuration.

Terminology Rules:
- Loaded from rules.txt (Sete Montes, Ekklezia, etc.)
"""

import os
import sys
import re
import argparse
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from google import genai
from dotenv import load_dotenv

# --- Helper Functions ---

def clean_srt_content(content: str) -> str:
    """
    Remove timestamps and deduplicate lines common in "rollup" subtitles (Youtube).
    Adapted from lexis-chunk.py.
    """
    # Normalize line breaks
    content = content.replace('\r\n', '\n')
    
    # Regex to identify subtitle blocks:
    # Number
    # Timestamp --> Timestamp
    # Text... (can be multiple lines)
    # \n (separator blank line)
    
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:(?!\n\n).)*?)(?=\n\n|$)', re.DOTALL)
    
    blocks = []
    for match in pattern.finditer(content):
        text_block = match.group(4).strip()
        
        # Clean HTML tags
        text_block = re.sub(r'<[^>]*>', '', text_block)
        
        if text_block:
            blocks.append(text_block)

    # Logical Deduplication
    cleaned_lines = []
    if blocks:
        # Add the first complete block
        current_text = blocks[0]
        cleaned_lines.append(current_text)
        
        for i in range(1, len(blocks)):
            prev_text = blocks[i-1]
            curr_text = blocks[i]
            
            # Case 1: Current block starts with the previous block (e.g. Prev="A", Curr="A\nB")
            # We want only "B".
            if curr_text.startswith(prev_text):
                new_part = curr_text[len(prev_text):].strip()
                if new_part:
                    cleaned_lines.append(new_part)
                continue
                
            # Case 2: Line by line strategy for "A\nB" -> "B\nC"
            prev_lines = [l.strip() for l in prev_text.split('\n') if l.strip()]
            curr_lines = [l.strip() for l in curr_text.split('\n') if l.strip()]
            
            start_idx = 0
            if prev_lines and curr_lines:
                if curr_lines[0] == prev_lines[-1]:
                    start_idx = 1
                elif len(prev_lines) < len(curr_lines) and curr_lines[:len(prev_lines)] == prev_lines:
                    start_idx = len(prev_lines)

            for j in range(start_idx, len(curr_lines)):
                cleaned_lines.append(curr_lines[j])

    return ' '.join(cleaned_lines)

def format_duration(seconds: float) -> str:
    """Formats duration into human readable string."""
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
        
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Silence Google GenAI library logs (only show warnings/errors)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# Load Environment Variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))
load_dotenv(os.path.join(script_dir, 'to-notion', '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# --- Rules Loading ---

def load_rules(rules_path: str = "rules.txt") -> List[Dict[str, Any]]:
    """Loads terminology rules from a text file (Original -> Replacement)."""
    absolute_path = os.path.join(script_dir, rules_path)
    rules = []
    if not os.path.exists(absolute_path):
        logger.warning(f"Arquivo de regras não encontrado: {absolute_path}. Usando regras vazias.")
        return rules
    
    try:
        with open(absolute_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                is_regex = False
                if line.startswith('REGEX:'):
                    is_regex = True
                    line = line[6:].strip()
                
                if '->' in line:
                    parts = line.split('->', 1)
                    original = parts[0].strip()
                    replacement = parts[1].strip()
                    rules.append({
                        "original": original,
                        "replacement": replacement,
                        "is_regex": is_regex
                    })
        return rules
    except Exception as e:
        logger.error(f"Erro ao carregar {rules_path}: {e}")
        return []

def enforce_terminology(text: str, rules: List[Dict[str, Any]]) -> str:
    """Enforces nomenclature rules loaded from configuration."""
    for rule in rules:
        original = rule["original"]
        replacement = rule["replacement"]
        if rule["is_regex"]:
            try:
                text = re.sub(original, replacement, text)
            except Exception as e:
                logger.error(f"Erro na regex '{original}': {e}")
        else:
            text = text.replace(original, replacement)
            
    return text

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extracts title and event date from filename patterns."""
    clean_name = filename.replace(" Transcrição.txt", "").replace(".txt", "")
    clean_name = clean_name.replace(" Transcrição.srt", "").replace(".srt", "")
    clean_name = clean_name.strip()
    
    # Try to find date patterns like "Jan 2026"
    date_match = re.search(r'(Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)\s+(\d{4})', clean_name, re.I)
    
    months_map = {
        "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
        "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
        "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro"
    }
    
    title = clean_name
    event_date = "N/A"
    
    if date_match:
        month_abbr = date_match.group(1).capitalize()
        year = date_match.group(2)
        # Use first 3 letters for mapping
        key = month_abbr[:3]
        if key == "Mai": key = "Mai" # Ensure Maio/Mai works
        full_month = months_map.get(key, month_abbr)
        event_date = f"{full_month} de {year}"
        
        if "MasterMind" in clean_name:
             title = f"MasterMind {full_month} {year}"
    
    return {"title": title, "event_date": event_date}

# --- Core Logic ---

def get_rag_prompt(text: str, filename: str, title: str, current_date: str, event_date: str) -> str:
    """Returns the structured prompt for Gemini."""
    return f"""
    Sua missão é adaptar esta transcrição de vídeo para ser uma fonte RAG (Retrieval-Augmented Generation) de alta qualidade.
    
    ESTRUTURA REQUERIDA (Markdown):
    
    1. # Fonte RAG: {title}
    
    2. ## Metadados do Documento
    - **ID:** [Crie um ID curto, ex: LIVE-00X]
    - **Data da Transcrição:** {current_date}
    - **Data do Evento:** {event_date}
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
    
    ARQUIVO ORIGINAL: {filename}
    CONTEÚDO:
    {text}
    """

def process_with_gemini(client: genai.Client, text: str, filename: str, title: str, current_date: str, event_date: str, max_retries: int = 3) -> str:
    """Uses Gemini 1.5 Flash to structure the transcript for RAG with retry logic."""
    prompt = get_rag_prompt(text, filename, title, current_date, event_date)
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = (attempt + 1) * 5
                logger.warning(f"Rate limit atingido para {filename}. Aguardando {wait_time}s (Tentativa {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                logger.error(f"Erro na API do Gemini para {filename}: {e}")
                if attempt == max_retries - 1:
                    return ""
                time.sleep(2)
    return ""

def process_file(client: genai.Client, file_path: str, output_dir: str, rules: Dict[str, Any]):
    """Processes a single .txt or .srt file and saves the .md result."""
    filename = os.path.basename(file_path)
    
    # Check if output already exists (Idempotency)
    name, _ = os.path.splitext(filename)
    output_filename = f"{name}_rag.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        logger.info(f"Pulando: {filename} (Output já existe)")
        return

    logger.info(f"Iniciando processamento: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if filename.lower().endswith('.srt'):
            logger.info(f"Convertendo .srt para texto limpo: {filename}")
            content = clean_srt_content(content)
        
        if not content.strip():
            logger.warning(f"Arquivo vazio: {filename}")
            return

        # 0. Prep Metadata
        meta = extract_metadata_from_filename(filename)
        current_date = datetime.now().strftime("%d de %B de %Y")
        # Handle locale-specific month if possible, but for simplicity we can use a map or stick to system
        # Actually, let's just use manual month mapping for current_date to be safe with user's PT-BR preference
        months_pt = {
            1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
            7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
        }
        now = datetime.now()
        current_date_str = f"{now.day} de {months_pt[now.month]} de {now.year}"

        # 1. Gemini Processing
        optimized_text = process_with_gemini(
            client, content, filename, 
            meta['title'], current_date_str, meta['event_date']
        )
        
        if not optimized_text:
            logger.error(f"Falha ao gerar conteúdo para {filename}")
            return
            
        # 2. Terminology Enforcement (Dynamic Rules)
        final_text = enforce_terminology(optimized_text, rules)
        
        # 3. Save Markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
            f.write("\n\n---\n\n## Transcrição Completa Original\n\n")
            f.write(content)
            
        # logger.info(f"Sucesso: {output_filename}")
        
    except Exception as e:
        logger.error(f"Erro ao processar arquivo {filename}: {e}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Processador de Transcrições para RAG.")
    parser.add_argument("--dir", default=".", help="Diretório contendo os arquivos .txt ou .srt")
    parser.add_argument("--output", help="Diretório de saída (padrão: mesmo da entrada)")
    parser.add_argument("--workers", type=int, default=3, help="Threads simultâneas (padrão: 3)")
    parser.add_argument("--rules", default="rules.txt", help="Arquivo de regras (padrão: rules.txt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        logger.error(f"Diretório não encontrado: {args.dir}")
        return
        
    output_dir = args.output if args.output else args.dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Filter out known config/system files
    excluded_files = {"historico.txt","cookies.txt", "requirements.txt", "rules.txt", "LICENSE", "README.md"}
    files = [
        f for f in os.listdir(args.dir) 
        if (f.endswith('.txt') or f.endswith('.srt'))
        and "_rag" not in f 
        and f not in excluded_files
        and not f.startswith(".")
    ]
    
    if not files:
        logger.info("Nenhuma transcrição encontrada para processar.")
        return
        
    logger.info(f"Encontrados {len(files)} arquivos. Iniciando processamento paralelo ({args.workers} workers)...")
    
    # Load terminology rules once
    rules = load_rules(args.rules)
    
    # Initialize shared Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    start_time = time.time()
    
    total_files = len(files)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, client, os.path.join(args.dir, f), output_dir, rules): f for f in files}
        
        for i, future in enumerate(as_completed(futures), 1):
            filename = futures[future]
            try:
                future.result()
                logger.info(f"[{i}/{total_files}] Concluído: {filename}")
            except Exception as e:
                logger.error(f"[{i}/{total_files}] Falha no worker para {filename}: {e}")
                
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Processamento de lote concluído em {format_duration(elapsed)}.")

if __name__ == "__main__":
    main()
