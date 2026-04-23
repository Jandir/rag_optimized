# RAG Processor 🚀

Este repositório contém ferramentas para transformar transcrições brutas de vídeos em documentos estruturados e semanticamente ricos, otimizados para sistemas de RAG (Retrieval-Augmented Generation).

Existem duas versões do processador disponíveis:
1.  **Versão LLM (`rag_processor.py`)**: Utiliza a API do Google Gemini para estruturação avançada.
2.  **Versão Local (`rag_processor_local.py`)**: Utiliza Processamento de Linguagem Natural (NLP) heurístico local, sem custos de API.

---

## ❓ Por que usar este script?

Transcrições diretas de áudio (`.txt` ou `.srt`) geralmente apresentam problemas para IA:
- **Ruído**: Vícios de linguagem e repetições.
- **Terminologia Inconsistente**: Termos específicos (ex: "Ekklezia", "Sete Montes") podem ser mal interpretados.
- **Falta de Estrutura**: Dificulta a recuperação precisa de informações.

---

## ✨ Funcionalidades (Versão Local Heurística)

A versão local foi recentemente refatorada para seguir padrões de **alta qualidade de código** e **pedagogia**:

- **🧠 Inteligência Local**:
    - **spaCy**: Identifica entidades (Pessoas, Lugares, Organizações).
    - **YAKE**: Extrai automaticamente as palavras-chave mais importantes.
    - **TextTiling**: Segmenta o texto em seções baseadas em mudanças de tópico.
- **🧹 Limpeza Avançada**: Algoritmo especializado para legendas "rollup" (remove repetições e timestamps).
- **📚 Padrão Ekklezia**: Código escrito com nomenclatura tipada (ex: `video_id_str`, `rules_list`) para máxima clareza.
- **📖 Comentários Educativos**: O código contém explicações detalhadas para programadores iniciantes.
- **🧪 Suíte de Testes**: Inclui testes unitários automatizados para garantir a integridade das funções de limpeza e metadados.

---

## 🚀 Como Usar

### Instalação

1. Clone o repositório.
2. Crie e ative o ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   # Instale o modelo de português do spaCy
   python3 -m spacy download pt_core_news_sm
   ```

### Execução da Versão Local

```bash
# Processa todos os arquivos da pasta atual
python3 rag_processor_local.py

# Processa arquivos de uma pasta específica e define saída
python3 rag_processor_local.py --dir ./transcricoes --output ./processados
```

---

## 🧪 Testes Unitários

Para garantir que a lógica de limpeza de SRT e extração de metadados esteja funcionando:

```bash
PYTHONPATH=. ./.venv/bin/python3 tests/test_rag_processor.py
```

---

## ⚙️ Configuração de Regras (rules.txt)

O arquivo `rules.txt` permite definir substituições de termos. Suporta Regex:

```text
# Formato: Termo Errado -> Termo Correto
Sete Montanhas -> Sete Montes
Ecclesia -> Ekklezia

# Regex (inicie com REGEX:)
REGEX: (?i)vixe\s+maria -> caramba
```

---

## 📄 Estrutura do Arquivo Gerado (`_rag.txt`)

O output é um Markdown contendo:
1.  **Metadados**: Título, Data, ID do Vídeo, Assunto Principal.
2.  **Seções Temáticas**: Texto dividido por tópicos com Tags sugeridas.
3.  **Transcrição Original**: O texto bruto completo ao final para referência.
