# RAG Processor

Este projeto automatiza a adaptação de transcrições de vídeo (YouTube) para um formato otimizado para RAG (Retrieval-Augmented Generation), utilizando a API do Google Gemini.

## Funcionalidades

- **Batch Processing**: Processa automaticamente múltiplos arquivos `.txt` em um diretório.
- **Adaptação Semântica**: Estrutura o conteúdo em Markdown com metadados, tags e seções temáticas.
- **Salvaguardas de Terminologia**: Garante a aplicação de regras específicas:
  - "Sete Montanhas" &rarr; **Sete Montes**
  - "Ecclesia" &rarr; **Ekklezia**
### 5. Verificação de Idempotência
- Implementada verificação de existência do arquivo `.md` antes de processar.
- O script agora exibe `Pulando: [arquivo]` quando o output já existe.
- Testado com sucesso no diretório `nexusmastermind`.

#### Resultados de Teste
```
2026-02-11 11:13:05,099 [INFO] Pulando: Live 1 Transcrição.txt (Output já existe em Live 1 Transcrição_rag_optimized.md)
2026-02-11 11:13:05,099 [INFO] Pulando: Live 2 Transcrição.txt (Output já existe em Live 2 Transcrição_rag_optimized.md)
2026-02-11 11:13:05,099 [INFO] Pulando: Live 3 Transcrição.txt (Output já existe em Live 3 Transcrição_rag_optimized.md)
```
- **Otimização RAG**: Limpeza de ruídos de fala e foco em princípios e estratégias.

## Instalação

1. Clone o repositório ou baixe os arquivos.
2. Crie um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure sua chave de API no arquivo `.env`:
   ```env
   GEMINI_API_KEY=sua_chave_aqui
   ```

## Uso

Execute o script apontando para o diretório que contém as transcrições:

```bash
python main.py --dir /caminho/para/transcricoes
```

Se nenhum diretório for especificado, o script processará o diretório atual:

```bash
python main.py
```

## Uso Global (Alias)

O alias `rag_processor` foi configurado no seu `~/.zshrc`. Para ativá-lo agora, execute:
```bash
source ~/.zshrc
```

Agora você pode processar transcrições de qualquer pasta apenas digitando:
```bash
rag_processor
```
Ou especificando um diretório:
```bash
rag_processor --dir /caminho/para/pasta
```

### Argumentos

- `--dir`: Diretório de entrada contendo arquivos `.txt` (padrão: `.`).
- `--output`: Diretório de saída para os arquivos `.md` (padrão: mesmo da entrada).
- `--workers`: Número de threads simultâneas para processamento (padrão: 3).
- `--rules`: Caminho para o arquivo de regras (padrão: `rules.txt`).

## Personalização de Regras (rules.txt)

Você pode adicionar novas regras de substituição editando o arquivo `rules.txt` de forma simples:

```text
# Termo Original -> Termo Novo
Sete Montanhas -> Sete Montes
Ecclesia -> Ekklezia

# Para usar Regex, use o prefixo REGEX:
REGEX: (?i)palavra-chave -> Substituto
```

- **Comentários**: Linhas que começam com `#` são ignoradas.
- **Substituição Simples**: Usa o formato `De -> Para`.
- **Regex**: Use `REGEX: Padrão -> Substituto` para substituições avançadas.

## Estrutura do Documento Gerado

O script gera arquivos `.md` com o sufixo `_rag_optimized` contendo:
1. Título da Fonte RAG
2. Metadados (ID, Data, Assunto, Público-Alvo, Terminologia)
3. Seções Temáticas com Tags
