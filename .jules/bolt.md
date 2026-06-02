## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2026-05-21 - [Pre-compile dynamic user regexes]
**Learning:** Using `re.sub(pattern, repl, string)` inside loops for dynamic configuration-based rules is a performance trap because Python dynamically compiles the regex pattern into a cached object on every iteration if it's not pre-compiled, causing a significant slowdown when processing multiple text blocks or files.
**Action:** When applying dynamic rules loaded from configuration or user input (like `rules.txt`), pre-compile them during the initialization/loading phase using `re.compile(pattern)` (handling compilation errors with try/except), and use `pattern_obj.sub(repl, string)` during the loop processing phase.

## 2026-06-02 - Optimize spaCy NLP Loading
**Learning:** Instantiating full spaCy language models (like `pt_core_news_sm`) is extremely expensive and memory-intensive. Most components like parsers, taggers, and lemmatizers are not always needed for simple Entity Extraction (NER) and Sentence Segmentation.
**Action:** Always disable unused spaCy pipeline components via the `disable` parameter (`disable=["tagger", "morphologizer", "lemmatizer", "attribute_ruler", "parser"]`) to optimize execution speed. Use the lightweight `add_pipe("sentencizer")` as an alternative if only sentence boundaries are needed.
