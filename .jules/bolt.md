## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2026-05-21 - [Pre-compile dynamic user regexes]
**Learning:** Using `re.sub(pattern, repl, string)` inside loops for dynamic configuration-based rules is a performance trap because Python dynamically compiles the regex pattern into a cached object on every iteration if it's not pre-compiled, causing a significant slowdown when processing multiple text blocks or files.
**Action:** When applying dynamic rules loaded from configuration or user input (like `rules.txt`), pre-compile them during the initialization/loading phase using `re.compile(pattern)` (handling compilation errors with try/except), and use `pattern_obj.sub(repl, string)` during the loop processing phase.

## 2026-05-22 - [Disable unused spaCy components]
**Learning:** By default, `spacy.load()` instantiates the entire NLP pipeline, including memory-intensive and slow components like `parser`, `tagger`, and `lemmatizer`. If an application only needs specific functions (like Named Entity Recognition (`NER`) and sentence segmentation (`sents`)), loading the full model creates unnecessary overhead.
**Action:** Always inspect which spaCy features are actually used. Use the `disable` parameter in `spacy.load` (e.g., `disable=["tagger", "parser", ...]`) to turn off unused components, and add lightweight alternatives (like `nlp.add_pipe("sentencizer")` for fast sentence splitting) to drastically speed up processing and reduce memory usage.
