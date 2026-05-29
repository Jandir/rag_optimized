## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2026-05-21 - [Pre-compile dynamic user regexes]
**Learning:** Using `re.sub(pattern, repl, string)` inside loops for dynamic configuration-based rules is a performance trap because Python dynamically compiles the regex pattern into a cached object on every iteration if it's not pre-compiled, causing a significant slowdown when processing multiple text blocks or files.
**Action:** When applying dynamic rules loaded from configuration or user input (like `rules.txt`), pre-compile them during the initialization/loading phase using `re.compile(pattern)` (handling compilation errors with try/except), and use `pattern_obj.sub(repl, string)` during the loop processing phase.

## 2024-05-29 - Optimize spaCy Loading
**Learning:** Loading the full spaCy pipeline (`pt_core_news_sm`) is computationally expensive and uses significant memory. When we only need specific features like Named Entity Recognition (NER) and sentence boundaries, the default pipeline includes unnecessary components like `tagger`, `morphologizer`, `lemmatizer`, `attribute_ruler`, and `parser`.
**Action:** Always disable unused components in `spacy.load()` using the `disable` parameter. Use lightweight alternatives, such as `nlp.add_pipe("sentencizer")`, to maintain sentence segmentation functionality while significantly reducing initialization time and memory footprint.
