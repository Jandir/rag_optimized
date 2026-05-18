## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2024-05-18 - [Dynamic Regex Compilation in Loops]
**Learning:** Re-compiling dynamic regular expressions on every iteration inside a text processing loop (like enforcing terminology rules via `re.sub`) is a major performance bottleneck, especially when processing large transcripts with multiple rules.
**Action:** Always pre-compile dynamic regular expressions during the initialization/loading phase (e.g., when loading rules from a file) and store the compiled regex objects. Use a `try...except` block during both compilation and application to gracefully handle syntax errors in user-defined patterns without crashing the processing loop.
