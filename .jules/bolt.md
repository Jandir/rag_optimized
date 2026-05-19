## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2026-05-19 - [Dynamic Regex Pre-compilation and Error Handling]
**Learning:** `re.sub()` compiled inline inside processing loops (like dynamic rule application) introduces a performance hit. Furthermore, failing to handle dynamic/user-supplied regex replacements with `try...except` block causes the entire text processor application to crash on invalid syntax.
**Action:** Always eagerly pre-compile regex patterns via `re.compile()` immediately during configuration/rules load. When replacing dynamically provided pattern content with `pattern.sub()`, always wrap the operation in a `try...except` to protect the application from runtime termination due to irregular user inputs.
