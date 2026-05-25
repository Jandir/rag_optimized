## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2024-05-18 - [Optimization] Pre-compiling dynamic Regex rules
**Learning:** Compiling dynamic regular expressions on the fly (inside a loop) causes severe performance bottlenecks, especially when applying multiple rules across large documents (like video transcripts).
**Action:** When working with dynamic regex rules loaded from configuration files, always pre-compile them during initialization (`re.compile`) and store the compiled patterns in memory. Furthermore, when executing `pattern.sub()`, wrap it in a `try...except` block to prevent failures from user-defined malformed replacement strings (such as invalid group references) at runtime.

## 2024-05-30 - [Performance Anti-Pattern: Dynamic Regex Compilation and Dictionary Recreation in Loops]
**Learning:** Compiling regex patterns (e.g., `re.sub(pattern, replacement, text)`) or re-instantiating static dictionaries inside frequently called functions (like per-file processors) creates significant, avoidable overhead. Benchmarks showed a measurable difference between dynamic execution and pre-compilation of regular expressions in this codebase.
**Action:** Always pre-compile regex rules (`re.compile`) during application startup or configuration loading (e.g., in `load_rules`). Apply using `compiled_pattern.sub()` instead. Additionally, move static mapping dictionaries (like month translations) to module-level constants to avoid recreation during runtime iterations over files.
