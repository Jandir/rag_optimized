## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2024-05-18 - [Optimization] Pre-compiling dynamic Regex rules
**Learning:** Compiling dynamic regular expressions on the fly (inside a loop) causes severe performance bottlenecks, especially when applying multiple rules across large documents (like video transcripts).
**Action:** When working with dynamic regex rules loaded from configuration files, always pre-compile them during initialization (`re.compile`) and store the compiled patterns in memory. Furthermore, when executing `pattern.sub()`, wrap it in a `try...except` block to prevent failures from user-defined malformed replacement strings (such as invalid group references) at runtime.

## 2025-02-27 - [Avoiding Dictionary Re-instantiation in Loops]
**Learning:** Defining static dictionaries (like months mappings) locally inside functions that are called repeatedly (e.g. `extract_metadata_from_filename` or during a file processing loop) incurs significant overhead due to constant re-instantiation in memory.
**Action:** Always define static mapping dictionaries at the module/global level to ensure they are created once, saving memory allocations and processing time, especially during bulk operations.
