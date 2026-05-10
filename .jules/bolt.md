## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2024-05-10 - [Precompiling Terminology Rules Regex & Date Regex]
**Learning:** Calling `re.sub(pattern, replacement, text)` with string patterns inside loops or frequently called functions (like `enforce_terminology`) compiles the regex on every execution, adding unnecessary overhead.
**Action:** When loading rules or configuration that define regexes, pre-compile them using `re.compile()` immediately upon loading. Then use the pre-compiled `pattern.sub()` method in the execution loop to avoid repeated compilation. Similarly, use module-level constant pre-compilations (e.g., `DATE_EXTRACT_PATTERN = re.compile(...)`) instead of `re.search(...)` with string patterns inline.
