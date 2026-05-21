## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.
## 2024-11-20 - [Avoid Dictionary Instantiation in Loops/Functions]
**Learning:** Re-instantiating static dictionaries inside functions or loops causes unnecessary overhead and is a minor anti-pattern.
**Action:** Move static dictionaries to the module level to reuse them across multiple calls, improving performance.
