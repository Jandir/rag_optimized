## 2024-05-08 - NLP Processor Initializations
**Learning:** Initializing NLP objects like `yake.KeywordExtractor` and compiling Regex patterns inside loops or frequently called functions adds significant overhead. Python NLP libraries often have high instantiation costs (loading models/stopwords).
**Action:** Always inspect classes for objects that can be instantiated once in `__init__` and reused (like Regex Patterns and Feature Extractors) instead of per-call or per-iteration.
