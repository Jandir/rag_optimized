## 2026-05-09 - [Precompiling Regex & Avoiding Instantiation inside Loops]
**Learning:** `yake.KeywordExtractor` objects have a significantly slow initialization step. Instantiating it per section or inside loops creates a notable performance bottleneck. Repeatedly calling `re.compile()` inside functions (like `clean_srt_content` and `extract_metadata_from_filename`) also adds overhead.
**Action:** When working with Python text-processing code, look for module-level `re.compile()` opportunities and always hoist NLP objects like `yake.KeywordExtractor` or `spacy.load` models into class initializations (`__init__`) instead of function loops to maintain performance.

## 2026-05-21 - [Pre-compile dynamic user regexes]
**Learning:** Using `re.sub(pattern, repl, string)` inside loops for dynamic configuration-based rules is a performance trap because Python dynamically compiles the regex pattern into a cached object on every iteration if it's not pre-compiled, causing a significant slowdown when processing multiple text blocks or files.
**Action:** When applying dynamic rules loaded from configuration or user input (like `rules.txt`), pre-compile them during the initialization/loading phase using `re.compile(pattern)` (handling compilation errors with try/except), and use `pattern_obj.sub(repl, string)` during the loop processing phase.

## 2026-06-04 - [Optimize spaCy initialization and pipeline]
**Learning:** Loading the full spaCy NLP pipeline using `spacy.load()` is highly inefficient if only specific components are needed (e.g., NER and sentences). The default pipeline loads components like `tagger`, `morphologizer`, `lemmatizer`, `attribute_ruler`, and `parser`, which consume significant memory and CPU time during processing.
**Action:** Always specify the `disable` argument in `spacy.load()` to turn off unused pipeline components. Furthermore, replace the heavy `parser` component with the lightweight `sentencizer` using `nlp.add_pipe("sentencizer")` when only sentence boundary detection is required.
## 2025-05-18 - String Manipulation Overhead in High-Volume Parsing
**Learning:** Sequential `.replace()` calls on strings, especially in frequently executed functions like metadata extractors or rules engines, create hidden overhead by generating multiple intermediate string objects. Similarly, `in` checks are much faster than full string replacements. In regex parsing, avoiding unnecessary capture groups and using non-greedy matches `(.*?)` is often noticeably faster than complex negative lookaheads for block parsing.
**Action:** When enforcing large sets of terminology rules, always precede string replacements with an `if original in text:` check to avoid unnecessary operations. For metadata cleanup, use conditional `.endswith()` slicing instead of chained replacements. For regex, minimize capture groups to the bare minimum needed.

## 2024-06-24 - [Avoid Multiline Regex for Structured Blocks]
**Learning:** For parsing structured text blocks separated by predictable delimiters (like double newlines in SRT files), native string operations like `.split('\n\n')` with iterative line checks are significantly faster (up to ~2x) than using complex multi-line regular expressions with negative lookaheads (`re.DOTALL`, `(?=\n\n|$)`). Also, caching list comprehensions and using `list.extend()` instead of loop-appends can speed up line-by-line deduplication operations by ~30-40%.
**Action:** When extracting blocks of text that are predictably delimited (like blank lines), prefer `text.split('\n\n')` over regex `finditer()`. Always cache results of expensive operations (like string splitting) that are reused across loop iterations, and prefer bulk list operations like `.extend()` over `.append()` in a loop.

## 2026-06-25 - [Use str.find instead of split for parsing structured blocks]
**Learning:** When trying to locate specific markers within structured text blocks separated by predictable delimiters (like the '-->' timestamp line in SRT files), splitting the block into lines and iterating over them is inefficient and causes unnecessary memory allocations. Fast native string searches (`str.find`) combined with slicing is computationally faster.
**Action:** When parsing blocks, prefer `str.find()` and string slicing over `.split('\n')` and iterating whenever you need to find a single marker and extract the text after it.
