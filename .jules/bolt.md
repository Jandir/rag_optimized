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

## 2024-08-02 - [Avoid creating intermediate lists when searching substrings]
**Learning:** When parsing structured text blocks separated by predictable delimiters (like double newlines in SRT files) to extract a specific portion after a marker, splitting the block into a list of lines and iterating over it is slower and uses more memory than using native `str.find` to locate the marker and slicing the string directly. Benchmarks show `str.find` is ~2-3x faster.
**Action:** To locate specific markers within those blocks (like the '-->' timestamp line), use fast native string searches (`str.find`) and index slicing instead of splitting blocks into individual lines and iterating, which avoids unnecessary memory allocations and is computationally faster.

## 2026-08-11 - [String Concatenation Optimization in Loops]
**Learning:** String concatenation with `+=` in dynamic loops for markdown generation can lead to O(n²) memory reallocation overhead, and for performance consistency we should use a list of parts and `''.join()` instead.
**Action:** Use list append and `''.join()` instead of `+=` chaining for generating large text documents in loops.
## 2026-08-19 - [Avoid Multiline Regex for Structured Blocks in SRT Parsing]
**Learning:** Multiline regexes with lookaheads are computationally expensive and inefficient when parsing predictable `.srt` structures. Pre-compiling tag removal regexes and utilizing string `.split('\n\n')` alongside native string slicing (`str.find`) reduces parsing overhead significantly (up to ~2-3x speedup).
**Action:** Replaced the multiline regex in `_parse_srt_blocks` in both `rag_processor.py` and `rag_processor_local.py` with `split('\n\n')` and native string operations (`str.find`) and pre-compiled the `r'<[^>]*>'` HTML tag removal regex at the module level.
## 2025-01-20 - Caching Parsed Strings in Loop Deduplication
**Learning:** Parsing strings inside sequential string comparison (like finding overlaps in SRT 'rollup' blocks) creates redundant overhead because `prev_text_str` was split multiple times across loop iterations.
**Action:** Extract expensive operations like `.split()` and list comprehensions out of sequential processing functions (`_handle_partial_overlap`), and cache them (`prev_lines_cache_list`) in the loop to be reused across iterations.
