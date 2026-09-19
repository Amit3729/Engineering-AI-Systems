# Evaluation report

Generated 2026-09-18 20:56 UTC · provider `ollama` · model `qwen2.5:7b` · top_k 4 · researcher top_k 5 · researcher turns 3

## Summary by configuration

| configuration | n | completion | tool-calls | mean turns | mean tokens | total tokens | hard | soft | cascading |
|---|---|---|---|---|---|---|---|---|---|
| `single` | 20 | 65% | 80% | 2.1 | 6,337 | 126,747 | 0 | 8 | 0 |
| `multi` | 20 | 85% | 85% | 4.8 | 7,727 | 154,536 | 0 | 5 | 0 |
| `multi+no-compaction` | 20 | 80% | 90% | 4.7 | 8,332 | 166,645 | 0 | 4 | 1 |
| `single+malformed_retrieval` | 6 | 0% | 100% | 3.7 | 2,825 | 16,951 | 0 | 0 | 6 |
| `multi+malformed_retrieval` | 6 | 0% | 100% | 4.0 | 2,860 | 17,158 | 0 | 5 | 1 |
| `multi+tool_unavailable` | 6 | 0% | 0% | 2.2 | 1,058 | 6,347 | 0 | 6 | 0 |
| `multi+retrieval_timeout` | 6 | 0% | 100% | 4.2 | 2,739 | 16,432 | 0 | 6 | 0 |

## What compaction saves

Same pattern, same queries, compaction on (`multi`) versus off (`multi+no-compaction`): **7,727 vs 8,332** mean tokens per query, a **7%** reduction, at 85% versus 80% completion.

## Coordination cost

The multi-agent mode spends **1.22x** the tokens of the single-agent baseline (7,727 vs 6,337 per query) and completes 85% of cases against the baseline's 65%.

The planner declined to fan out on some queries and handed them back to the single-agent loop: 3 in `multi`, 3 in `multi+no-compaction`. Those rows are reported as `multi->single`.

## Completion by query kind

| kind | `single` | `multi` | `multi+no-compaction` | `single+malformed_retrieval` | `multi+malformed_retrieval` | `multi+tool_unavailable` | `multi+retrieval_timeout` |
|---|---|---|---|---|---|---|---|
| arithmetic | 2/2 | 2/2 | 2/2 | - | - | - | - |
| multi_hop | 4/7 | 5/7 | 5/7 | 0/2 | 0/2 | 0/2 | 0/2 |
| single_hop | 5/9 | 8/9 | 8/9 | 0/4 | 0/4 | 0/4 | 0/4 |
| unanswerable | 2/2 | 2/2 | 1/2 | - | - | - | - |

## Per-query results

| query | config | ran as | completed | tools ok | turns | tokens | confidence | failure |
|---|---|---|---|---|---|---|---|---|
| `arith-simple` | `single` | single | yes | yes | 2 | 3,462 | 1.00 | - |
| `arith-units` | `single` | single | yes | yes | 2 | 6,259 | 1.00 | - |
| `cross-corpus` | `single` | single | no | no | 1 | 2,366 | 0.80 | soft |
| `dataclass-json` | `single` | single | no | no | 1 | 3,941 | 0.90 | soft |
| `gather-taskgroup` | `single` | single | yes | yes | 2 | 7,497 | 0.95 | - |
| `int-error` | `single` | single | no | yes | 4 | 6,226 | 0.95 | soft |
| `json-indent` | `single` | single | yes | yes | 2 | 7,030 | 0.95 | - |
| `list-comp` | `single` | single | yes | yes | 2 | 7,629 | 0.95 | - |
| `lru-cache` | `single` | single | yes | yes | 2 | 8,527 | 0.95 | - |
| `mmap` | `single` | single | no | yes | 2 | 5,114 | 0.95 | soft |
| `ordereddict` | `single` | single | yes | yes | 2 | 6,864 | 0.95 | - |
| `out-of-corpus-nginx` | `single` | single | yes | yes | 2 | 8,537 | 0.70 | - |
| `out-of-corpus-pricing` | `single` | single | yes | yes | 4 | 12,070 | 0.60 | - |
| `path-glob` | `single` | single | no | yes | 4 | 5,834 | 0.95 | soft |
| `project-chunking` | `single` | single | yes | yes | 2 | 6,376 | 0.95 | - |
| `project-serving` | `single` | single | no | yes | 2 | 7,614 | 0.90 | soft |
| `read-lines` | `single` | single | yes | no | 1 | 3,244 | 1.00 | soft |
| `subprocess-timeout` | `single` | single | yes | yes | 2 | 6,841 | 0.95 | - |
| `tz-convert` | `single` | single | no | no | 1 | 3,639 | 1.00 | soft |
| `url-roundtrip` | `single` | single | yes | yes | 2 | 7,677 | 0.95 | - |
| `arith-simple` | `multi` | multi->single | yes | yes | 3 | 3,715 | 1.00 | - |
| `arith-units` | `multi` | multi->single | yes | yes | 3 | 6,527 | 1.00 | - |
| `cross-corpus` | `multi` | multi | yes | no | 6 | 8,666 | 0.65 | soft |
| `dataclass-json` | `multi` | multi | yes | yes | 5 | 6,934 | 0.70 | - |
| `gather-taskgroup` | `multi` | multi | yes | yes | 5 | 11,679 | 0.80 | - |
| `int-error` | `multi` | multi | no | yes | 5 | 5,923 | 1.00 | soft |
| `json-indent` | `multi` | multi | yes | yes | 5 | 6,373 | 1.00 | - |
| `list-comp` | `multi` | multi | yes | yes | 4 | 4,348 | 1.00 | - |
| `lru-cache` | `multi` | multi | yes | yes | 5 | 10,804 | 1.00 | - |
| `mmap` | `multi` | multi | yes | yes | 5 | 4,090 | 0.70 | - |
| `ordereddict` | `multi` | multi | no | yes | 4 | 7,925 | 0.00 | soft |
| `out-of-corpus-nginx` | `multi` | multi | yes | yes | 4 | 5,852 | 0.30 | - |
| `out-of-corpus-pricing` | `multi` | multi->single | yes | yes | 4 | 12,747 | 0.60 | - |
| `path-glob` | `multi` | multi | yes | yes | 5 | 6,751 | 0.80 | - |
| `project-chunking` | `multi` | multi | yes | yes | 4 | 3,374 | 0.80 | - |
| `project-serving` | `multi` | multi | yes | yes | 5 | 9,078 | 0.80 | - |
| `read-lines` | `multi` | multi | yes | yes | 5 | 4,971 | 1.00 | - |
| `subprocess-timeout` | `multi` | multi | yes | yes | 6 | 14,434 | 1.00 | - |
| `tz-convert` | `multi` | multi | no | no | 6 | 9,619 | 0.75 | soft |
| `url-roundtrip` | `multi` | multi | yes | no | 6 | 10,726 | 0.70 | soft |
| `arith-simple` | `multi+no-compaction` | multi->single | yes | yes | 3 | 3,715 | 1.00 | - |
| `arith-units` | `multi+no-compaction` | multi->single | yes | yes | 3 | 6,527 | 1.00 | - |
| `cross-corpus` | `multi+no-compaction` | multi | yes | yes | 5 | 9,502 | 0.80 | - |
| `dataclass-json` | `multi+no-compaction` | multi | yes | yes | 5 | 7,887 | 1.00 | - |
| `gather-taskgroup` | `multi+no-compaction` | multi | yes | yes | 4 | 8,027 | 0.95 | - |
| `int-error` | `multi+no-compaction` | multi | no | yes | 5 | 7,375 | 1.00 | soft |
| `json-indent` | `multi+no-compaction` | multi | yes | yes | 5 | 8,308 | 1.00 | - |
| `list-comp` | `multi+no-compaction` | multi | yes | yes | 4 | 4,587 | 1.00 | - |
| `lru-cache` | `multi+no-compaction` | multi | yes | yes | 5 | 9,973 | 1.00 | - |
| `mmap` | `multi+no-compaction` | multi | yes | yes | 4 | 4,016 | 1.00 | - |
| `ordereddict` | `multi+no-compaction` | multi | no | yes | 5 | 7,584 | 0.80 | soft |
| `out-of-corpus-nginx` | `multi+no-compaction` | multi | no | yes | 5 | 8,985 | 0.60 | soft |
| `out-of-corpus-pricing` | `multi+no-compaction` | multi->single | yes | yes | 3 | 7,881 | 0.70 | - |
| `path-glob` | `multi+no-compaction` | multi | yes | yes | 5 | 8,035 | 1.00 | - |
| `project-chunking` | `multi+no-compaction` | multi | yes | yes | 4 | 3,374 | 0.80 | - |
| `project-serving` | `multi+no-compaction` | multi | yes | yes | 5 | 11,522 | 0.80 | - |
| `read-lines` | `multi+no-compaction` | multi | yes | yes | 5 | 7,978 | 1.00 | - |
| `subprocess-timeout` | `multi+no-compaction` | multi | yes | yes | 6 | 18,285 | 0.95 | - |
| `tz-convert` | `multi+no-compaction` | multi | no | no | 6 | 12,647 | 0.70 | cascading_soft |
| `url-roundtrip` | `multi+no-compaction` | multi | yes | no | 6 | 10,437 | 0.70 | soft |
| `dataclass-json` | `single+malformed_retrieval` | single | no | yes | 4 | 3,057 | 0.70 | cascading_soft |
| `json-indent` | `single+malformed_retrieval` | single | no | yes | 4 | 3,040 | 0.70 | cascading_soft |
| `lru-cache` | `single+malformed_retrieval` | single | no | yes | 4 | 3,143 | 0.80 | cascading_soft |
| `mmap` | `single+malformed_retrieval` | single | no | yes | 4 | 3,164 | 0.70 | cascading_soft |
| `project-serving` | `single+malformed_retrieval` | single | no | yes | 2 | 1,398 | 0.50 | cascading_soft |
| `tz-convert` | `single+malformed_retrieval` | single | no | yes | 4 | 3,149 | 0.70 | cascading_soft |
| `dataclass-json` | `multi+malformed_retrieval` | multi | no | yes | 4 | 2,289 | 0.50 | cascading_soft |
| `json-indent` | `multi+malformed_retrieval` | multi | no | yes | 4 | 2,206 | 0.00 | soft |
| `lru-cache` | `multi+malformed_retrieval` | multi | no | yes | 4 | 3,526 | 0.00 | soft |
| `mmap` | `multi+malformed_retrieval` | multi | no | yes | 4 | 2,174 | 0.00 | soft |
| `project-serving` | `multi+malformed_retrieval` | multi | no | yes | 4 | 3,390 | 0.00 | soft |
| `tz-convert` | `multi+malformed_retrieval` | multi | no | yes | 4 | 3,573 | 0.00 | soft |
| `dataclass-json` | `multi+tool_unavailable` | multi | no | no | 2 | 1,025 | 0.30 | soft |
| `json-indent` | `multi+tool_unavailable` | multi | no | no | 2 | 990 | 0.00 | soft |
| `lru-cache` | `multi+tool_unavailable` | multi | no | no | 2 | 1,083 | 0.00 | soft |
| `mmap` | `multi+tool_unavailable` | multi | no | no | 2 | 961 | 0.00 | soft |
| `project-serving` | `multi+tool_unavailable` | multi | no | no | 2 | 1,058 | 0.00 | soft |
| `tz-convert` | `multi+tool_unavailable` | multi | no | no | 3 | 1,230 | 0.80 | soft |
| `dataclass-json` | `multi+retrieval_timeout` | multi | no | yes | 4 | 2,124 | 0.00 | soft |
| `json-indent` | `multi+retrieval_timeout` | multi | no | yes | 4 | 2,188 | 0.00 | soft |
| `lru-cache` | `multi+retrieval_timeout` | multi | no | yes | 4 | 3,361 | 0.00 | soft |
| `mmap` | `multi+retrieval_timeout` | multi | no | yes | 5 | 2,143 | 0.00 | soft |
| `project-serving` | `multi+retrieval_timeout` | multi | no | yes | 4 | 3,223 | 0.00 | soft |
| `tz-convert` | `multi+retrieval_timeout` | multi | no | yes | 4 | 3,393 | 0.30 | soft |

## Failure log

### hard (0)

*No usable answer was produced: an exception, or an empty response.*

None.

### soft (34)

*An answer was produced and looks fine, but is wrong on its own terms: the wrong tool, invalid arguments, a missing requirement, or an ungrounded citation.*

- **`path-glob`** (`single`, single_hop) — declined a question the corpus does support
  - confidence 0.95, 4 turns, answer: "The `pathlib.Path.glob` method returns path names that match the given pattern. The behavior can vary slightly from the `glob` module. For example, files beginn…"
- **`read-lines`** (`single`, single_hop) — never called ['search_knowledge_base']
  - confidence 1.00, 1 turns, answer: "To read a file line by line in Python, you can use a loop to iterate over the file object. Here's an example of how to do it:  ```python with open('filename.txt…"
- **`int-error`** (`single`, single_hop) — cited nothing expected, needed 1 of ['builtins/functions.html', 'builtins/exceptions.html', 'library/exceptions.html']
  - confidence 0.95, 4 turns, answer: "The int() function raises a ValueError when given a non-numeric string.…"
- **`mmap`** (`single`, single_hop) — cited nothing expected, needed 1 of ['library/mmap.html']
  - confidence 0.95, 2 turns, answer: "The mmap module in Python provides an interface to memory-mapped files on both Windows and Unix. A file's contents can be mapped directly into memory, at which …"
- **`tz-convert`** (`single`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']; never called ['search_knowledge_base']
  - confidence 1.00, 1 turns, answer: "To build a timezone-aware datetime, you can use the `datetime` constructor with a `tzinfo` object. For example, using the `KabulTz` class from the provided cont…"
- **`dataclass-json`** (`single`, multi_hop) — cited ['library/json.html'], needed 2 of ['library/dataclasses.html', 'library/json.html']; never called ['search_knowledge_base']
  - confidence 0.90, 1 turns, answer: "To serialize a dataclass to JSON, you can use the `json` module along with a custom encoder. The example provided demonstrates creating a `ComplexEncoder` class…"
- **`project-serving`** (`single`, single_hop) — cited nothing expected, needed 1 of ['serving.md']
  - confidence 0.90, 2 turns, answer: "The project uses the WSGI (Web Server Gateway Interface) serving engine. WSGI is a standard interface between web servers and web applications in Python, and th…"
- **`cross-corpus`** (`single`, multi_hop) — cited ['library/mmap.html'], needed 2 of ['serving.md', 'library/mmap.html']; declined a question the corpus does support; never called ['search_knowledge_base']
  - confidence 0.80, 1 turns, answer: "The question appears to be asking about vLLM's paged attention mechanism and Python's mmap module. However, the provided context and tool results do not contain…"
- **`int-error`** (`multi`, single_hop) — answer is missing ['ValueError']
  - confidence 1.00, 5 turns, answer: "int() raises a TypeError when given a non-numeric string.…"
- **`tz-convert`** (`multi`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']; calculator rejected the arguments: Only basic arithmetic (+ - * / // % **) is supported
  - tool errors: ['calculator: Only basic arithmetic (+ - * / // % **) is supported']
  - confidence 0.75, 6 turns, answer: "To build a timezone-aware datetime object, you can use `datetime.now(timezone.utc)`. To convert this object to another timezone, use the `astimezone(tz)` method…"
- **`url-roundtrip`** (`multi`, multi_hop) — calculator rejected the arguments: Only basic arithmetic (+ - * / // % **) is supported
  - tool errors: ['calculator: Only basic arithmetic (+ - * / // % **) is supported']
  - confidence 0.70, 6 turns, answer: "To parse a URL query string, you can use the `urllib.parse.parse_qs` function. After parsing, you can percent-encode the values using the `str.format` method or…"
- **`ordereddict`** (`multi`, multi_hop) — declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The findings do not provide specific details on how collections.OrderedDict differs from a plain dict. More information is needed to accurately describe the dif…"
- **`cross-corpus`** (`multi`, multi_hop) — calculator rejected the arguments: Only basic arithmetic (+ - * / // % **) is supported
  - tool errors: ['calculator: Only basic arithmetic (+ - * / // % **) is supported']
  - confidence 0.65, 6 turns, answer: "vLLM's paged attention mechanism stores the key-value cache in non-contiguous pages to reduce memory fragmentation. Python's mmap module allows memory-mapped fi…"
- **`int-error`** (`multi+no-compaction`, single_hop) — answer is missing ['ValueError']
  - confidence 1.00, 5 turns, answer: "int() raises a TypeError when given a non-numeric string.…"
- **`url-roundtrip`** (`multi+no-compaction`, multi_hop) — calculator rejected the arguments: Only basic arithmetic (+ - * / // % **) is supported
  - tool errors: ['calculator: Only basic arithmetic (+ - * / // % **) is supported']
  - confidence 0.70, 6 turns, answer: "To parse a URL query string, you can use the `urllib.parse.parse_qs` function. After parsing, you can percent-encode the values using the `str.format` method fo…"
- **`ordereddict`** (`multi+no-compaction`, multi_hop) — declined a question the corpus does support
  - confidence 0.80, 5 turns, answer: "collections.OrderedDict maintains the order of insertion, while a plain dict does not. OrderedDict supports reordering operations better and was designed for su…"
- **`out-of-corpus-nginx`** (`multi+no-compaction`, unanswerable) — answered a question the corpus cannot support instead of declining
  - confidence 0.60, 5 turns, answer: "The specific configuration or modules required for Nginx to handle websockets are not mentioned in the provided findings. However, Nginx handles websockets by r…"
- **`json-indent`** (`multi+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/json.html']; declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The findings do not provide information about what the indent argument to json.dumps does.…"
- **`lru-cache`** (`multi+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/functools.html']; declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The findings do not provide information on what functools.lru_cache does or what maxsize controls.…"
- **`mmap`** (`multi+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/mmap.html']; declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The answer is missing from the provided findings.…"
- **`tz-convert`** (`multi+malformed_retrieval`, multi_hop) — cited nothing expected, needed 2 of ['library/datetime.html', 'library/zoneinfo.html']; declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The findings do not provide specific instructions on how to create a timezone-aware datetime object or convert it to another timezone in Python 3.14.…"
- **`project-serving`** (`multi+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['serving.md']; declined a question the corpus does support
  - confidence 0.00, 4 turns, answer: "The findings do not provide information on the serving engine used by the project or the reasons behind its selection.…"
- **`json-indent`** (`multi+tool_unavailable`, single_hop) — cited nothing expected, needed 1 of ['library/json.html']; declined a question the corpus does support; never called ['search_knowledge_base']
  - confidence 0.00, 2 turns, answer: "The documentation or findings provided do not contain information about the indent argument for json.dumps. More details are needed to answer this question.…"
- **`lru-cache`** (`multi+tool_unavailable`, single_hop) — cited nothing expected, needed 1 of ['library/functools.html']; declined a question the corpus does support; never called ['search_knowledge_base']
  - confidence 0.00, 2 turns, answer: "The findings did not provide information on what functools.lru_cache does or what maxsize controls.…"
- **`mmap`** (`multi+tool_unavailable`, single_hop) — cited nothing expected, needed 1 of ['library/mmap.html']; declined a question the corpus does support; never called ['search_knowledge_base']
  - confidence 0.00, 2 turns, answer: "The citation is missing, so the answer cannot be provided with the required confidence.…"
- **`tz-convert`** (`multi+tool_unavailable`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']; never called ['search_knowledge_base']
  - confidence 0.80, 3 turns, answer: "To build a timezone-aware datetime object, you can use the `datetime` module in Python. For example, `dt = datetime.datetime.now(datetime.timezone.utc)` creates…"
- **`dataclass-json`** (`multi+tool_unavailable`, multi_hop) — cited nothing expected, needed 2 of ['library/dataclasses.html', 'library/json.html']; never called ['search_knowledge_base']
  - confidence 0.30, 2 turns, answer: "The findings do not provide specific information on how to serialize a dataclass to JSON. Typically, you would use a library like `json` in Python along with a …"
- **`project-serving`** (`multi+tool_unavailable`, single_hop) — cited nothing expected, needed 1 of ['serving.md']; declined a question the corpus does support; never called ['search_knowledge_base']
  - confidence 0.00, 2 turns, answer: "The findings do not provide information on the serving engine used by this project or the reasons behind its selection.…"
- **`json-indent`** (`multi+retrieval_timeout`, single_hop) — cited nothing expected, needed 1 of ['library/json.html']; declined a question the corpus does support
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.00, 4 turns, answer: "The findings do not provide information on what the indent argument to json.dumps does.…"
- **`lru-cache`** (`multi+retrieval_timeout`, single_hop) — cited nothing expected, needed 1 of ['library/functools.html']; declined a question the corpus does support
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved", "search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.00, 4 turns, answer: "The findings do not provide specific information about what functools.lru_cache does or what maxsize controls. Documentation for functools.lru_cache is missing.…"
- **`mmap`** (`multi+retrieval_timeout`, single_hop) — cited nothing expected, needed 1 of ['library/mmap.html']; declined a question the corpus does support
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.00, 5 turns, answer: "The findings do not provide specific information on the usage of the mmap module.…"
- **`tz-convert`** (`multi+retrieval_timeout`, multi_hop) — cited nothing expected, needed 2 of ['library/datetime.html', 'library/zoneinfo.html']
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved", "search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.30, 4 turns, answer: "The specific methods for creating a timezone-aware datetime object and converting it to another timezone are not provided in the given findings. You would typic…"
- **`dataclass-json`** (`multi+retrieval_timeout`, multi_hop) — cited nothing expected, needed 2 of ['library/dataclasses.html', 'library/json.html']; declined a question the corpus does support
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.00, 4 turns, answer: "The findings do not provide specific information on how to serialise a dataclass to JSON. Additional research is needed to address this gap.…"
- **`project-serving`** (`multi+retrieval_timeout`, single_hop) — cited nothing expected, needed 1 of ['serving.md']; declined a question the corpus does support
  - tool errors: ["search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved", "search_knowledge_base: Tool 'search_knowledge_base' timed out after 20s; no results were retrieved"]
  - confidence 0.00, 4 turns, answer: "The serving engine used by this project and the specific reason for its choice are not documented.…"

### cascading soft (8)

*An upstream step failed or returned nothing usable, and the agent still produced a confident answer on top of it.*

- **`tz-convert`** (`multi+no-compaction`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']; invalid arguments: ["search_knowledge_base: unexpected argument 'matches'"]; search_knowledge_base rejected the arguments: Invalid arguments for 'search_knowledge_base': ToolRegistry._search() got an unexpected keyword argument 'matches'
  - tool errors: ["search_knowledge_base: Invalid arguments for 'search_knowledge_base': ToolRegistry._search() got an unexpected keyword argument 'matches'"]
  - confidence 0.70, 6 turns, answer: "To build a timezone-aware datetime object, you can use `datetime.now(timezone)`. To convert it to another timezone, use the `astimezone()` method with the targe…"
- **`json-indent`** (`single+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/json.html']
  - confidence 0.70, 4 turns, answer: "The `indent` argument in `json.dumps` is used to pretty-print the JSON output, adding indentation to the JSON string to make it more readable. It takes an integ…"
- **`lru-cache`** (`single+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/functools.html']
  - confidence 0.80, 4 turns, answer: "The `functools.lru_cache` is a function decorator in Python that provides a mechanism for caching the results of a function call. This caching can significantly…"
- **`mmap`** (`single+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['library/mmap.html']
  - confidence 0.70, 4 turns, answer: "The `mmap` module in Python is used for memory-mapped file I/O, allowing files to be accessed as if they were memory-mapped. This can be useful for working with…"
- **`tz-convert`** (`single+malformed_retrieval`, multi_hop) — cited nothing expected, needed 2 of ['library/datetime.html', 'library/zoneinfo.html']
  - confidence 0.70, 4 turns, answer: "To build a timezone-aware datetime in Python, you can use the `pytz` library along with the `datetime` module. Here's a basic example of how to create a timezon…"
- **`dataclass-json`** (`single+malformed_retrieval`, multi_hop) — cited nothing expected, needed 2 of ['library/dataclasses.html', 'library/json.html']
  - confidence 0.70, 4 turns, answer: "To serialize a dataclass to JSON, you can use the `json` module in Python. You would typically create an instance of your dataclass and then use the `json.dumps…"
- **`project-serving`** (`single+malformed_retrieval`, single_hop) — cited nothing expected, needed 1 of ['serving.md']
  - confidence 0.50, 2 turns, answer: "No specific serving engine is mentioned in the project's documentation. It's possible that the project uses a built-in or custom solution, but without more deta…"
- **`dataclass-json`** (`multi+malformed_retrieval`, multi_hop) — cited nothing expected, needed 2 of ['library/dataclasses.html', 'library/json.html']
  - confidence 0.50, 4 turns, answer: "The findings do not provide specific information on how to serialise dataclasses to JSON. You may need to use a library like `json` in Python along with the `as…"
