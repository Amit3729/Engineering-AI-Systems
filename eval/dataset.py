"""The evaluation set.

Every case names documents that exist in the indexed corpus, so a miss is a
retrieval failure rather than a typo. The four kinds exercise different parts of
the system: ``single_hop`` needs one lookup, ``multi_hop`` needs two or more in
different modules, ``arithmetic`` must reach the calculator and not the index,
and ``unanswerable`` must be declined rather than answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEARCH = "search_knowledge_base"


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    kind: str
    # Substrings matched against the documents the run actually cited or retrieved.
    expect_documents: tuple[str, ...] = ()
    # At least this many of the above must appear; defaults to all of them.
    min_documents: int | None = None
    expect_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    # Substrings that must appear in the answer text, compared case-insensitively.
    must_include: tuple[str, ...] = ()
    # True when the correct behaviour is to decline for lack of evidence.
    expect_decline: bool = False
    notes: str = ""

    @property
    def required_documents(self) -> int:
        if self.min_documents is not None:
            return self.min_documents
        return len(self.expect_documents)


CASES: tuple[Case, ...] = (
    # ---------------------------------------------------------- single hop
    Case("json-indent", "What does the indent argument to json.dumps do?", "single_hop",
         expect_documents=("library/json.html",), expect_tools=(SEARCH,)),
    Case("lru-cache", "What does functools.lru_cache do and what does maxsize control?", "single_hop",
         expect_documents=("library/functools.html",), expect_tools=(SEARCH,)),
    Case("path-glob", "What does pathlib.Path.glob return?", "single_hop",
         expect_documents=("library/pathlib.html",), expect_tools=(SEARCH,)),
    Case("list-comp", "What is a list comprehension in Python?", "single_hop",
         expect_documents=("tutorial/datastructures.html", "reference/expressions.html"), min_documents=1,
         expect_tools=(SEARCH,)),
    Case("read-lines", "How do I read a file line by line in Python?", "single_hop",
         expect_documents=("tutorial/inputoutput.html", "library/io.html", "builtins/functions.html"), min_documents=1,
         expect_tools=(SEARCH,)),
    Case("int-error", "Which exception does int() raise when given a non-numeric string?", "single_hop",
         expect_documents=("builtins/functions.html", "builtins/exceptions.html", "library/exceptions.html"),
         min_documents=1, expect_tools=(SEARCH,), must_include=("ValueError",)),
    Case("mmap", "What is the mmap module used for?", "single_hop",
         expect_documents=("library/mmap.html",), expect_tools=(SEARCH,)),

    # ----------------------------------------------------------- multi hop
    Case("tz-convert", "How do I build a timezone-aware datetime and convert it to another timezone?", "multi_hop",
         expect_documents=("library/datetime.html", "library/zoneinfo.html"), min_documents=2, expect_tools=(SEARCH,),
         notes="needs datetime for the aware object and zoneinfo for the target zone"),
    Case("gather-taskgroup", "What is the difference between asyncio.gather and asyncio.TaskGroup?", "multi_hop",
         expect_documents=("library/asyncio-task.html",), expect_tools=(SEARCH,)),
    Case("url-roundtrip", "How do I parse a URL query string and then percent-encode the values again?", "multi_hop",
         expect_documents=("library/urllib.parse.html",), expect_tools=(SEARCH,)),
    Case("dataclass-json", "How do I serialise a dataclass to JSON?", "multi_hop",
         expect_documents=("library/dataclasses.html", "library/json.html"), min_documents=2, expect_tools=(SEARCH,)),
    Case("subprocess-timeout", "How do I run a subprocess, capture stdout and stderr, and apply a timeout?", "multi_hop",
         expect_documents=("library/subprocess.html",), expect_tools=(SEARCH,)),
    Case("ordereddict", "How does collections.OrderedDict differ from a plain dict?", "multi_hop",
         expect_documents=("library/collections.html", "builtins/stdtypes.html"), min_documents=1,
         expect_tools=(SEARCH,)),

    # ---------------------------------------------------------- arithmetic
    Case("arith-simple", "What is 12 * 4 + 7?", "arithmetic",
         expect_tools=("calculator",), forbid_tools=(SEARCH,), must_include=("55",)),
    Case("arith-units", "Compute 16535 * 4096 / 1048576 and give the result.", "arithmetic",
         expect_tools=("calculator",), forbid_tools=(SEARCH,), must_include=("64.5",)),

    # ------------------------------------------------------- project notes
    Case("project-serving", "What serving engine does this project use, and why was it chosen?", "single_hop",
         expect_documents=("serving.md",), expect_tools=(SEARCH,)),
    Case("project-chunking", "How does this project's retrieval layer chunk documents?", "single_hop",
         expect_documents=("retrieval.md",), expect_tools=(SEARCH,)),

    # ------------------------------------------------------ across corpora
    Case("cross-corpus", "What does vLLM's paged attention do, and separately, what does Python's mmap module do?",
         "multi_hop", expect_documents=("serving.md", "library/mmap.html"), min_documents=2, expect_tools=(SEARCH,),
         notes="one lookup per corpus; the source filter should differ between them"),

    # -------------------------------------------------------- unanswerable
    Case("out-of-corpus-pricing", "What does the OpenAI batch API cost per million tokens in 2026?", "unanswerable",
         expect_decline=True, notes="not in either corpus; a priced answer is a hallucination"),
    Case("out-of-corpus-nginx", "How do I configure nginx reverse proxy buffering for websockets?", "unanswerable",
         expect_decline=True, notes="plausible-sounding but absent from the index"),
)


def select(kinds: tuple[str, ...] = (), ids: tuple[str, ...] = (), limit: int | None = None) -> list[Case]:
    cases = [case for case in CASES if (not kinds or case.kind in kinds) and (not ids or case.id in ids)]
    return cases[:limit] if limit else cases
