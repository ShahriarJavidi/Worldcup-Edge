import pytest

from wm.config import DEFAULT_EMBEDDING_MODEL
from wm.rag.analysis import match_title
from wm.rag.ingest import ingest_text
from wm.rag.retrieval import match_context
from wm.rag.sources import html_to_text
from wm.rag.store import KIND_TEAM, KnowledgeBase, kind_of


def test_html_to_text_drops_scripts_and_keeps_blocks():
    html = "<html><script>x=1</script><h1>Canada 1-1 Bosnia</h1><p>David  scored.</p></html>"
    assert html_to_text(html) == "Canada 1-1 Bosnia\nDavid scored."


LEGACY_ANALYSIS = """# Structured Match Analysis Report

---

## Match
**Competition:** FIFA World Cup 2026, Group B

---

## Teams
**Home:** Canada
**Away:** Bosnia and Herzegovina

## Result
**Canada 1-1 Bosnia and Herzegovina**
"""


def test_match_title():
    assert match_title("Title: Canada vs Qatar, World Cup, 2026-06-18\n\nMatch\n...") == (
        "Canada vs Qatar, World Cup, 2026-06-18"
    )
    assert match_title(LEGACY_ANALYSIS) == "Canada vs Bosnia and Herzegovina"
    assert match_title("Teams\nSpain\nUruguay\nResult\n2-0") == "Spain vs Uruguay"
    assert match_title("## Teams\n**Mexico** vs **South Africa**\n---\n## Result") == "Mexico vs South Africa"
    assert match_title("## Teams\nTeam A: Korea Republic\nTeam B: Czechia") == "Korea Republic vs Czechia"
    assert match_title("He praised all Teams in the Premier League.") is None
    assert match_title("no headings") is None


def test_kind_inferred_for_legacy_chunks():
    assert kind_of({"source": "api_football:Iran"}) == KIND_TEAM
    assert kind_of({"kind": "raw_text", "source": "api_football:Iran"}) == "raw_text"


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    pytest.importorskip("sentence_transformers")
    try:
        kb = KnowledgeBase(tmp_path_factory.mktemp("kb"), DEFAULT_EMBEDDING_MODEL)
        filler = "Recent fixtures:\n" + "\n".join(f"2026-0{i % 9 + 1}-01: Home 1 - 0 Away" for i in range(40))
        for team in ("Canada", "Iran", "Spain"):
            injuries = f"Recent injury / absence notes:\n{team}: Player X - hamstring"
            kb.add(
                f"{filler}\n\n{injuries}", source=f"api_football:{team}", kind=KIND_TEAM, title=f"Team dossier: {team}"
            )
    except OSError as exc:  # embedding model not downloadable offline
        pytest.skip(str(exc))
    return kb


def test_titles_make_mid_document_chunks_findable(kb):
    hits = kb.search("Canada injuries", k=3)
    assert hits[0].source == "api_football:Canada"
    assert all(h.text.startswith("[Team dossier: ") for h in hits)


def test_match_context_covers_both_teams(kb):
    sources = {h.source for h in match_context(kb, "Canada", "Spain", k=6)}
    assert {"api_football:Canada", "api_football:Spain"} <= sources


def test_ingest_raw_text(kb):
    result = ingest_text(kb, "Short note about Spain's pressing.", source="note")
    assert result.chunks == 1 and result.kind == "raw_text"


def test_re_adding_a_source_replaces_it(kb):
    kb.add("Old note about Qatar.", source="note:qatar", kind="raw_text")
    kb.add("New note about Qatar.", source="note:qatar", kind="raw_text")
    texts = [h.text for h in kb.search("Qatar note", k=10) if h.source == "note:qatar"]
    assert texts == ["[note:qatar]\nNew note about Qatar."]


def test_delete(kb):
    kb.add("Temporary.", source="tmp", kind="raw_text")
    assert kb.delete("tmp") == 1
    assert kb.delete("tmp") == 0


def test_reindex_upgrades_legacy_chunks(kb):
    from langchain_core.documents import Document

    legacy = "Recent injury / absence notes:\nGhana: Player Y - knee"
    kb._store.add_documents(
        [
            Document(page_content=legacy, metadata={"source": "api_football:Ghana"}),
            Document(page_content=legacy, metadata={"source": "api_football:ghana"}),  # case duplicate
        ]
    )
    result = kb.reindex()
    assert (result.retitled, result.duplicates_removed) == (1, 1)
    hit = kb.search("Ghana injuries", k=1, where={"kind": KIND_TEAM})[0]
    assert hit.text == "[Team dossier: Ghana]\n" + legacy
    assert kb.reindex() == type(result)(0, 0)  # idempotent
