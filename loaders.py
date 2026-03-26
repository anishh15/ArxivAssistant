"""
Document loader for Arxiv research papers.
Fetches live papers dynamically — no pre-loaded PDFs.

Uses the `arxiv` library directly (instead of LangChain's ArxivLoader)
to get proper field-based search (title, abstract, etc.).
"""

import re
import arxiv
from langchain_core.documents import Document
from config import MAX_ARXIV_DOCS


# Regex to detect arxiv paper IDs like "2311.09521" or "2311.09521v2"
_ARXIV_ID_PATTERN = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')

def _download_pdf_text(result) -> str:
    """Try to download and extract full text from the paper's PDF."""
    try:
        import tempfile, os, pymupdf
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = result.download_pdf(dirpath=tmpdir)
            doc = pymupdf.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text.strip()
    except Exception as e:
        print(f"[Arxiv] PDF extraction failed for '{result.title}': {e}")
        return ""


def _results_to_documents(results) -> list:
    """Convert arxiv search results to LangChain Documents with full PDF text."""
    docs = []
    for r in results:
        metadata = {
            "Title": r.title,
            "Authors": ", ".join(a.name for a in r.authors),
            "Summary": r.summary,
            "Published": str(r.published.date()),
            "Entry ID": r.entry_id,
            "source_type": "Arxiv",
        }
        # Try full PDF text, fall back to abstract
        pdf_text = _download_pdf_text(r)
        if pdf_text:
            content = f"Title: {r.title}\n\nAbstract: {r.summary}\n\nFull Text:\n{pdf_text}"
        else:
            content = f"Title: {r.title}\n\nAbstract: {r.summary}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _build_keyword_query(raw_query: str) -> str:
    """
    Build an Arxiv API query that searches across all fields.

    Splits the user query into significant words and joins them with AND
    so each keyword must appear somewhere in the paper (title, abstract, etc.).
    """
    # Remove very short/common words that break Arxiv search
    stop_words = {"a", "an", "the", "is", "of", "in", "on", "to", "for", "and", "or", "as", "with", "by", "at"}
    words = [w for w in raw_query.strip().split() if w.lower() not in stop_words and len(w) > 1]

    if not words:
        return raw_query.strip()

    # Join with AND for each term across all fields
    return " AND ".join(f'all:"{w}"' for w in words)


def _fetch(query: str, max_results: int, sort_by=arxiv.SortCriterion.Relevance) -> list:
    """Fetch papers from Arxiv using the direct arxiv library."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
        results = list(client.results(search))
        return _results_to_documents(results)
    except Exception as e:
        print(f"[Arxiv] Error fetching documents: {e}")
        return []


def _doc_id(doc) -> str:
    """Extract a unique ID from a document for deduplication."""
    return doc.metadata.get("Entry ID", doc.metadata.get("Title", ""))


def load_arxiv_docs(query: str, max_results: int = MAX_ARXIV_DOCS) -> list:
    """
    Fetch research papers from Arxiv with a smart two-pass strategy:

    1. **Title search** first (ti: prefix) for precise paper matching.
    2. **Keyword search** across all fields to fill remaining slots with
       topically relevant papers, deduplicating against title hits.
    3. Arxiv IDs (e.g. '1706.03762') always fetch the exact paper directly.

    Args:
        query: Search query, paper title, or Arxiv paper ID
        max_results: Maximum number of papers to fetch

    Returns:
        List of LangChain Document objects with paper content and metadata
    """
    clean = query.strip()

    # Arxiv ID → fetch exact paper directly
    if _ARXIV_ID_PATTERN.match(clean):
        return _fetch(f'id:{clean}', 1) or _fetch(clean, 1)

    # Pass 1: Title-specific search
    title_query = f'ti:"{clean}"'
    title_docs = _fetch(title_query, max_results)

    # If we already have enough, return early
    if len(title_docs) >= max_results:
        return title_docs[:max_results]

    # Pass 2: Fill remaining slots with keyword search across all fields
    keyword_query = _build_keyword_query(clean)
    remaining = max_results - len(title_docs)
    keyword_docs = _fetch(keyword_query, remaining + len(title_docs))

    # Deduplicate: keep title hits, add keyword hits that aren't duplicates
    seen_ids = {_doc_id(d) for d in title_docs}
    for doc in keyword_docs:
        if len(title_docs) >= max_results:
            break
        if _doc_id(doc) not in seen_ids:
            seen_ids.add(_doc_id(doc))
            title_docs.append(doc)

    return title_docs
