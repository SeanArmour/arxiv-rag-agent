#!/usr/bin/env python3
"""
arXiv Paper Fetcher for AI/ML Research Assistant

This script fetches recent papers from arXiv in AI/ML categories and saves them
to the documents/ folder for RAG ingestion.

Categories covered:
- cs.AI (Artificial Intelligence)
- cs.CL (Computation and Language / NLP)
- cs.LG (Machine Learning)

Search terms:
- Large Language Models
- RAG (Retrieval-Augmented Generation)
- Agents
- Prompt Engineering
"""

import arxiv
import time
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text for use in filenames.

    Args:
        text: Text to sanitize
        max_length: Maximum length of sanitized text

    Returns:
        Sanitized filename-safe string
    """
    # Remove special characters, keep alphanumeric, spaces, hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    # Truncate to max length
    text = text[:max_length]
    # Remove trailing underscores
    text = text.rstrip('_')
    return text


def fetch_papers_by_query(
    query: str,
    max_results: int = 25,
    days_back: int = 90
) -> List[arxiv.Result]:
    """
    Fetch papers from arXiv based on search query.

    Args:
        query: Search query string
        max_results: Maximum number of results to fetch
        days_back: How many days back to search

    Returns:
        List of arxiv.Result objects
    """
    print(f"\n🔍 Searching for: '{query}'")
    print(f"   Fetching up to {max_results} papers from last {days_back} days...")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Construct search with date filter and categories
    search = arxiv.Search(
        query=f"{query} AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    results = []
    try:
        for result in search.results():
            # Filter by date (arxiv package doesn't support date filtering directly)
            if result.published.replace(tzinfo=None) >= start_date:
                results.append(result)

        print(f"   ✓ Found {len(results)} papers")
        return results

    except Exception as e:
        print(f"   ✗ Error fetching papers: {e}")
        return []


def save_paper(paper: arxiv.Result, output_dir: str = "documents") -> bool:
    """
    Save paper metadata and abstract to a text file.

    Args:
        paper: arxiv.Result object
        output_dir: Directory to save papers

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract arXiv ID (remove version number)
        arxiv_id = paper.entry_id.split('/')[-1].split('v')[0]

        # Create filename
        date_str = paper.published.strftime('%Y-%m-%d')
        title_clean = sanitize_filename(paper.title, max_length=60)
        filename = f"{date_str}_{arxiv_id}_{title_clean}.txt"
        filepath = os.path.join(output_dir, filename)

        # Format authors
        authors = ", ".join([author.name for author in paper.authors[:5]])
        if len(paper.authors) > 5:
            authors += f" et al. ({len(paper.authors)} total)"

        # Extract categories
        categories = ", ".join(paper.categories)

        # Create document content
        content = f"""Title: {paper.title}

Authors: {authors}

Published: {paper.published.strftime('%Y-%m-%d')}

arXiv ID: {arxiv_id}

Categories: {categories}

URL: {paper.entry_id}

PDF: {paper.pdf_url}

Abstract:
{paper.summary}

---
Source: arXiv
Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"   ✗ Error saving paper {paper.title[:50]}: {e}")
        return False


def main():
    """Main function to fetch and save papers."""

    print("=" * 80)
    print("📚 arXiv Paper Fetcher for AI/ML Research Assistant")
    print("=" * 80)

    # Create documents directory if it doesn't exist
    os.makedirs("documents", exist_ok=True)

    # Define search queries and how many papers to fetch for each
    search_queries = [
        ("large language models", 30),
        ("RAG OR retrieval augmented generation", 20),
        ("AI agents OR autonomous agents", 20),
        ("prompt engineering OR prompting", 15),
    ]

    all_papers = []
    seen_ids = set()

    # Fetch papers for each query
    for query, max_results in search_queries:
        papers = fetch_papers_by_query(query, max_results=max_results, days_back=90)

        # Deduplicate by arXiv ID
        for paper in papers:
            arxiv_id = paper.entry_id.split('/')[-1].split('v')[0]
            if arxiv_id not in seen_ids:
                all_papers.append(paper)
                seen_ids.add(arxiv_id)

        # Be respectful to arXiv API - add delay between queries
        time.sleep(3)

    print(f"\n📊 Total unique papers found: {len(all_papers)}")

    # Save papers
    print(f"\n💾 Saving papers to documents/ folder...")
    saved_count = 0

    for i, paper in enumerate(all_papers, 1):
        if save_paper(paper):
            saved_count += 1
            if i % 10 == 0:
                print(f"   Saved {i}/{len(all_papers)} papers...")

    print(f"\n✅ Successfully saved {saved_count}/{len(all_papers)} papers!")
    print(f"📁 Papers saved to: documents/")
    print("\n" + "=" * 80)
    print("Next step: Run 'python main.py' to start the RAG chatbot")
    print("=" * 80)


if __name__ == "__main__":
    main()
