#!/usr/bin/env python3
"""
Streamlit Web Interface for AI/ML Research Assistant

A beautiful, user-friendly web interface for the RAG chatbot that makes it
easy to search papers, adjust settings, and manage your research database.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st
import arxiv
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .stat-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Configuration defaults
DEFAULT_DOCUMENTS_DIR = "documents"
DEFAULT_CHROMA_DB_DIR = "chroma_db"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K_RESULTS = 3
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4"


@st.cache_resource
def initialize_assistant(
    documents_dir: str,
    chroma_db_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model: str,
    llm_model: str
):
    """Initialize the research assistant with caching."""

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY not found. Please set it in the sidebar.")
        return None

    # Initialize components
    embeddings = OpenAIEmbeddings(model=embedding_model)
    llm = ChatOpenAI(model=llm_model, temperature=0)

    # Initialize vector store
    vectorstore = load_or_create_vectorstore(
        documents_dir, chroma_db_dir, embeddings, chunk_size, chunk_overlap
    )

    if vectorstore is None:
        return None

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create tools
    tools = create_tools(vectorstore, top_k)

    # Create agent
    agent_executor = create_agent(llm, tools, memory)

    return {
        'agent': agent_executor,
        'vectorstore': vectorstore,
        'embeddings': embeddings,
        'llm': llm,
        'memory': memory
    }


def load_or_create_vectorstore(
    documents_dir: str,
    chroma_db_dir: str,
    embeddings,
    chunk_size: int,
    chunk_overlap: int
):
    """Load existing vector store or create new one."""

    # Try to load existing database
    if os.path.exists(chroma_db_dir) and os.listdir(chroma_db_dir):
        try:
            vectorstore = Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embeddings
            )
            # Test if it has documents
            test_results = vectorstore.similarity_search("test", k=1)
            if test_results:
                return vectorstore
        except Exception as e:
            st.warning(f"Could not load existing database: {e}. Creating new one...")

    # Build new database
    return build_vectorstore(documents_dir, chroma_db_dir, embeddings, chunk_size, chunk_overlap)


def build_vectorstore(
    documents_dir: str,
    chroma_db_dir: str,
    embeddings,
    chunk_size: int,
    chunk_overlap: int
):
    """Build vector database from documents."""

    if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
        st.warning(f"⚠️ No documents found in {documents_dir}/")
        st.info("Use the 'Fetch Papers' tab to download papers first.")
        # Create empty vectorstore
        return Chroma(
            persist_directory=chroma_db_dir,
            embedding_function=embeddings
        )

    with st.spinner("📚 Loading documents..."):
        loader = DirectoryLoader(
            documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        try:
            documents = loader.load()
            st.success(f"✓ Loaded {len(documents)} papers")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            return Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embeddings
            )

    with st.spinner("✂️ Splitting documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"✓ Created {len(chunks)} chunks")

    with st.spinner("🧠 Creating embeddings (this may take a minute)..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_db_dir
        )
        st.success("✓ Vector database created!")

    return vectorstore


def create_tools(vectorstore, top_k: int) -> List[Tool]:
    """Create agent tools."""

    def search_papers(query: str) -> str:
        """Search local vector database."""
        try:
            results = vectorstore.similarity_search(query, k=top_k)
            if not results:
                return "No papers found in the local database for this query."

            output = f"Found {len(results)} relevant papers:\n\n"
            for i, doc in enumerate(results, 1):
                content = doc.page_content
                lines = content.split('\n')

                title = authors = date = arxiv_id = url = "Unknown"

                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Authors:"):
                        authors = line.replace("Authors:", "").strip()
                    elif line.startswith("Published:"):
                        date = line.replace("Published:", "").strip()
                    elif line.startswith("arXiv ID:"):
                        arxiv_id = line.replace("arXiv ID:", "").strip()
                    elif line.startswith("URL:"):
                        url = line.replace("URL:", "").strip()

                abstract_start = content.find("Abstract:")
                if abstract_start != -1:
                    abstract = content[abstract_start:].replace("Abstract:", "").strip()
                    abstract = abstract.split("---")[0].strip()
                    if len(abstract) > 300:
                        abstract = abstract[:300] + "..."
                else:
                    abstract = "No abstract available."

                output += f"{i}. **{title}**\n"
                output += f"   Authors: {authors}\n"
                output += f"   Date: {date}\n"
                output += f"   arXiv: {arxiv_id}\n"
                output += f"   Link: {url}\n"
                output += f"   Summary: {abstract}\n\n"

            return output
        except Exception as e:
            return f"Error searching papers: {str(e)}"

    def fetch_latest_papers(query: str) -> str:
        """Fetch latest papers from arXiv."""
        try:
            search = arxiv.Search(
                query=f"{query} AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)",
                max_results=5,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            results = list(search.results())
            if not results:
                return f"No recent papers found on arXiv for: {query}"

            output = f"Latest {len(results)} papers from arXiv on '{query}':\n\n"
            for i, paper in enumerate(results, 1):
                arxiv_id = paper.entry_id.split('/')[-1].split('v')[0]
                authors = ", ".join([a.name for a in paper.authors[:3]])
                if len(paper.authors) > 3:
                    authors += " et al."

                abstract = paper.summary.replace('\n', ' ').strip()
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."

                output += f"{i}. **{paper.title}**\n"
                output += f"   Authors: {authors}\n"
                output += f"   Date: {paper.published.strftime('%Y-%m-%d')}\n"
                output += f"   arXiv: {arxiv_id}\n"
                output += f"   Link: {paper.entry_id}\n"
                output += f"   Summary: {abstract}\n\n"

            return output
        except Exception as e:
            return f"Error fetching papers from arXiv: {str(e)}"

    def summarize_paper(arxiv_id: str) -> str:
        """Summarize specific paper by arXiv ID."""
        try:
            arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results(), None)

            if not paper:
                return f"Paper not found with arXiv ID: {arxiv_id}"

            authors = ", ".join([a.name for a in paper.authors[:5]])
            if len(paper.authors) > 5:
                authors += f" et al. ({len(paper.authors)} total authors)"

            categories = ", ".join(paper.categories)

            output = f"**{paper.title}**\n\n"
            output += f"**Authors:** {authors}\n\n"
            output += f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n\n"
            output += f"**Categories:** {categories}\n\n"
            output += f"**arXiv ID:** {arxiv_id}\n\n"
            output += f"**Links:**\n"
            output += f"- Paper: {paper.entry_id}\n"
            output += f"- PDF: {paper.pdf_url}\n\n"
            output += f"**Abstract:**\n{paper.summary}\n"

            return output
        except Exception as e:
            return f"Error fetching paper {arxiv_id}: {str(e)}"

    return [
        Tool(
            name="PaperSearch",
            func=search_papers,
            description=(
                "Search the local knowledge base of AI/ML research papers. "
                "Use this to find papers on specific topics from the papers "
                "we've already collected. Input should be a search query. "
                "Returns the top most relevant papers with metadata."
            )
        ),
        Tool(
            name="FetchLatestPapers",
            func=fetch_latest_papers,
            description=(
                "Query arXiv API in real-time to find the very latest papers "
                "on a specific topic. Use this when the user asks for 'recent', "
                "'latest', or 'new' papers, or when searching for something that "
                "might not be in our local database. Input should be a search query. "
                "Returns up to 5 recent papers from arXiv."
            )
        ),
        Tool(
            name="SummarizePaper",
            func=summarize_paper,
            description=(
                "Fetch and summarize a specific paper from arXiv by its arXiv ID. "
                "Use this when the user provides an arXiv ID (like 'arxiv:2401.12345' "
                "or just '2401.12345') or asks for details about a specific paper. "
                "Input should be the arXiv ID. Returns the paper's metadata and summary."
            )
        ),
    ]


def create_agent(llm, tools: List[Tool], memory):
    """Create the agent executor."""

    template = """Answer the following questions as best you can. You are an AI research assistant that helps users stay current with AI/ML research papers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When presenting papers to the user:
- Always include the title, authors, date, and arXiv link
- Format the output in a clean, readable way
- Highlight key findings from abstracts
- If multiple papers are found, number them for easy reference

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )


def get_database_stats(documents_dir: str, chroma_db_dir: str) -> Dict:
    """Get statistics about the database."""
    stats = {
        'num_documents': 0,
        'db_exists': False,
        'db_size_mb': 0
    }

    # Count documents
    if os.path.exists(documents_dir):
        stats['num_documents'] = len([f for f in os.listdir(documents_dir) if f.endswith('.txt')])

    # Check database
    if os.path.exists(chroma_db_dir) and os.listdir(chroma_db_dir):
        stats['db_exists'] = True
        # Calculate size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(chroma_db_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        stats['db_size_mb'] = round(total_size / (1024 * 1024), 2)

    return stats


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<p class="main-header">🤖 AI/ML Research Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your RAG-powered chatbot for staying current with AI research</p>', unsafe_allow_html=True)

    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Key
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )

        if api_key and api_key != os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✓ API key updated!")

        st.divider()

        # Model Settings
        st.subheader("🧠 Model Settings")

        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            index=0,
            help="GPT-4 is smarter but more expensive. GPT-3.5 is faster and cheaper."
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
            help="text-embedding-3-small offers the best balance of quality and cost."
        )

        st.divider()

        # Search Settings
        st.subheader("🔍 Search Settings")

        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=3,
            help="How many papers to return from searches"
        )

        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=1000,
            value=500,
            step=50,
            help="Size of text chunks for embeddings"
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=100,
            step=10,
            help="Overlap between chunks to preserve context"
        )

        st.divider()

        # Database Stats
        st.subheader("📊 Database Stats")
        stats = get_database_stats(DEFAULT_DOCUMENTS_DIR, DEFAULT_CHROMA_DB_DIR)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", stats['num_documents'])
        with col2:
            st.metric("DB Size", f"{stats['db_size_mb']} MB")

        if stats['db_exists']:
            st.success("✓ Vector DB ready")
        else:
            st.warning("⚠️ No vector DB")

        st.divider()

        # Quick Actions
        st.subheader("🔧 Quick Actions")

        if st.button("🔄 Rebuild Database", use_container_width=True):
            import shutil
            if os.path.exists(DEFAULT_CHROMA_DB_DIR):
                shutil.rmtree(DEFAULT_CHROMA_DB_DIR)
            st.cache_resource.clear()
            st.success("Database cleared! Refresh to rebuild.")
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📥 Fetch Papers", "📚 Browse Papers", "ℹ️ Help"])

    with tab1:
        chat_interface(llm_model, embedding_model, chunk_size, chunk_overlap, top_k)

    with tab2:
        fetch_papers_interface()

    with tab3:
        browse_papers_interface()

    with tab4:
        help_interface()


def chat_interface(llm_model, embedding_model, chunk_size, chunk_overlap, top_k):
    """Chat interface tab."""

    # Initialize assistant
    assistant = initialize_assistant(
        DEFAULT_DOCUMENTS_DIR,
        DEFAULT_CHROMA_DB_DIR,
        chunk_size,
        chunk_overlap,
        top_k,
        embedding_model,
        llm_model
    )

    if assistant is None:
        st.error("Failed to initialize assistant. Please check your settings and API key.")
        return

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Example queries
    if len(st.session_state.messages) == 0:
        st.info("👋 **Welcome!** Try asking questions like:")
        example_cols = st.columns(2)
        with example_cols[0]:
            st.markdown("""
            - *"What are recent papers about prompt engineering?"*
            - *"Find papers on RAG from the last month"*
            - *"Show me papers about LLM agents"*
            """)
        with example_cols[1]:
            st.markdown("""
            - *"Summarize paper arxiv:2310.11511"*
            - *"What's new in multimodal AI?"*
            - *"Papers by Anthropic researchers"*
            """)

    # Chat input
    if prompt := st.chat_input("Ask about AI/ML research papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = assistant['agent'].invoke({"input": prompt})
                    answer = response.get("output", "I'm not sure how to respond to that.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def fetch_papers_interface():
    """Paper fetching interface."""

    st.header("📥 Fetch Papers from arXiv")
    st.markdown("Download recent AI/ML research papers to build your knowledge base.")

    # Search configuration
    st.subheader("Search Configuration")

    col1, col2 = st.columns(2)

    with col1:
        days_back = st.number_input(
            "Days to look back",
            min_value=7,
            max_value=365,
            value=90,
            help="How far back to search for papers"
        )

    with col2:
        max_results_per_query = st.number_input(
            "Max results per query",
            min_value=5,
            max_value=50,
            value=25,
            help="Maximum papers to fetch per search query"
        )

    # Search queries
    st.subheader("Search Queries")
    st.markdown("Add topics you want to track:")

    if 'search_queries' not in st.session_state:
        st.session_state.search_queries = [
            "large language models",
            "RAG OR retrieval augmented generation",
            "AI agents OR autonomous agents",
            "prompt engineering OR prompting"
        ]

    # Display and edit queries
    for i, query in enumerate(st.session_state.search_queries):
        col1, col2 = st.columns([4, 1])
        with col1:
            new_query = st.text_input(f"Query {i+1}", value=query, key=f"query_{i}", label_visibility="collapsed")
            st.session_state.search_queries[i] = new_query
        with col2:
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.search_queries.pop(i)
                st.rerun()

    # Add new query
    if st.button("➕ Add Query"):
        st.session_state.search_queries.append("")
        st.rerun()

    st.divider()

    # Fetch button
    if st.button("🚀 Fetch Papers", type="primary", use_container_width=True):
        fetch_papers(st.session_state.search_queries, days_back, max_results_per_query)


def fetch_papers(queries: List[str], days_back: int, max_results: int):
    """Fetch papers from arXiv."""

    from datetime import timedelta
    import re

    def sanitize_filename(text: str, max_length: int = 100) -> str:
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', '_', text)
        text = text[:max_length].rstrip('_')
        return text

    progress_bar = st.progress(0)
    status_text = st.empty()

    os.makedirs(DEFAULT_DOCUMENTS_DIR, exist_ok=True)

    all_papers = []
    seen_ids = set()

    total_queries = len([q for q in queries if q.strip()])

    for idx, query in enumerate(queries):
        if not query.strip():
            continue

        status_text.text(f"🔍 Searching for: '{query}'...")
        progress_bar.progress((idx) / total_queries)

        try:
            search = arxiv.Search(
                query=f"{query} AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            for result in search.results():
                arxiv_id = result.entry_id.split('/')[-1].split('v')[0]

                if result.published.replace(tzinfo=None) >= start_date and arxiv_id not in seen_ids:
                    all_papers.append(result)
                    seen_ids.add(arxiv_id)

            time.sleep(3)  # Respect arXiv rate limits

        except Exception as e:
            st.error(f"Error fetching papers for '{query}': {e}")

    status_text.text(f"💾 Saving {len(all_papers)} papers...")
    progress_bar.progress(0.9)

    saved_count = 0
    for paper in all_papers:
        try:
            arxiv_id = paper.entry_id.split('/')[-1].split('v')[0]
            date_str = paper.published.strftime('%Y-%m-%d')
            title_clean = sanitize_filename(paper.title, max_length=60)
            filename = f"{date_str}_{arxiv_id}_{title_clean}.txt"
            filepath = os.path.join(DEFAULT_DOCUMENTS_DIR, filename)

            authors = ", ".join([author.name for author in paper.authors[:5]])
            if len(paper.authors) > 5:
                authors += f" et al. ({len(paper.authors)} total)"

            categories = ", ".join(paper.categories)

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

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            saved_count += 1
        except Exception as e:
            st.error(f"Error saving paper: {e}")

    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()

    st.success(f"✅ Successfully saved {saved_count} papers to documents/ folder!")
    st.info("💡 Now rebuild the database in the sidebar to index these papers.")


def browse_papers_interface():
    """Browse downloaded papers."""

    st.header("📚 Browse Downloaded Papers")

    if not os.path.exists(DEFAULT_DOCUMENTS_DIR):
        st.warning("No documents folder found.")
        return

    papers = [f for f in os.listdir(DEFAULT_DOCUMENTS_DIR) if f.endswith('.txt')]

    if not papers:
        st.info("No papers downloaded yet. Use the 'Fetch Papers' tab to download some!")
        return

    st.markdown(f"**Total papers: {len(papers)}**")

    # Sort options
    sort_by = st.selectbox("Sort by", ["Date (newest first)", "Date (oldest first)", "Title"])

    if sort_by == "Date (newest first)":
        papers.sort(reverse=True)
    elif sort_by == "Date (oldest first)":
        papers.sort()
    else:
        papers.sort(key=lambda x: x.split('_', 2)[2] if len(x.split('_')) > 2 else x)

    # Pagination
    papers_per_page = 10
    total_pages = (len(papers) - 1) // papers_per_page + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

    start_idx = page * papers_per_page
    end_idx = min(start_idx + papers_per_page, len(papers))

    # Display papers
    for paper_file in papers[start_idx:end_idx]:
        filepath = os.path.join(DEFAULT_DOCUMENTS_DIR, paper_file)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            title = authors = date = arxiv_id = url = "Unknown"

            for line in lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("Authors:"):
                    authors = line.replace("Authors:", "").strip()
                elif line.startswith("Published:"):
                    date = line.replace("Published:", "").strip()
                elif line.startswith("arXiv ID:"):
                    arxiv_id = line.replace("arXiv ID:", "").strip()
                elif line.startswith("URL:"):
                    url = line.replace("URL:", "").strip()

            with st.expander(f"📄 {title}"):
                st.markdown(f"**Authors:** {authors}")
                st.markdown(f"**Date:** {date}")
                st.markdown(f"**arXiv ID:** {arxiv_id}")
                st.markdown(f"**Link:** [{url}]({url})")

                if st.button("View Full Content", key=f"view_{arxiv_id}"):
                    st.text_area("Full Content", content, height=400)

        except Exception as e:
            st.error(f"Error reading {paper_file}: {e}")


def help_interface():
    """Help and documentation."""

    st.header("ℹ️ Help & Documentation")

    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Set API Key**: Enter your OpenAI API key in the sidebar
    2. **Fetch Papers**: Use the "Fetch Papers" tab to download research papers
    3. **Rebuild Database**: Click "Rebuild Database" in the sidebar to index papers
    4. **Start Chatting**: Ask questions in the Chat tab!
    """)

    st.divider()

    st.subheader("💬 Example Queries")
    st.markdown("""
    **Search local database:**
    - "What papers discuss chain-of-thought prompting?"
    - "Find papers about fine-tuning LLMs"
    - "Show me research on agent architectures"

    **Fetch latest from arXiv:**
    - "What's new on arXiv about RAG this week?"
    - "Find the most recent papers on prompt engineering"
    - "Are there any new papers about multimodal LLMs?"

    **Get specific papers:**
    - "Summarize paper arxiv:2310.11511"
    - "Tell me about 2401.12345"

    **Follow-up questions:**
    - "Tell me more about the second paper"
    - "What are the key differences between these approaches?"
    """)

    st.divider()

    st.subheader("⚙️ Settings Guide")
    st.markdown("""
    **LLM Model:**
    - **GPT-4**: Best quality, slower, more expensive (~$0.03/query)
    - **GPT-3.5-Turbo**: Good quality, faster, cheaper (~$0.001/query)

    **Number of Results:**
    - Higher values (5-10) give more context but longer responses
    - Lower values (1-3) give focused, concise answers

    **Chunk Size:**
    - Larger chunks (800-1000): Better for long context
    - Smaller chunks (300-500): Better for precise matching
    """)

    st.divider()

    st.subheader("💰 Cost Estimation")
    st.markdown("""
    - **Embedding 100 papers**: ~$0.002
    - **GPT-4 query**: ~$0.01-0.03
    - **GPT-3.5 query**: ~$0.001
    - **Typical monthly use**: $5-10
    """)

    st.divider()

    st.subheader("🔧 Troubleshooting")
    st.markdown("""
    **"No documents found":**
    - Use the "Fetch Papers" tab to download papers first

    **"Error loading database":**
    - Click "Rebuild Database" in the sidebar

    **Slow responses:**
    - Switch from GPT-4 to GPT-3.5-Turbo
    - Reduce "Number of Results"

    **Rate limit errors:**
    - Wait 60 seconds before trying again
    - Reduce query frequency
    """)

    st.divider()

    st.subheader("📚 Resources")
    st.markdown("""
    - [GitHub Repository](https://github.com/SeanArmour/arxiv-rag-agent)
    - [OpenAI API Documentation](https://platform.openai.com/docs)
    - [arXiv API](https://arxiv.org/help/api)
    - [LangChain Docs](https://python.langchain.com/)
    """)


if __name__ == "__main__":
    main()
