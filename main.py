#!/usr/bin/env python3
"""
AI/ML Research Assistant - RAG Chatbot

A production-quality RAG chatbot that helps researchers stay current with
AI/ML research papers using arXiv, LangChain, and OpenAI.

Features:
- Semantic search over local paper database
- Real-time arXiv API queries
- Paper summarization
- Conversation memory for follow-up questions
- Intelligent agent that selects appropriate tools
"""

import os
import sys
from typing import List, Dict, Optional
from datetime import datetime

import arxiv
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# Load environment variables
load_dotenv()

# Configuration
DOCUMENTS_DIR = "documents"
CHROMA_DB_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"
TOP_K_RESULTS = 3


class ResearchAssistant:
    """Main RAG chatbot class for AI/ML research assistance."""

    def __init__(self):
        """Initialize the research assistant with RAG and agent components."""

        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Please create a .env file with your OpenAI API key")
            print("See .env.example for template")
            sys.exit(1)

        print("🤖 Initializing AI Research Assistant...")
        print("=" * 80)

        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        # Initialize vector store
        self.vectorstore = self._initialize_vectorstore()

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize agent with tools
        self.agent_executor = self._initialize_agent()

        print("✅ Assistant ready! Type 'quit' to exit\n")
        print("=" * 80)

    def _initialize_vectorstore(self) -> Chroma:
        """
        Initialize or load the Chroma vector database.

        Returns:
            Chroma vectorstore instance
        """
        # Check if vector database already exists
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            print(f"📂 Loading existing vector database from {CHROMA_DB_DIR}/")
            try:
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self.embeddings
                )
                # Test if it has documents
                test_results = vectorstore.similarity_search("test", k=1)
                if test_results:
                    print(f"   ✓ Loaded vector database with existing documents")
                    return vectorstore
                else:
                    print(f"   ⚠ Database exists but is empty, rebuilding...")
            except Exception as e:
                print(f"   ⚠ Error loading database: {e}")
                print(f"   Rebuilding vector database...")

        # Build new vector database
        return self._build_vectorstore()

    def _build_vectorstore(self) -> Chroma:
        """
        Build vector database from documents folder.

        Returns:
            Chroma vectorstore instance
        """
        print(f"📚 Loading papers from {DOCUMENTS_DIR}/")

        # Check if documents exist
        if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
            print(f"   ⚠ No documents found in {DOCUMENTS_DIR}/")
            print(f"   Run 'python load_papers.py' first to fetch papers")
            print(f"   Creating empty vector database for now...")

            # Create empty vectorstore
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            return vectorstore

        # Load documents
        loader = DirectoryLoader(
            DOCUMENTS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        try:
            documents = loader.load()
            print(f"   ✓ Loaded {len(documents)} papers")
        except Exception as e:
            print(f"   ✗ Error loading documents: {e}")
            # Create empty vectorstore
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings
            )
            return vectorstore

        # Split documents into chunks
        print(f"✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks")

        # Create embeddings and vector store
        print(f"🧠 Creating embeddings and building vector database...")
        print(f"   (This may take a minute...)")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )

        print(f"   ✓ Vector database created and saved to {CHROMA_DB_DIR}/")

        return vectorstore

    def _initialize_agent(self) -> AgentExecutor:
        """
        Initialize LangChain agent with tools.

        Returns:
            AgentExecutor instance
        """
        print(f"🔧 Initializing agent with tools...")

        # Define tools
        tools = [
            Tool(
                name="PaperSearch",
                func=self._search_papers,
                description=(
                    "Search the local knowledge base of AI/ML research papers. "
                    "Use this to find papers on specific topics from the papers "
                    "we've already collected. Input should be a search query. "
                    "Returns the top 3 most relevant papers with metadata."
                )
            ),
            Tool(
                name="FetchLatestPapers",
                func=self._fetch_latest_papers,
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
                func=self._summarize_paper,
                description=(
                    "Fetch and summarize a specific paper from arXiv by its arXiv ID. "
                    "Use this when the user provides an arXiv ID (like 'arxiv:2401.12345' "
                    "or just '2401.12345') or asks for details about a specific paper. "
                    "Input should be the arXiv ID. Returns the paper's metadata and summary."
                )
            ),
        ]

        # Create ReAct agent prompt
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

        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        print(f"   ✓ Agent initialized with {len(tools)} tools")

        return agent_executor

    def _search_papers(self, query: str) -> str:
        """
        Search local vector database for relevant papers.

        Args:
            query: Search query

        Returns:
            Formatted string with search results
        """
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(query, k=TOP_K_RESULTS)

            if not results:
                return "No papers found in the local database for this query."

            # Format results
            output = f"Found {len(results)} relevant papers:\n\n"

            for i, doc in enumerate(results, 1):
                # Extract metadata from document content
                content = doc.page_content
                lines = content.split('\n')

                # Parse paper info
                title = "Unknown"
                authors = "Unknown"
                date = "Unknown"
                arxiv_id = "Unknown"
                url = "Unknown"

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

                # Extract abstract snippet
                abstract_start = content.find("Abstract:")
                if abstract_start != -1:
                    abstract = content[abstract_start:].replace("Abstract:", "").strip()
                    abstract = abstract.split("---")[0].strip()
                    # Truncate if too long
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

    def _fetch_latest_papers(self, query: str) -> str:
        """
        Fetch latest papers from arXiv API in real-time.

        Args:
            query: Search query

        Returns:
            Formatted string with latest papers
        """
        try:
            # Search arXiv
            search = arxiv.Search(
                query=f"{query} AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)",
                max_results=5,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            results = list(search.results())

            if not results:
                return f"No recent papers found on arXiv for: {query}"

            # Format results
            output = f"Latest {len(results)} papers from arXiv on '{query}':\n\n"

            for i, paper in enumerate(results, 1):
                arxiv_id = paper.entry_id.split('/')[-1].split('v')[0]
                authors = ", ".join([a.name for a in paper.authors[:3]])
                if len(paper.authors) > 3:
                    authors += " et al."

                # Truncate abstract
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

    def _summarize_paper(self, arxiv_id: str) -> str:
        """
        Fetch and summarize a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv ID (with or without 'arxiv:' prefix)

        Returns:
            Formatted paper summary
        """
        try:
            # Clean arXiv ID
            arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()

            # Search for the paper
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results(), None)

            if not paper:
                return f"Paper not found with arXiv ID: {arxiv_id}"

            # Format authors
            authors = ", ".join([a.name for a in paper.authors[:5]])
            if len(paper.authors) > 5:
                authors += f" et al. ({len(paper.authors)} total authors)"

            # Format categories
            categories = ", ".join(paper.categories)

            # Build summary
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

    def chat(self, user_input: str) -> str:
        """
        Process user input and return assistant response.

        Args:
            user_input: User's question or request

        Returns:
            Assistant's response
        """
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "I'm not sure how to respond to that.")
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def run_cli(self):
        """Run the interactive command-line interface."""

        print("\n💬 Chat with the AI Research Assistant")
        print("=" * 80)
        print("\nExample queries:")
        print('  - "What are recent papers about prompt engineering?"')
        print('  - "Find papers on RAG published in the last month"')
        print('  - "Summarize paper arxiv:2401.12345"')
        print('  - "What are the top papers about fine-tuning LLMs?"')
        print("\nType 'quit' to exit\n")
        print("=" * 80)

        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                # Check for exit command
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Happy researching!")
                    break

                # Get response
                print("\n🤖 Assistant:")
                response = self.chat(user_input)
                print(f"\n{response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy researching!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point for the research assistant."""
    try:
        assistant = ResearchAssistant()
        assistant.run_cli()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
