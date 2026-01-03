# AI/ML Research Assistant - RAG Chatbot

A production-quality RAG (Retrieval-Augmented Generation) chatbot that helps researchers, students, and ML practitioners stay current with the latest AI/ML research papers from arXiv.

## Project Overview

This research assistant combines the power of:
- **arXiv API** for accessing the latest research papers
- **LangChain** for building the agent and RAG pipeline
- **OpenAI GPT-4** for intelligent responses and tool selection
- **Chroma vector database** for semantic search over papers
- **Conversation memory** for context-aware follow-up questions

### Key Capabilities

- **Semantic Search**: Find relevant papers from your local knowledge base using natural language queries
- **Real-time arXiv Integration**: Fetch the very latest papers directly from arXiv
- **Paper Summarization**: Get detailed summaries of specific papers by arXiv ID
- **Intelligent Agent**: Automatically selects the right tool based on your question
- **Conversation Memory**: Ask follow-up questions like "tell me more about the second paper"
- **Production-Ready**: Error handling, persistent storage, clean CLI interface

---

## Why This Matters

### The Problem
The AI/ML research landscape moves incredibly fast:
- **100+ papers published daily** on arXiv in AI/ML categories alone
- Keeping up with the latest developments is nearly impossible
- Finding relevant papers for your research or projects is time-consuming
- Traditional search methods don't understand context or semantics

### The Solution
This RAG chatbot:
- **Prevents hallucinations** by grounding responses in real research papers
- **Saves hours** of manual paper searching and filtering
- **Provides semantic search** to find papers based on concepts, not just keywords
- **Demonstrates enterprise AI patterns** that are valuable for production systems
- **Automates knowledge base creation** from unstructured research papers

### Real-World Applications
- Literature review for graduate research
- Staying current for job interviews in ML/AI roles
- Finding papers relevant to your current projects
- Discovering new techniques and methodologies
- Building a personal research knowledge base

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
│              "What are recent papers on RAG?"                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangChain Agent (GPT-4)                       │
│            Decides which tool(s) to use                          │
└─────┬───────────────┬───────────────┬───────────────────────────┘
      │               │               │
      ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Paper   │   │  Fetch   │   │  Summarize   │
│  Search  │   │  Latest  │   │    Paper     │
│  (RAG)   │   │  Papers  │   │  (arXiv ID)  │
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │              │                 │
     ▼              ▼                 ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Chroma  │   │  arXiv   │   │    arXiv     │
│ Vector   │   │   API    │   │     API      │
│   DB     │   │ (Search) │   │  (Fetch ID)  │
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │              │                 │
     └──────────────┴─────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Formatted Response  │
         │  with Paper Metadata │
         └──────────────────────┘


Data Flow:
┌─────────────┐
│  arXiv API  │ ──────► Fetch 50-100 papers
└─────────────┘
      │
      ▼
┌─────────────┐
│ documents/  │ ──────► Save as .txt files
└─────────────┘
      │
      ▼
┌─────────────┐
│Text Splitter│ ──────► 500 char chunks, 100 overlap
└─────────────┘
      │
      ▼
┌─────────────┐
│  Embeddings │ ──────► OpenAI text-embedding-3-small
└─────────────┘
      │
      ▼
┌─────────────┐
│  Chroma DB  │ ──────► Persistent vector store
└─────────────┘
      │
      ▼
┌─────────────┐
│ Retrieval   │ ──────► Top 3 relevant papers
└─────────────┘
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SeanArmour/arxiv-rag-agent.git
cd arxiv-rag-agent
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# .env should contain:
# OPENAI_API_KEY=your_actual_api_key_here
```

5. **Fetch initial papers**
```bash
python load_papers.py
```

This will fetch 50-100 recent AI/ML papers from arXiv and save them to the `documents/` folder. This takes about 2-3 minutes.

6. **Start the chatbot**
```bash
python main.py
```

The first run will create the vector database (takes ~1 minute). Subsequent runs will load the existing database instantly.

---

## Example Usage

### Example 1: Search Local Database

```
🧑 You: What are recent papers about prompt engineering?

🤖 Assistant:

Found 3 relevant papers:

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   Authors: Jason Wei, Xuezhi Wang, Dale Schuurmans et al.
   Date: 2023-11-15
   arXiv: 2201.11903
   Link: https://arxiv.org/abs/2201.11903
   Summary: We explore how generating a chain of thought—a series of
   intermediate reasoning steps—significantly improves the ability of
   large language models to perform complex reasoning...

2. **Automatic Prompt Engineering for Large Language Models**
   Authors: Zhou et al.
   Date: 2023-10-22
   arXiv: 2211.01910
   Link: https://arxiv.org/abs/2211.01910
   Summary: This paper introduces automatic prompt engineering (APE) to
   automatically generate and select optimal prompts for various tasks...

3. **Prompting is Programming: A Query Language for Large Language Models**
   Authors: Beurer-Kellner, Fischer, Vechev
   Date: 2023-09-30
   arXiv: 2212.06094
   Link: https://arxiv.org/abs/2212.06094
   Summary: We propose a framework that treats prompting as a programming
   paradigm, introducing LMQL (Language Model Query Language)...
```

### Example 2: Fetch Latest Papers

```
🧑 You: Find papers on RAG published in the last month

🤖 Assistant:

Latest 5 papers from arXiv on 'RAG':

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
   Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.
   Date: 2024-01-15
   arXiv: 2005.11401
   Link: https://arxiv.org/abs/2005.11401
   Summary: Large pre-trained language models have been shown to store
   factual knowledge in their parameters. However, their ability to
   access and precisely manipulate knowledge is still limited...

2. **Self-RAG: Learning to Retrieve, Generate, and Critique**
   Authors: Akari Asai, Zeqiu Wu, Yizhong Wang et al.
   Date: 2024-01-12
   arXiv: 2310.11511
   Link: https://arxiv.org/abs/2310.11511
   Summary: We introduce Self-RAG, a framework that trains an LM to
   adaptively retrieve passages on-demand and generate text...
```

### Example 3: Summarize Specific Paper

```
🧑 You: Summarize paper arxiv:2310.11511

🤖 Assistant:

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**

**Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi

**Published:** 2023-10-17

**Categories:** cs.CL, cs.AI, cs.LG

**arXiv ID:** 2310.11511

**Links:**
- Paper: https://arxiv.org/abs/2310.11511
- PDF: https://arxiv.org/pdf/2310.11511.pdf

**Abstract:**
Despite their remarkable capabilities, large language models (LLMs) often
produce responses containing factual inaccuracies due to their sole reliance
on parametric knowledge. Retrieval-Augmented Generation (RAG) addresses this
limitation by incorporating external knowledge...
```

### Example 4: Follow-up Questions (Conversation Memory)

```
🧑 You: What are papers about LLM agents?

🤖 Assistant: [Returns 3 papers about agents]

🧑 You: Tell me more about the second paper

🤖 Assistant: [Provides detailed information about the second paper from
the previous response, demonstrating conversation memory]
```

---

## Technical Implementation

### Document Chunking Strategy

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 100 characters
- **Rationale**: Paper abstracts are relatively short (200-500 words). Smaller chunks ensure that semantically similar content is retrieved together without losing context. The overlap ensures important information at chunk boundaries isn't lost.

### Embedding Model

- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimensions**: 1536
- **Why**: Excellent balance of quality and cost. Optimized for semantic search tasks and handles academic language well.

### Agent Type

- **Type**: ReAct (Reasoning + Acting) Agent
- **LLM**: GPT-4
- **Why**: ReAct agents explicitly show their reasoning process, making it easy to debug and understand tool selection. GPT-4 provides the best instruction following for complex tool use.

### Tool Descriptions

1. **PaperSearch**
   - Performs semantic similarity search over the local Chroma vector database
   - Returns top 3 most relevant papers based on cosine similarity of embeddings
   - Fast (~100ms) and works offline

2. **FetchLatestPapers**
   - Queries arXiv API in real-time using the `arxiv` Python package
   - Filters by categories: cs.AI, cs.CL, cs.LG
   - Sorted by submission date (newest first)
   - Returns up to 5 papers

3. **SummarizePaper**
   - Fetches a specific paper by arXiv ID
   - Returns complete metadata: title, authors, abstract, links
   - Useful for getting full details on a paper mentioned in search results

### Conversation Memory

- **Type**: `ConversationBufferMemory`
- **Stores**: Full conversation history in memory
- **Enables**: Follow-up questions, context awareness, referencing previous papers
- **Limitation**: Buffer memory has no size limit, so very long conversations could hit token limits (future enhancement: use `ConversationSummaryMemory`)

---

## Future Enhancements

Ideas for extending this project (not implemented in MVP):

### Content Extraction
- **Full PDF text extraction**: Use PyPDF2 or pdfplumber to extract full paper content, not just abstracts
- **Section-specific search**: Search only methods, results, or conclusions
- **Figure and table extraction**: Include visual content in the knowledge base

### Advanced Features
- **Citation graph analysis**: Understand paper relationships and influence
- **Author tracking**: Follow specific researchers and their work
- **Auto-update papers daily**: Cron job to fetch new papers automatically
- **Export reading list**: Save interesting papers to Notion, Zotero, or BibTeX
- **Paper comparison**: Side-by-side comparison of related papers
- **Trend analysis**: Identify hot topics and emerging areas

### Integrations
- **Slack/Discord bot**: Make the assistant available in team chat
- **Web interface**: Build a Streamlit or Gradio UI
- **Conference filtering**: Filter by specific venues (NeurIPS, ICML, ACL, etc.)
- **Arxiv categories**: Support more categories beyond AI/ML

### Quality Improvements
- **Use Claude for long context**: Summarize entire papers (100+ pages) using Claude's 200K context
- **Re-ranking**: Use cross-encoder for better retrieval accuracy
- **Query expansion**: Automatically expand queries with synonyms and related terms
- **Feedback loop**: Track which papers users find helpful to improve recommendations

### Personalization
- **Reading history**: Track papers you've read vs. to-read
- **Custom collections**: Organize papers by project or topic
- **Recommendation system**: Suggest papers based on reading history

---

## Use Cases

### For Graduate Students
- **Literature reviews**: Quickly find all relevant papers on your research topic
- **Staying current**: Keep up with the latest developments in your field
- **Finding gaps**: Identify underexplored areas by analyzing paper trends

### For ML Engineers
- **Project research**: Find papers relevant to your current work projects
- **Learning new techniques**: Discover state-of-the-art methods for specific problems
- **Technical interviews**: Stay current with latest developments for job interviews

### For Research Scientists
- **Paper discovery**: Find papers you might have missed in your area
- **Cross-domain insights**: Discover techniques from adjacent fields
- **Collaboration**: Identify potential collaborators working on similar problems

### For Product Managers
- **Technology trends**: Understand where AI/ML research is heading
- **Feasibility assessment**: Find research validating potential features
- **Competitive analysis**: Track what research competitors might be leveraging

---

## Project Structure

```
arxiv-rag-agent/
├── documents/              # Stores fetched paper abstracts (.txt files)
├── chroma_db/             # Persistent vector database (gitignored)
├── load_papers.py         # Script to fetch papers from arXiv
├── main.py                # Main RAG agent chatbot
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .env                   # Your API keys (gitignored)
├── .gitignore            # Git ignore file
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## Technology Stack

- **Python 3.8+**: Core language
- **LangChain**: Agent framework and RAG pipeline
- **OpenAI API**: GPT-4 for reasoning, embeddings for semantic search
- **Chroma**: Vector database for persistent storage
- **arXiv API**: Research paper source
- **python-dotenv**: Environment variable management

---

## Performance Considerations

- **Initial database build**: ~1 minute for 100 papers
- **Query latency**:
  - Local search: ~100ms
  - arXiv API call: ~1-2 seconds
  - Full agent response: ~3-5 seconds
- **Cost**:
  - Embeddings: ~$0.002 per 100 papers
  - GPT-4 queries: ~$0.01-0.03 per query
  - Total for MVP: < $1 for initial setup

---

## Troubleshooting

### "OPENAI_API_KEY not found"
Make sure you've created a `.env` file with your OpenAI API key. See Setup Instructions step 4.

### "No documents found in documents/"
Run `python load_papers.py` first to fetch papers from arXiv.

### "Rate limit exceeded"
You've hit OpenAI's API rate limits. Wait a few minutes or upgrade your API tier.

### "Error loading database"
Delete the `chroma_db/` folder and run `python main.py` again to rebuild the database.

---

## Contributing

Contributions are welcome! Some ideas:
- Add support for more paper sources (bioRxiv, ACL Anthology, etc.)
- Improve paper parsing and chunking strategies
- Add evaluation metrics for retrieval quality
- Build a web interface
- Add unit tests

---

## License

MIT License - feel free to use this for your own research and projects!

---

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Papers sourced from [arXiv](https://arxiv.org/)
- Vector database powered by [Chroma](https://www.trychroma.com/)
- AI models by [OpenAI](https://openai.com/)

---

## Author

Built as a practical research tool for staying current with AI/ML developments.

**Contact**: [GitHub Issues](https://github.com/SeanArmour/arxiv-rag-agent/issues)

---

*Last updated: January 2026*
