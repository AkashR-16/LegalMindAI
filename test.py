import os
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.playground import Playground, serve_playground_app

from dotenv import load_dotenv

load_dotenv()
QDRANT_URL_LOCAL = os.getenv('QDRANT_URL_LOCALHOST')

# Create vector db with properly configured Ollama embedder
# Ollama embedder typically expects the model name without specifying "embeddings"
# as it will handle that internally
vector_db = Qdrant(
    collection="pdf_documents",
    url=QDRANT_URL_LOCAL, 
    embedder=OllamaEmbedder(model="nomic-embed-text"),  # Using nomic-embed-text which is designed for embeddings
)

# Create knowledge base for PDF documents
path="./pdf_documents"

knowledge_base = PDFKnowledgeBase(
    path=path,
    vector_db=vector_db,
)

# Load the knowledge base: Uncomment for first run to load the knowledge base
knowledge_base.load(upsert=True)

# Create a PDF RAG agent using Ollama's Llama 3.1 model
pdf_rag_agent = Agent(
    name="PDF Document Assistant",
    model=Ollama(model="llama3.2"),  # Using the proper parameter 'model' instead of 'id'
    knowledge=knowledge_base,
    description="You are a helpful PDF Document Assistant. Your goal is to extract and provide information from the PDF documents in your knowledge base. Always cite your sources and provide direct references to the documents you're pulling information from.",
    instructions=[
        "1. PDF Knowledge Retrieval:",
        "   - ALWAYS search the PDF knowledge base first using search_knowledge_base tool",
        "   - Extract relevant sections from ALL returned PDF documents",
        "   - Properly cite document names, page numbers, and sections when possible",
        "2. Information Relevance:",
        "   - Only answer questions that can be addressed with information from the PDF knowledge base",
        "   - If the PDFs don't contain relevant information, respond: 'I don't have sufficient information about this in my PDF knowledge base.'",
        "   - Don't make up information that isn't in the documents",
        "3. Contextual Understanding:",
        "   - Use get_chat_history tool to maintain conversation continuity",
        "   - Connect information across different PDF documents when relevant",
        "   - Consider the context of the PDFs (technical, legal, educational, etc.)",
        "4. Response Formatting:",
        "   - Include direct quotes from PDFs when appropriate, using quotation marks",
        "   - Structure responses with headings and bullet points for clarity",
        "   - For technical content, preserve formatting like tables and lists when helpful",
        "   - Always include document title and page reference for each piece of information",
        "5. Clarification Process:",
        "   - If a query is ambiguous, ask for clarification about which document or section the user is interested in",
        "   - Suggest specific PDF documents that might contain the answer",
        "   - Offer to search for related terms if initial search yields limited results",
        "6. Limitations Handling:",
        "   - Be transparent about limitations in PDF parsing (e.g., tables, images, charts)",
        "   - Acknowledge when information might be incomplete due to PDF formatting issues",
        "   - Suggest alternative queries if current one doesn't yield useful results",
    ],
    search_knowledge=True,
    markdown=True,
    storage=SqliteAgentStorage(table_name="pdf_rag", db_file="pdf_rag_agent.db"),
    show_tool_calls=True,
    # Uncomment for debugging
    # debug_mode=True,
)

# Test function for the agent
def test_agent():
    try:
        pdf_rag_agent.print_response(
            "What information do you have in your PDF knowledge base?",
            stream=True,
        )
    except Exception as e:
        print(f"An error occurred: {e}")

# Setup the Playground app
app = Playground(agents=[pdf_rag_agent]).get_app()

if __name__ == "__main__":

    
    serve_playground_app("app:app", reload=True)