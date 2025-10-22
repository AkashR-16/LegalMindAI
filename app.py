import os
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.models.groq import Groq
from agno.embedder.ollama import OllamaEmbedder
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.playground import Playground, serve_playground_app

from dotenv import load_dotenv

load_dotenv()
QDRANT_URL_LOCAL = os.getenv('QDRANT_URL_LOCALHOST')

#create vector db
vector_db = Qdrant(
    collection="legalmindai",
    url=QDRANT_URL_LOCAL, 
    embedder=OpenAIEmbedder(),
    #embedder = OllamaEmbedder(),
)

#create knowledge base
path="./legal_documents"

knowledge_base = PDFKnowledgeBase(
    path=path,
    vector_db=vector_db,
)

# Load the knowledge base: Comment after first run as the knowledge base is already loaded

#knowledge_base.load()

   

# Create an agent
legal_agent = Agent(
    name="Legal Agent",
    #model=Ollama(id="mistral"),
    model=OpenAIChat(id="o3-mini"),
    knowledge=knowledge_base,
    description="You are a helpful Legal Agent called 'Legal Agent' and your goal is to assist the user in the best way possible. Provide information on legal documents and answer any legal queries. Also provide the user with the sources of the information.",
    instructions=[
        "1. Knowledge Base Search:",
        "   - ALWAYS start by searching the knowledge base using search_knowledge_base tool",
        "   - Analyze ALL returned documents thoroughly before responding",
        "   - If multiple documents are returned, synthesize the information coherently",
        "2. Relevance:",
        "   - If knowledge base search yields insufficient results, just respond to the user with 'I am sorry, I could not find any relevant information on this topic.' Strictly follow the above. Do not answer general questions that are not related to the knowledge base. ",
        "3. Context Management:",
        "   - Use get_chat_history tool to maintain conversation continuity",
        "   - Reference previous interactions when relevant",
        "   - Keep track of user preferences and prior clarifications",
        "4. Response Quality:",
        "   - Provide specific citations and sources for claims",
        "   - Structure responses with clear sections and bullet points when appropriate",
        "   - Include relevant quotes from source materials",
        "   - Avoid hedging phrases like 'based on my knowledge' or 'depending on the information'",
        "5. User Interaction:",
        "   - Ask for clarification if the query is ambiguous",
        "   - Break down complex questions into manageable parts",
        "   - Proactively suggest related topics or follow-up questions",
        "6. Error Handling:",
        "   - If no relevant information is found, clearly state this",
        "   - Suggest alternative approaches or questions",
        "   - Be transparent about limitations in available information",
    ],
    search_knowledge=True,
    markdown=True,
    storage=SqliteAgentStorage(table_name="agno", db_file="agents_rag.db"),
    show_tool_calls=True,
    #debug_mode=True,
)

'''

# Test the response
try:
    legal_agent.print_response(
        "tell me about the Provision for carriage of goods of dangerous or hazardous nature to human life",
        stream=True,
    )
except Exception as e:
    print(f"An error occurred: {e}")

'''




app = Playground(agents=[legal_agent]).get_app()

if __name__ == "__main__":
    # Load the knowledge base: Comment after first run as the knowledge base is already loaded
    #knowledge_base.load(upsert=True)
    serve_playground_app("app:app", reload=True)


