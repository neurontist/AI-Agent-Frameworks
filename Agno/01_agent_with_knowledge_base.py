"""
agent_with_knowledge_base.py
----------------------------

This module configures and launches a friendly storyteller agent using the Agno framework. The agent is designed to narrate, summarize, and answer questions about stories, using the book as its knowledge base.

Key Components:
    - Agent: The main conversational agent, equipped with storytelling abilities and persistent memory.
    - SqliteDb: Provides persistent storage for agent sessions and conversation history.
    - Knowledge: Connects the agent to the full text of 'Grandma's Bag of Stories'.
    - AgentOS: Orchestrates the agent and exposes it as an application interface.

Attributes:
    agent_db (SqliteDb): SQLite database instance for agent storage.
    instructions (str): Multi-line string detailing the agent's storytelling responsibilities and workflow.
    agent (Agent): Configured agent instance for storytelling.
    agent_os (AgentOS): Operating system abstraction for managing the agent.
    app_os: Application instance generated from the agent OS.

Usage:
    Run this module as the main program to start the agent service with hot-reloading enabled. The agent can narrate, summarize, and answer questions about stories.

Notes:
    - The agent is intended for storytelling and book-based Q&A.
    - All story-related responses are sourced from the knowledge base (the book).
    - The agent maintains conversational context and session history using SQLite.
"""

# Import core Agno framework components and data science tools
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.models.openrouter import OpenRouter
from agno.os import AgentOS


# Initialize persistent SQLite database for agent session and history storage
agent_db = SqliteDb(db_file="tmp/agent_storage.db")


# Set up the vector database for semantic search over the book
vector_db = LanceDb(
    name="story_embeddings",
    embedder=SentenceTransformerEmbedder(),
    search_type=SearchType.hybrid,
    uri="tmp/lancedb_story_embeddings",
    table_name="knowledge_embeddings",
)


# Configure the knowledge base with the book
knowledge = Knowledge(
    name="story_knowledge_base",
    description="A knowledge base containing the full text of 5 different stories. The agent can answer questions, narrate, and summarize stories, and provide details about characters, morals, and events from the book.",
    vector_db=vector_db,
    max_results=5,
    contents_db=agent_db,
)


# Agent instructions: define the agent's storytelling workflow and responsibilities
instructions = """
You are a friendly and engaging storyteller agent with access to the full text of 'story_book.pdf', which contains multiple classic children's stories. Your responsibilities include:

1. Narrating any story from 'story_book.pdf' in a captivating and child-friendly manner when asked.
2. Answering questions about the stories, characters, morals, and events from any story in the PDF.
3. Summarizing stories or sections from the PDF in a way that is easy to understand and enjoyable.
4. Providing details, quotes, or explanations based on the content of any story in the PDF.
5. Always search the knowledge base (containing the full PDF) first for any question or storytelling request, and use information from the PDF in your responses.
6. If you cannot find the answer or story in the knowledge base, let the user know or ask for clarification.

If you are unsure, ask for clarification or suggest next steps.
"""


# Instantiate the storyteller agent with relevant configuration
agent = Agent(
    name="Storyteller Agent",  # Agent's display name
    model=OpenRouter(id="openai/gpt-5.2-chat"),  # Language model backend
    instructions=instructions,  # Storytelling instructions
    db=agent_db,  # Persistent storage for sessions/history
    knowledge=knowledge,  # Knowledge base with the book
    search_knowledge=True,  # Always search knowledge base first
    add_history_to_context=True,  # Include conversation history in context
    num_history_runs=3,  # Number of previous runs to include
    markdown=True,  # Format responses in Markdown
)


# Create the agent operating system abstraction
agent_os = AgentOS(
    id="storyteller_os",  # Unique identifier for the OS
    description="A storyteller agent for narrating and answering questions about stories.",
    agents=[agent],  # List of agents managed by this OS
)


# Expose the application interface for serving
app_os = agent_os.get_app()


if __name__ == "__main__":
    knowledge.add_content(name="story_embeddings", path="./docs/story_book.pdf")
    # Start the agent service with hot-reloading enabled
    agent_os.serve(app="01_agent_with_knowledge_base:app_os", reload=True)
