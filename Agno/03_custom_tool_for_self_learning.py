from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tools.hackernews import HackerNewsTools
from agno.knowledge import Knowledge
from agno.models.openrouter import OpenRouter
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.os import AgentOS


agent_db = SqliteDb(db_file="tmp/agent_storage.db")

vec_db = LanceDb(
    uri="tmp/lancedb_self_learning",
    embedder=SentenceTransformerEmbedder(),
    search_type=SearchType.hybrid,
    name="self_learning_embeddings",
    table_name="self_learning_table",
)

learning_kb = Knowledge(
    name="self_learning_kb",
    description="A knowledge base for self-learning resources including articles, tutorials, and documentation on various topics.",
    vector_db=vec_db,
    max_results=5,
    contents_db=agent_db,
)


def self_learning(title: str, learning: str):
    """Custom tool to save new learnings to the knowledge base."""
    content = f"Title: {title}\nLearning: {learning}"
    learning_kb.add_content(name=title, text_content=content, skip_if_exists=True)
    return f"Learning titled '{title}' has been saved to the knowledge base."


instructions = """\
You are a News Reporter Agent that learns and improves over time.

You have two special abilities:
1. Search your knowledge base for previously saved learnings
2. Save new insights using the save_learning tool

## Workflow

1. Check Knowledge First
   - Before answering, search for relevant prior learnings
   - Apply any relevant insights to your response

2. Gather Information
   - Use HackerNews tools for news, articles.
   - Combine with your knowledge base insights

3. Propose Learnings
   - After answering, consider: is there a reusable insight here?
   - If yes, propose it in this format:

---
**Proposed Learning**

Title: [concise title]
Learning: [the insight — specific and actionable]

Save this? (yes/no)
---

- Only call save_learning AFTER the user says "yes"
- If user says "no", acknowledge and move on

## What Makes a Good Learning

- Specific: "Tech P/E ratios typically range 20-35x" not "P/E varies"
- Actionable: Can be applied to future questions
- Reusable: Useful beyond this one conversation

Don't save: Raw data, one-off facts, or obvious information.\
"""


agent = Agent(
    name="News Reporter Agent",  # Agent's display name
    instructions=instructions,  # Task instructions for the agent
    model=OpenRouter(id="z-ai/glm-4.6v"),  # Language model backend
    db=agent_db,  # Persistent storage for sessions/history
    knowledge=learning_kb,  # Knowledge base for self-learning
    tools=[
        HackerNewsTools(),  # Tool for accessing Hacker News data
        self_learning # Custom tool for saving learnings
    ],
    search_knowledge=True,  # Always search knowledge base first
    add_history_to_context=True,  # Include conversation history in context
    num_history_runs=5,  # Number of previous runs to include
    markdown=True,  # Format responses in Markdown
)

agent_os = AgentOS(
    id="self_learning_os",  # Unique identifier for the OS
    description="A news reporter agent that learns and improves over time by saving and reusing insights.",
    agents=[agent],  # List of agents managed by this OS
)

app_os = agent_os.get_app()

if __name__ == "__main__":
    # Start the agent service with hot-reloading enabled
    agent_os.serve(app="03_custom_tool_for_self_learning:app_os", reload=True)
