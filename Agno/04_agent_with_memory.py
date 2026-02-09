"""
This example shows how to give your agent memory of user preferences.
The agent remembers facts about you across all conversations.

Different from storage (which persists conversation history), memory
persists user-level information: preferences, facts, context.

Key concepts:
- MemoryManager: Extracts and stores user memories from conversations
- enable_agentic_memory: Agent decides when to store/recall via tool calls (efficient)
- enable_user_memories: Memory manager runs after every response (guaranteed capture)
- user_id: Links memories to a specific user


"""

from agno.agent import Agent
from agno.tools.visualization import VisualizationTools
from agno.tools.pandas import PandasTools
from agno.tools.csv_toolkit import CsvTools
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from agno.os import AgentOS
from rich.pretty import pprint

agent_db = SqliteDb(db_file="tmp/agent_storage.db")
memory_manager = MemoryManager(
    model=OpenRouter(id="z-ai/glm-4.6v"),
    db=agent_db,
    additional_instructions="""
    Capture the user's behaviours, interests, their preferences, and their goals.
    """,
)

instructions = """

You are a junior data scientist agent. Your responsibilities include:

1. Loading and inspecting datasets using Pandas and CSV tools.
2. Cleaning data: handle missing values, remove duplicates, and ensure data consistency.
3. Performing exploratory data analysis (EDA):
	- Generate summary statistics (mean, median, mode, etc.).
	- Identify data types and distributions.
	- Detect outliers and anomalies.
4. Visualizing data using the visualization tools:
	- Create histograms, box plots, scatter plots, and bar charts.
	- Use appropriate plots to explore relationships between variables.
5. Preparing data for modeling:
	- Encode categorical variables.
	- Normalize or scale features if needed.
6. Building and evaluating simple models (e.g., linear regression, classification) if required.
7. Reporting findings in a clear, concise, and beginner-friendly manner.

Always use the provided tools (PandasTools, CsvTools, visualization) for your tasks. If you are unsure, ask for clarification or suggest next steps.
"""

user_id = "abc@example.com"

agent = Agent(
    name="Junior Data Scientist Agent",
    instructions=instructions,
    model=OpenRouter(id="z-ai/glm-4.6v"),
    db=agent_db,
    memory_manager=memory_manager,
    enable_user_memories=True,
    tools=[
        PandasTools(enable_create_pandas_dataframe=False),
        CsvTools(csvs=["./docs/Student_Performance.csv"]),
        VisualizationTools(output_dir="visualizations"),
    ],
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)


if __name__ == "__main__":
    agent.print_response(
        "I prefer visualizations over tables. I like visualizations that are easy to interpret. Give me summary statistics for the subject wise performance of the students.",
        user_id=user_id,
        stream=True,
    )

    # The agent now knows your preferences
    agent.print_response(
        "Can you show me a summary of the Student_Performance.csv dataset?",
        user_id=user_id,
        stream=True,
    )

    memories = agent.get_user_memories(user_id=user_id)
    print("\n" + "=" * 60)
    print("Stored Memories:")
    print("=" * 60)
    pprint(memories)