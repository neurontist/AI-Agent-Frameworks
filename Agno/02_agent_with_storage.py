"""
agent_with_storage.py
---------------------

This module configures and launches a junior data scientist agent using the Agno framework. The agent is designed to assist users with beginner-friendly data analysis tasks, including loading, cleaning, exploring, visualizing, and modeling datasets, with a focus on CSV data.

Key Components:
    - Agent: The main conversational agent, equipped with data science tools and persistent memory.
    - SqliteDb: Provides persistent storage for agent sessions and conversation history.
    - OpenRouter: Specifies the language model backend for the agent.
    - AgentOS: Orchestrates the agent and exposes it as an application interface.

Attributes:
    agent_db (SqliteDb): SQLite database instance for agent storage.
    instructions (str): Multi-line string detailing the agent's responsibilities and workflow.
    agent (Agent): Configured agent instance with tools for pandas, CSV, and visualization.
    agent_os (AgentOS): Operating system abstraction for managing the agent.
    app_os: Application instance generated from the agent OS.

Usage:
    Run this module as the main program to start the agent service with hot-reloading enabled. The agent can load and analyze the 'Student_Performance.csv' dataset, perform EDA, visualize data, and prepare it for modeling. Example queries are provided in commented code for interactive sessions.

Notes:
    - The agent is intended for beginner-friendly data science tasks.
    - All data operations should utilize the provided tools (PandasTools, CsvTools, VisualizationTools).
    - The agent maintains conversational context and session history using SQLite.
"""

# Import core Agno framework components and data science tools
from agno.agent import Agent
from agno.tools.visualization import VisualizationTools
from agno.tools.pandas import PandasTools
from agno.tools.csv_toolkit import CsvTools
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.os import AgentOS


# Initialize persistent SQLite database for agent session and history storage
agent_db = SqliteDb(db_file="tmp/agent_storage.db")


# Agent instructions: define the agent's workflow and responsibilities
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


# Instantiate the data scientist agent with relevant tools and configuration
agent = Agent(
    name="Junior Data Scientist Agent 2.0",  # Agent's display name
    instructions=instructions,  # Task instructions for the agent
    model=OpenRouter(id="z-ai/glm-4.6v"),  # Language model backend
    db=agent_db,  # Persistent storage for sessions/history
    tools=[
        PandasTools(),  # Tool for pandas-based data operations
        CsvTools(csvs=["./docs/Student_Performance.csv"]),  # Tool for CSV file operations
        VisualizationTools(
            output_dir="visualizations"
        ),  # Tool for generating visualizations
    ],
    add_history_to_context=True,  # Include conversation history in context
    num_history_runs=5,  # Number of previous runs to include
    markdown=True,  # Format responses in Markdown
)


# Create the agent operating system abstraction
agent_os = AgentOS(
    id="data_science_os",  # Unique identifier for the OS
    description="A data scientist for analyzing and manipulating data.",
    agents=[agent],  # List of agents managed by this OS
)


# Expose the application interface for serving
app_os = agent_os.get_app()


if __name__ == "__main__":
    # Start the agent service with hot-reloading enabled
    agent_os.serve(app="agent_with_storage:app_os", reload=True)

    # Example interactive session (uncomment to use in scripts or notebooks):
    # session_id = "data_science_session_1"
    # agent.print_response(
    #     "Which school type has maximum study hours?",
    #     session_id=session_id,
    #     stream=True,
    # )
    # agent.print_response(
    #     "Compare that to school with minimum study hours.",
    #     session_id=session_id,
    #     stream=True,
    # )
    # agent.print_response(
    #     "Aggregate the average scores for both the school types.",
    #     session_id=session_id,
    #     stream=True,
    # )
