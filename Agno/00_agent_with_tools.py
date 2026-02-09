from agno.agent import Agent
from agno.tools.visualization import VisualizationTools
from agno.tools.pandas import PandasTools
from agno.tools.csv_toolkit import CsvTools
from agno.models.openrouter import OpenRouter

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

agent = Agent(
    name="Junior Data Scientist Agent",
    instructions=instructions,
    model=OpenRouter(id="z-ai/glm-4.6v"),
    tools=[
        PandasTools(),
        CsvTools(csvs=["./docs/Student_Performance.csv"]),
        VisualizationTools(output_dir="visualizations"),
    ],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "Give me summary statistics for the subject wise performance of the students.",
        stream=True,
    )
