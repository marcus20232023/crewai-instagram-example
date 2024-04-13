import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from instagram.tools.search import SearchTools
from langchain_openai import ChatOpenAI
# Uncomment the following line to use an example of a custom tool
# from instagram.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

Togetherllm = ChatOpenAI(temperature=0.7,base_url='https://api.together.xyz/v1', api_key=TOGETHER_API_KEY, model="NOUSRESEARCH/NOUS-HERMES-2-MIXTRAL-8X7B-SFT")


@CrewBase
class InstagramCrew:
    """Instagram crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def market_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["market_researcher"],
            tools=[
              SearchTools.search_internet,
              SearchTools.search_instagram,
              SearchTools.open_page,
            ],
            llm=Togetherllm,
            verbose=True,
        )

    @agent
    def content_strategist(self) -> Agent:
        return Agent(config=self.agents_config["content_strategist"], llm=Togetherllm, verbose=True)

    @agent
    def visual_creator(self) -> Agent:
        return Agent(
            config=self.agents_config["visual_creator"],
            verbose=True,
            allow_delegation=False,
            llm=Togetherllm,
        )

    @agent
    def copywriter(self) -> Agent:
        return Agent(config=self.agents_config["copywriter"], llm=Togetherllm,verbose=True)

    @task
    def market_research(self) -> Task:
        return Task(
            config=self.tasks_config["market_research"],
            agent=self.market_researcher(),
            llm=Togetherllm,
            output_file="market_research.md",
        )

    @task
    def content_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_strategy"],
            agent=self.content_strategist(),
        )

    @task
    def visual_content_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["visual_content_creation"],
            agent=self.visual_creator(),
            output_file="visual-content.md",
        )

    @task
    def copywriting_task(self) -> Task:
        return Task(
            config=self.tasks_config["copywriting"],
            agent=self.copywriter(),
        )

    @task
    def report_final_content_strategy(self) -> Task:
        return Task(
            config=self.tasks_config["report_final_content_strategy"],
            agent=self.content_strategist(),
            output_file="final-content-strategy.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Instagram crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            llm=Togetherllm,
            verbose=2,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
