from phi.agent import Agent
# from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv('c:/codellm/.env')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.1-8b-instant", api_key=GROQ_API_KEY),
    # model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["You answer should only contain the information retrieved from search. Don't repeat searches. Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
# web_agent.print_response("Whats happening in France?", stream=True)
web_agent.print_response("which are the top 5 AI LLM models as of today")