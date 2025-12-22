import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# 1. Setup your API Key
load_dotenv('c:/codellm/.env')  #load_dotenv()
GROQ_API_KEY=os.environ['GROQ_API_KEY']

# 2. Define the State
# We use 'add_messages' to tell LangGraph to append new messages to the history rather than overwriting the whole list.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Define the LLM Node
# This function takes the current state (conversation history) and returns a new AI message.
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant") #"llama3-70b-8192"

def chatbot(state: State):
    # The state contains a list of messages (User, AI, System)
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a dictionary with the key 'messages' to update the state
    return {"messages": [response]}

# 4. Build the Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

graphBytes = graph.get_graph().draw_mermaid_png()
with open("simpleChatBot.png", "wb") as f:
    f.write(graphBytes)

# 5. Run the Application
print("--- Starting Chatbot (type 'quit' to exit) ---")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    # Run the graph with the user's input
    # We pass the input as a "UserMessage" implicitly by passing a dictionary
    initial_state = {"messages": [("user", user_input)]}
    
    # stream_mode="values" prints the output as it is generated
    for event in graph.stream(initial_state):
        for value in event.values():
            # The last message in the list is the AI's response
            if hasattr(value["messages"][-1], "content"):
                 print(f"AI: {value['messages'][-1].content}")