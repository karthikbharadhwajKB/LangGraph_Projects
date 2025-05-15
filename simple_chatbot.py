from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# loading environment variables from .env file
load_dotenv()

# Initialize the chat model
llm = ChatGoogleGenerativeAI(
     model="gemini-2.0-flash"
)

class State(TypedDict):
    """
    State of the chatbot.
    """
    messages: Annotated[list, add_messages]


# Define the state graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# building the graph
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)

# compile the graph
graph = graph_builder.compile()

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting chat. Goodbye!")
        break

    state = graph.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })

    bot_response = state["messages"][-1].content
    print(f"Bot: {bot_response}")



