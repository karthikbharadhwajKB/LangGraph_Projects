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

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
     ...,
     description="Classify the message as either emotional or logical based on the message content."
    )

# Define the State for the Agent
class State(TypedDict):
    """
    State of the chatbot.
    """
    messages: Annotated[list, add_messages]
    message_type: str | None
    
    
# Define Nodes for the State Graph    
def classify_message(state: State): 
    last_message = state['messages'][-1] 
    
    # Classify the message as either emotional or logical
    classifier_llm = llm.with_structured_output(
        MessageClassifier
    )
    
    result = classifier_llm.invoke([
        {
            "role": "system", 
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {
            "role": "user", 
            "content": last_message.content
        }
    ])
    
    return {"message_type": result.message_type}
    
    

def router(state: State):
    message_type = state.get("message_type", "logical")
    
    if message_type == "emotional": 
        return {"next": "therapist"}
    
    return {"next": "logical"} 
    
     

def therapist_agent(state: State):
    last_message = state['messages'][-1]
    
    # Therapist agent logic
    messages = [
        {
            "role": "system", 
            "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
        },
        {
            "role": "user", 
            "content": last_message.content
        }
    ] 
    
    response = llm.invoke(messages)
    
    return {"messages": [{'role': 'assistant', 'content': response.content}]}

def logical_agent(state: State):
    last_message = state['messages'][-1]
    
    # Logical agent logic
    messages = [
        {
            "role": "system", 
            "content": """You are a purely logical assistant. Focus only on facts and information.
                        Provide clear, concise answers based on logic and evidence.
                        Do not address emotions or provide emotional support.
                        Be direct and straightforward in your responses."""
        },
        {
            "role": "user", 
            "content": last_message.content
        }
    ] 
    
    response = llm.invoke(messages)
    
    return {"messages": [{'role': 'assistant', 'content': response.content}]}
     

# Buiding the State Graph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

    
# Add edges to the graph
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router", 
     lambda state: state.get("next"), 
     {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# Compile the graph
graph = graph_builder.compile()

def run_chatbot():
    # initialize the state
    state = {"messages": [], 
             "message_type": None}
    
    while True: 
        # Get user input
        user_input = input("User: ")
        
        # Check for exit condition
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break
        
        # Update the state with user input
        state['messages'] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]
        
        # Invoke the graph with the current state
        state = graph.invoke(state) 
        
        
        if state.get("messages") and len(state["messages"]) > 0:
            response = state["messages"][-1]
            message_type = state['message_type'] 
            print(f"{message_type} Assistant: {response.content}")
            
        
        
if __name__ == "__main__":
    run_chatbot()
