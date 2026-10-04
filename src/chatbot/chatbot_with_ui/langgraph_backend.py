from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages , BaseMessage 
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()
# defining the model 

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
)


# ollama model implementation
model_ollama = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b-v2",
    temperature=0.7,
    reasoning=False,   # qwen3 ka <think> output band karne ke liye
)

# defining the state 

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

# creating the fucntion for the nodes

def chat_bot(state : ChatState)-> dict :
    # fetching the messages
    message = state["messages"]

    # prompt for the model
    prompt = f"""
You are a helpful and friendly AI assistant.

Answer the user's message below:

User message:
{message}

Your responsibilities:
- Answer the user's question clearly and accurately.
- Keep your response simple and easy to understand.
- Be concise unless the user asks for a detailed explanation.
- If you are unsure about something, say so instead of making up information.
- Follow the user's instructions and context.
- Help with programming, general knowledge, writing, learning, and everyday tasks.
- Use examples when they make the explanation easier to understand.
- Maintain a professional and respectful tone.
"""

    # passing prompt among with user input to the model and extratcing the content part becouse model response have multiple other fields

    output = model.invoke(message).content

    # passin the output back to state

    return{"messages" : output}




# defining the graph 
graph = StateGraph(ChatState)

# memory checkpoint 

checkpoint = MemorySaver()

# defining the thread
thread_id = 1

# creating the nodes 

graph.add_node("chat_bot",chat_bot)

# connecting the nodes using edges

graph.add_edge(START , "chat_bot")
graph.add_edge("chat_bot" , END)

# compile the graph
workflow = graph.compile(checkpointer=checkpoint)

# while True :
#     user_input = input ("Enter your question here ...")
#     print("User Input :",user_input)
#     if user_input.strip().lower() in ["bye" , "exit" , "quit"]:
#         break
#     config = {"configurable" :{"thread_id":thread_id}}

#     result = workflow.invoke({"messages":HumanMessage(content=user_input)}, config=config)
#     print("AI :",result['messages'][-1].content )
