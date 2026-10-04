from langgraph.graph import StateGraph , START , END 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Literal , Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# creating model 

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1.0
)
# defining state 

class ChatState (TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages ]


# definig the funtion for the nodes 
def chat_bot (state : ChatState)-> dict:
    # take user msg 
    user_input = state["messages"]
    # give it to llm
    response = model.invoke(user_input)
    # returning the chnage
    return {"messages": response}


# creating the memory 
checkpointer = MemorySaver()

# defining graph
graph = StateGraph(ChatState)

# creating nodes 
graph.add_node("Chat" , chat_bot)

# connecting edges to nodes 

graph.add_edge(START , "Chat")
graph.add_edge("Chat" , END)

# compiling the graph 
# workflow =graph.compile()

workflow =  graph.compile(checkpointer=checkpointer)

# result = workflow.invoke({"messages": "hello"})

# print(result)


# looping the invoke
thread_id = 1
while True : 
    user_messages = input ("type your input...")
    print("user input : ",user_messages)
    if user_messages.strip().lower() in ["exit" , "quit" , "bye"]:
        break
    config = {"configurable":{'thread_id':thread_id}}
    result = workflow.invoke({"messages":HumanMessage(content=user_messages)} , config=config)
    print("AI : " , result['messages'][-1].content)



# --------------------------------
# 17. Generate Graph Image
# --------------------------------

png_data = workflow.get_graph().draw_mermaid_png()


# --------------------------------
# 18. Save Graph Image
# --------------------------------

with open("simple_chatbot_workflow_1.png", "wb") as f:
    f.write(png_data)


print("\nGraph saved as: simple_chatbot_workflow_1.png")


