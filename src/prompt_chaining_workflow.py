"""


"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_groq  import ChatGroq

from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 2.0
)

class BlogState(TypedDict):
    title : str
    outline : str
    content : str

def create_outline(state : BlogState)-> dict :
    title = state['title']
    prompt = f"Generate a detailed outline about {title}";
    outline = model.invoke(prompt).content
    return {"outline":outline}

def create_blog (state : BlogState)-> dict :
    title = state["title"]
    outline = state["outline"]
    prompt = f"Create a detailed report on the topic {title} with following the outline {outline}"
    content= model.invoke(prompt).content
    return {"content" : content}
# define the graph
graph = StateGraph(BlogState)

# creating nodes of graph

graph.add_node("outline",create_outline)
graph.add_node("blog" , create_blog)

# connecting nodes using edges 

graph.add_edge(START , "outline")
graph.add_edge("outline" , "blog")
graph.add_edge("blog" , END)

# compile the graph

workflow = graph.compile()

# intial state 

intial_state = {"title":"RISE OF AN AI IN INDIA AS A COMSUMER ASPECT"}

# excuting 

result = workflow.invoke(intial_state)
print(result)
print("----------------------------------------------------------------------------------------------")
print(type(result))

print("Title : ",result["title"])
print("outline : ",result["outline"])
print("content : ",result["content"])



