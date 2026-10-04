from langgraph.graph import StateGraph , START ,END
from typing import TypedDict 
import time
from langgraph.checkpoint.memory import InMemorySaver

# define the state 
class FaultState(TypedDict):
    input : str
    step1 : str
    step2 : str
    step3 : str

# defining the function
def step1 (state : FaultState)-> dict :
    print("step1 excuted")
    return{"input": state["input"] , "step1": "step1 done"}

def step2 (state : FaultState)-> dict :
    print("step2 higgind due to some crash or time delay")
    time.sleep(30)
    return {step2 : "step2 done"}

def step3 (state : FaultState)-> dict :
    print("step3 excuted")
    return {step3 : "step3 done" }

# defining the graph 

graph = StateGraph(FaultState)

# creating nodes

graph.add_node("step1" , step1)
graph.add_node("step2" , step2)
graph.add_node("step3" , step3)

# connecting nodes using edges 

graph.add_edge(START , "step1")
graph.add_edge("step1" , "step2")
graph.add_edge("step2" , "step3")
graph.add_edge("step3" , END)

# creatine checkpoint
checkpoint = InMemorySaver()

thread_id= 1

# compile the graph

workflow = graph.compile(checkpointer=checkpoint)


# excuting the workflow 

try :
    print("running the graph -- please manually interrupt during step 2")
    config = {"configurable":{"thread_id":thread_id}}
    workflow.invoke({"input":"start"},config = config)
except KeyboardInterrupt:
    print("kernal manually intrruped (crash simulated)")



