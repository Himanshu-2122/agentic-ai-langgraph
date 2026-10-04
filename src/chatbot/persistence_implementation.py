from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import  TypedDict 
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

# define the model 

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1.0
)
# define the state 

class JokeState (TypedDict) :
    topic : str
    joke : str
    explanation : str

# creating the function for the node 

# generating the jokes 

def generate_joke(state : JokeState)-> dict  :

    # get topic
    topic = state["topic"]

    # defing the prompt 
    prompt = f"please generate the funny jokes on giving topic and use emogies as much as you can \n {topic}"

    # pass to llm 
    joke = model.invoke(prompt).content

    # returning the change
    return {"joke":joke}


# explaining the jokes 

def explain_joke (state : JokeState) -> dict :
    # fetch the the generated jokes 
    generated_joke = state["joke"]


    # define the prompt 

    prompt = f"please explain this joke like a adult , and the joke is {generated_joke}"

    # pass to the llm / model

    joke_explaination = model.invoke(prompt).content

    # returnig the joke in to state

    return {"explanation":joke_explaination}



# define the graph

graph = StateGraph(JokeState)

# creating the nodes

graph.add_node("gen_joke" , generate_joke)
graph.add_node("explain_joke" , explain_joke)

# connecting nodes using edges 

graph.add_edge(START , "gen_joke")
graph.add_edge("gen_joke" , "explain_joke")
graph.add_edge("explain_joke" , END)

# creating memory checkpoint
checkpointer = InMemorySaver()
# compile the work flow 
workflow = graph.compile(checkpointer=checkpointer)

# excute the workflow

thread_id = 1

config = {"configurable":{'thread_id':thread_id}}

result = workflow.invoke({"topic":"PIZZA"},config = config)

# fetching final value of the workflow 

print(workflow.get_state(config))

# checkpoint history

history = list(workflow.get_state_history(config))

print(history)






for i, snapshot in enumerate(history, start=1):
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]

    print(f"Checkpoint {i}: {checkpoint_id}")


# getting particuler checkpoint using checkpoint id 
# print(workflow.get_state({"configurable":{"thread_id":"1", "checkpoint_id":"1f1beffd-7e08-61ae-8000-9a7c57857919"}}))

checkpoint_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1f1befff-60c4-6fd6-8001-d741a14403df"
    }
}

checkpoint_state = workflow.get_state(checkpoint_config)

print(checkpoint_state)

# print(result)

# -----------------------------
# 9. Generate Graph Image
# -----------------------------

png_data = workflow.get_graph().draw_mermaid_png()


# -----------------------------
# 10. Save Graph Image
# -----------------------------

with open("persistance_implementaion_1.png", "wb") as f:
    f.write(png_data)


print("\nGraph saved as: persistance_implementaion_1.png")