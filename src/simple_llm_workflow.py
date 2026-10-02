from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_groq  import ChatGroq

from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 2.0
)

# create class or state

class LLMState(TypedDict):
    question : str
    answer : str 


# creating LLM_QA node 

def LLM_QA (state : LLMState) -> LLMState:
    # extarct the question

    question = state ["question"]

    # form promt 

    prompt = f"Answer the question {question}"

    # invoke the llm and pass the state ans

    state['answer'] = model.invoke(prompt).content

    # return the state 

    return state

# create graph
graph = StateGraph(LLMState)

# create node of graph
graph.add_node("LLM" , LLM_QA)

# connecting node using edge

graph.add_edge(START , "LLM")
graph.add_edge("LLM" , END)

# compile graph

workflow = graph.compile()

# excuting 

result = workflow.invoke({
    "question": "who is the greatest footballer of all time"
})

print(result)


# -----------------------------
# 9. Generate Graph Image
# -----------------------------

png_data = workflow.get_graph().draw_mermaid_png()


# -----------------------------
# 10. Save Graph Image
# -----------------------------

with open("bmi_graph.png", "wb") as f:
    f.write(png_data)


print("\nGraph saved as: bmi_graph.png")
