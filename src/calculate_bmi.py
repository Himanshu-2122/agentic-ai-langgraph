from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. Define the State
# -----------------------------

class InputData(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    catogry : str


# -----------------------------
# 2. Create BMI Node
# -----------------------------

def bmi_calculator(state: InputData) -> InputData:
    weight = state["weight_kg"]
    height = state["height_m"]

    bmi = weight / (height ** 2)

    state["bmi"] = bmi

    return state

def body_type(state: InputData)-> InputData:
    bmi_score = state['bmi']
    if bmi_score < 18.5 :
        state['catogry'] = "underweight"

    elif bmi_score > 18.5 and bmi_score < 24.9 :
        state['catogry'] = "normal"

    elif bmi_score > 24.9 and bmi_score < 30 :
        state['catogry'] = "overweight"

    else:
        state['catogry'] = "obeese"

    return state


# -----------------------------
# 3. Create StateGraph
# -----------------------------

graph = StateGraph(InputData)


# -----------------------------
# 4. Add Node
# -----------------------------

graph.add_node("bmi", bmi_calculator)
graph.add_node("body", body_type)


# -----------------------------
# 5. Add Edges
# -----------------------------

graph.add_edge(START, "bmi")
graph.add_edge('bmi','body')
graph.add_edge("body", END)


# -----------------------------
# 6. Compile Graph
# -----------------------------

workflow = graph.compile()


# -----------------------------
# 7. Execute Graph
# -----------------------------

result = workflow.invoke({
    "weight_kg": 80,
    "height_m": 1.85
})


# -----------------------------
# 8. Print Result
# -----------------------------

print("BMI Result:")
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