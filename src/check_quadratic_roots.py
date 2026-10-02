from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


# --------------------------------
# State
# --------------------------------

class QuadraticState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminant: float
    result: str


# --------------------------------
# Node 1: Show Equation
# --------------------------------

def show_equation(state: QuadraticState) -> dict:

    equation = (
        f'{state["a"]}x² + '
        f'{state["b"]}x + '
        f'{state["c"]} = 0'
    )

    return {
        "equation": equation
    }


# --------------------------------
# Node 2: Calculate Discriminant
# --------------------------------

def calculate_discriminant(state: QuadraticState) -> dict:

    a = state["a"]
    b = state["b"]
    c = state["c"]

    discriminant = (b ** 2) - (4 * a * c)

    return {
        "discriminant": discriminant
    }


# --------------------------------
# Node 3: Real Roots
# --------------------------------

def real_root(state: QuadraticState) -> dict:

    root1 = (
        -state["b"] + state["discriminant"] ** 0.5
    ) / (2 * state["a"])

    root2 = (
        -state["b"] - state["discriminant"] ** 0.5
    ) / (2 * state["a"])

    result = f"The roots are {root1} and {root2}"

    return {
        "result": result
    }


# --------------------------------
# Node 4: Repeated Root
# --------------------------------

def repeated_root(state: QuadraticState) -> dict:

    root = (
        -state["b"]
    ) / (2 * state["a"])

    result = f"Only repeated root is {root}"

    return {
        "result": result
    }


# --------------------------------
# Node 5: Non-real Root
# --------------------------------

def non_real_root(state: QuadraticState) -> dict:

    result = "No real roots"

    return {
        "result": result
    }


# --------------------------------
# Conditional Function
# --------------------------------

def check_condition(
    state: QuadraticState
) -> Literal[
    "real_root",
    "repeated_root",
    "non_real_root"
]:

    if state["discriminant"] > 0:
        return "real_root"

    elif state["discriminant"] == 0:
        return "repeated_root"

    else:
        return "non_real_root"


# --------------------------------
# Create Graph
# --------------------------------

graph = StateGraph(QuadraticState)


# --------------------------------
# Add Nodes
# --------------------------------

graph.add_node(
    "show_equation",
    show_equation
)

graph.add_node(
    "discriminant",
    calculate_discriminant
)

graph.add_node(
    "real_root",
    real_root
)

graph.add_node(
    "repeated_root",
    repeated_root
)

graph.add_node(
    "non_real_root",
    non_real_root
)


# --------------------------------
# Add Edges
# --------------------------------

graph.add_edge(
    START,
    "show_equation"
)

graph.add_edge(
    "show_equation",
    "discriminant"
)


# Conditional edges
graph.add_conditional_edges(
    "discriminant",
    check_condition
)


# Each possible path → END

graph.add_edge(
    "real_root",
    END
)

graph.add_edge(
    "repeated_root",
    END
)

graph.add_edge(
    "non_real_root",
    END
)


# --------------------------------
# Compile
# --------------------------------

workflow = graph.compile()


# --------------------------------
# Input
# --------------------------------

initial_state = {
    "a": 2,
    "b": 4,
    "c": 2
}


# --------------------------------
# Execute
# --------------------------------

result = workflow.invoke(initial_state)


# --------------------------------
# Print Result
# --------------------------------

print("Equation:", result["equation"])

print("Discriminant:", result["discriminant"])

print("Result:", result["result"])


# --------------------------------
# Generate Graph Image
# --------------------------------

png_data = workflow.get_graph().draw_mermaid_png()


# --------------------------------
# Save Graph Image
# --------------------------------

with open("quadratic_graph.png", "wb") as f:
    f.write(png_data)


print("\nGraph saved as: quadratic_graph.png")