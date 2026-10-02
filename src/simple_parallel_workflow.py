from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# LLM
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=2.0
)


# State
class BatsmanState(TypedDict):
    run: int
    ball: int
    four: int
    six: int

    sr: float
    bpb: float
    bp: float
    summary: str


# -------------------------
# Strike Rate
# -------------------------
def sr(state: BatsmanState):

    run = state["run"]
    balls = state["ball"]

    strike_rate = (run / balls) * 100

    return {
        "sr": strike_rate
    }


# -------------------------
# Boundary Per Ball
# -------------------------
def bpb(state: BatsmanState):

    balls = state["ball"]
    fours = state["four"]
    sixes = state["six"]

    total_boundary = fours + sixes

    boundary_per_ball = total_boundary / balls

    return {
        "bpb": boundary_per_ball
    }


# -------------------------
# Boundary Percentage
# -------------------------
def bp(state: BatsmanState)->dict:

    run = state["run"]
    fours = state["four"]
    sixes = state["six"]

    boundary_run = (fours * 4) + (sixes * 6)

    boundary_percentage = (boundary_run / run) * 100

    return {
        "bp": boundary_percentage
    }


# -------------------------
# Summary
# -------------------------
def summary(state: BatsmanState):

    sr_value = state["sr"]
    bpb_value = state["bpb"]
    bp_value = state["bp"]

    summary_text = f"""
Batsman Stats
-------------------------

Strike Rate         : {sr_value}
Boundary Per Ball   : {bpb_value}
Boundary Percentage : {bp_value}%
"""

    return {
        "summary": summary_text
    }


# -------------------------
# Create Graph
# -------------------------

graph = StateGraph(BatsmanState)


# Add nodes
graph.add_node("sr", sr)
graph.add_node("bpb", bpb)
graph.add_node("bp", bp)
graph.add_node("summary", summary)


# -------------------------
# Parallel Execution
# -------------------------

graph.add_edge(START, "sr")
graph.add_edge(START, "bpb")
graph.add_edge(START, "bp")


# -------------------------
# Join at Summary
# -------------------------

graph.add_edge("sr", "summary")
graph.add_edge("bpb", "summary")
graph.add_edge("bp", "summary")


# End
graph.add_edge("summary", END)


# Compile
workflow = graph.compile()


# -------------------------
# Initial State
# -------------------------

initial_state = {
    "run": 100,
    "ball": 50,
    "four": 6,
    "six": 4
}


# -------------------------
# Execute
# -------------------------

result = workflow.invoke(initial_state)


# -------------------------
# Output
# -------------------------

print(result)

print("\n" + "=" * 50)

print(result["summary"])



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