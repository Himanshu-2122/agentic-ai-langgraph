# from langchain_groq import ChatGroq
# from langgraph.graph import START, END, StateGraph
# from dotenv import load_dotenv
# from typing import TypedDict, Literal
# from pydantic import BaseModel, Field


# load_dotenv()


# # --------------------------------
# # Model
# # --------------------------------

# model = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=1.0
# )


# # --------------------------------
# # Structured Output Schema
# # --------------------------------

# class ExtractSentiment(BaseModel):

#     sentiment: str = Field(
#         description="Return either positive or negative"
#     )


# structured_model = model.with_structured_output(
#     ExtractSentiment,
#     method="json_schema"
# )


# # --------------------------------
# # State
# # --------------------------------

# class SentimentState(TypedDict):

#     review: str
#     positive: str
#     negative: str
#     result: str


# # --------------------------------
# # Sentiment Router
# # --------------------------------

# def find_sentiment(
#     state: SentimentState
# ) -> Literal["positive", "negative"]:

#     prompt = f"""
# Extract the sentiment of the following review.

# Return only:
# - positive
# - negative

# Review:
# {state["review"]}
# """

#     output = structured_model.invoke(prompt)

#     if output.sentiment.lower() == "positive":
#         return "positive"

#     else:
#         return "negative"


# # --------------------------------
# # Positive Node
# # --------------------------------

# def positive_node(state: SentimentState) -> dict:

#     return {
#         "positive": "The review is positive",
#         "result": "Positive sentiment detected"
#     }


# # --------------------------------
# # Negative Node
# # --------------------------------

# def negative_node(state: SentimentState) -> dict:

#     return {
#         "negative": "The review is negative",
#         "result": "Negative sentiment detected"
#     }


# # --------------------------------
# # Create Graph
# # --------------------------------

# graph = StateGraph(SentimentState)


# # --------------------------------
# # Add Nodes
# # --------------------------------

# graph.add_node("positive", positive_node)

# graph.add_node("negative", negative_node)


# # --------------------------------
# # Conditional Routing
# # --------------------------------

# graph.add_conditional_edges(
#     START,
#     find_sentiment
# )


# # --------------------------------
# # End Edges
# # --------------------------------

# graph.add_edge("positive", END)

# graph.add_edge("negative", END)


# # --------------------------------
# # Compile
# # --------------------------------

# workflow = graph.compile()


# # --------------------------------
# # Input
# # --------------------------------

# initial_state = {
#     "review": "This product is amazing. I really loved it!",
#     "positive": "",
#     "negative": "",
#     "result": ""
# }


# # --------------------------------
# # Run
# # --------------------------------

# result = workflow.invoke(initial_state)

# print(result)

# import librabry and packabes 
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START , END
from dotenv import load_dotenv
from typing import TypedDict , Literal
from pydantic import Field , BaseModel


# loading .env
load_dotenv()

# defining the model 
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1.0
)
# define the schema
class SentimentSchema(BaseModel):
    sentiment : Literal["positive" , "negative"] = Field(description="give the sentiment of the review")

structured_model_1 = model.with_structured_output(
    SentimentSchema,
    method = "json_schema"
)

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


structured_model_2 = model.with_structured_output(
    DiagnosisSchema,
    method = "json_schema"
)




# defining the state
class ReviewState (TypedDict):
    review :str
    sentiment : Literal["positive" , "negative"]
    daignosis : dict 
    response : str

# defining the node function
def find_sentiment(state : ReviewState) -> dict:
    prompt = f" please the find the sentiment of \n {state["review"]} "
    sentiment = structured_model_1.invoke(prompt).sentiment
    return {"sentiment":sentiment}

def check_sentiment(state : ReviewState ) -> Literal["positive" , "negative"]:
    if state["sentiment"] == "positive":
        return "positive"
    else :
        return "negative"

def run_daignosis (state : ReviewState):
    prompt = f"""
    Diagnose this negative review:

    {state['review']}

    Return issue_type, tone, and urgency.
    """
    response = structured_model_2.invoke(prompt)
    return {"daignosis":response.model_dump()}


def positive_response(state : ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""

    response = model.invoke(prompt).content
    return {"response":response}

def negative_response(state : ReviewState):
    diagnosis = state['daignosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content
    return {"response":response}


# define the graph 

graph = StateGraph(ReviewState)

#creating the nodes

graph.add_node("find_sentiment" , find_sentiment)
graph.add_node("run_diagnosis" , run_daignosis)
graph.add_node("positive_response" , positive_response)
graph.add_node("negative_response" , negative_response)

# connecting the nodes using edge 

graph.add_edge(START , "find_sentiment")
graph.add_conditional_edges(
    "find_sentiment",
    check_sentiment,
    {
        "positive": "positive_response",
        "negative": "run_diagnosis"
    }
)
graph.add_edge("positive_response" , END)
graph.add_edge("run_diagnosis" , "negative_response")
graph.add_edge("negative_response" , END)



# compile the graph

workflow = graph.compile()



# excuting the graph

intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
result = workflow.invoke(intial_state)

print(result)


# -----------------------------
# 9. Generate Graph Image
# -----------------------------

png_data = workflow.get_graph().draw_mermaid_png()


# -----------------------------
# 10. Save Graph Image
# -----------------------------

with open("review_reply_workflow_full.png", "wb") as f:
    f.write(png_data)


print("\nGraph saved as: review_reply_workflow_full.png")






























