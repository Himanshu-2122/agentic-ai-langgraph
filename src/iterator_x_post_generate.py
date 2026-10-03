from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import operator
from dotenv import load_dotenv
from typing import TypedDict , Literal
from pydantic import BaseModel , Field

load_dotenv()




generator_llm = ChatGoogleGenerativeAI(model='gpt-4o-mini')
evaluator_llm = ChatGoogleGenerativeAI(model='gpt-4o-mini')
optimizer_llm = ChatGoogleGenerativeAI(model='gpt-4o-mini')


class TweetState(TypedDict):
    topic : str
    tweet : str
    evaluation : Literal["approved" , "needs_improvement"]
    feedback : str
    iteration : int
    max_iteration : int



def generate_tweet(state: TweetState):

    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    # send generator_llm
    response = model.invoke(messages).content

    # return response
    return {'tweet': response, 'tweet_history': [response]}




