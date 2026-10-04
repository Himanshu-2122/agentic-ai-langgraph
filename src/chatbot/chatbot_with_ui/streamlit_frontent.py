import streamlit as st
from langgraph_backend import workflow
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage
# user chat_message 
# with st.chat_message("user"):
#     st.text("hi")

# asisstent chat_message
# with st.chat_message("assistant"):
#     st.text('how can i assist you ..')


# session_state
if 'message_history' not in st.session_state:
    st.session_state["message_history"] = []


# print / displaying the past converation of the user and assistent
for message in st.session_state["message_history"]:
    with st.chat_message(message['role']):
        st.text(message["content"])
# user imput 

user_input = st.chat_input("type here..")

if user_input :

    thread_id = 1

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    config = {"configurable":{"thread_id":thread_id}}
    ai_message = st.write_stream( message_chunk.content for  message_chunk , metadata in workflow.stream(
        {"messages":HumanMessage(content=user_input)},
        config=config,
        stream_mode="messages",
    ))
        
    # response = workflow.invoke({"messages":HumanMessage(content=user_input)},config=config)
    # ai_message = response['messages'][-1].content
    st.session_state['message_history'].append({"role":"assistent" , "content":ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)