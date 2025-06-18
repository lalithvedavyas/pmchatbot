import streamlit as st
from chatbot import get_qa_chain, load_vectorstore
import os

st.set_page_config(page_title="PM Methodology Chatbot", layout="centered")
st.title("🤖 Project Management Methodology Chatbot")

if not os.path.exists("vectorstore"):
    with st.spinner("Creating knowledge base..."):
        load_vectorstore()

chain = get_qa_chain()

query = st.text_input("Ask me about project management methodologies:")
if query:
    with st.spinner("Thinking..."):
        result = chain.run(query)
        st.success(result)
