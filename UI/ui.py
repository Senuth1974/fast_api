import streamlit as st
from backend.core import run_llm  # your LLM call function

# Header
st.title("AI Assistant")
st.caption("Ask anything and get smart, context-aware answers.")

# Session State Init
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {"user": ..., "assistant": ...}

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    with st.spinner("Generating response..."):
        # Call your LLM
        response = run_llm(query=user_input, chat_history=st.session_state["chat_history"])
        answer = response.get("result", "I'm sorry, I couldn't find that information.")

    # Append to history
    st.session_state["chat_history"].append({"user": user_input, "assistant": answer})

# Chat Display
for chat in st.session_state["chat_history"]:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["assistant"])
