import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Chatbot")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    BACKEND_URL,
                    json={"message": prompt},
                    timeout=60
                )

                if response.status_code == 200:

                    data = response.json()

                    reply = data.get(
                        "response",
                        "No response received."
                    )

                else:

                    reply = f"Backend Error ({response.status_code})\n\n{response.text}"

            except requests.exceptions.ConnectionError:

                reply = (
                    "❌ Cannot connect to backend.\n\n"
                    "Make sure FastAPI is running:\n\n"
                    "python -m uvicorn app:app --reload"
                )

            except Exception as e:

                reply = f"Error:\n\n{str(e)}"

            st.markdown(reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

# Sidebar
with st.sidebar:

    st.title("Settings")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.write("Backend")

    if st.button("Check Backend"):

        try:

            r = requests.get("http://127.0.0.1:8000")

            if r.status_code == 200:
                st.success("Backend Connected ✅")
            else:
                st.error("Backend Error")

        except:
            st.error("Backend Not Running ❌")