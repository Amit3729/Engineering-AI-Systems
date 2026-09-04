import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Engineering AI Assistant", page_icon="AI", layout="wide")
st.title("Engineering AI Assistant")
st.caption("RAG-backed answers with structured output and controlled tools")

with st.sidebar:
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1)
    top_p = st.slider("Top p", 0.1, 1.0, 0.9, 0.05)
    if st.button("Re-index documents"):
        result = requests.post(f"{API_URL}/ingest", timeout=30)
        st.success(result.json())

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

if question := st.chat_input("Ask about the system or try: calculate 12 * 4"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and composing..."):
            response = requests.post(f"{API_URL}/chat", json={"question": question, "temperature": temperature, "top_p": top_p}, timeout=60)
        if response.ok:
            payload = response.json()
            st.markdown(payload["answer"])
            meta = f"provider: {payload['provider']} | sources: {len(payload['sources'])}" + (" | cached" if payload["cached"] else "")
            st.caption(meta)
            if payload["sources"]:
                with st.expander("Sources"):
                    for source in payload["sources"]:
                        st.write(f"**{source['document']}** ({source['score']:.2f})")
                        st.write(source["text"])
            st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "meta": meta})
        else:
            st.error(response.text)
