import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Engineering AI Assistant", page_icon="AI", layout="wide")
st.title("Engineering AI Assistant")
st.caption("RAG over ChromaDB with ONNX embeddings, model-driven tool calling, and structured JSON output")

with st.sidebar:
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1)
    top_p = st.slider("Top p", 0.1, 1.0, 0.9, 0.05)

    st.subheader("Index")
    if st.button("Re-index documents"):
        with st.spinner("Embedding corpus..."):
            result = requests.post(f"{API_URL}/ingest", params={"force": True}, timeout=300)
        st.success(result.json() if result.ok else result.text)

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException as error:
        st.error(f"API unreachable: {error}")
    else:
        st.caption(f"provider: `{health.get('provider', 'unknown')}`")
        if "index" in health:
            st.json(health["index"], expanded=False)
        else:
            # Version skew should not take the sidebar down.
            st.warning("API is running an older build. Restart it to pick up the current code.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

if question := st.chat_input("Ask about the system, or try: what is 128 * 47?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and composing..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "temperature": temperature, "top_p": top_p},
                    timeout=120,
                )
            except requests.RequestException as error:
                st.error(f"Could not reach the API: {error}")
                st.stop()

        if not response.ok:
            st.error(f"{response.status_code}: {response.text}")
            st.stop()

        payload = response.json()
        st.markdown(payload["answer"])

        meta = (
            f"{payload['provider']} / {payload['model']} | "
            f"confidence {payload['confidence']:.2f} | "
            f"{payload['latency_ms']} ms | "
            f"{len(payload['sources'])} sources"
            + (" | cached" if payload["cached"] else "")
        )
        st.caption(meta)

        if payload["tool_calls"]:
            with st.expander(f"Tool calls ({len(payload['tool_calls'])})"):
                for call in payload["tool_calls"]:
                    st.markdown(f"**{call['name']}**")
                    st.json({"arguments": call["arguments"], "result": call["result"]}, expanded=False)

        if payload["sources"]:
            with st.expander(f"Sources ({len(payload['sources'])})"):
                for source in payload["sources"]:
                    # A CPython page is worth naming by its section, not its chunk number.
                    where = source["document"] + (f"#{source['anchor']}" if source.get("anchor") else "")
                    heading = source.get("section") or source.get("title") or ""
                    st.markdown(f"**{where}** · chunk {source['chunk_id']} · similarity {source['score']:.3f}")
                    if heading:
                        st.caption(heading)
                    st.write(source["text"])

        st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "meta": meta})
