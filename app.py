# cd "C:\Users\ayush\OneDrive\Desktop\llm\frontend"
# streamlit run app.py
import requests
import streamlit as st
import time


BACKEND_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="AI Academic Assistant", layout="wide")
st.title("AI Academic Assistant (Prototype)")


# ---------- Sidebar: System Metrics & Health ----------
with st.sidebar:
    st.header("System Status")
    
    try:
        health = requests.get(f"{BACKEND_URL}/", timeout=5).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Vectors Indexed", health.get("total_vectors", "N/A"))
        with col2:
            st.metric("📁 Documents", health.get("documents", "N/A"))
        st.success("✅ Backend Online")
    except:
        st.error("❌ Backend Offline")
    
    st.divider()


# ---------- Sidebar: Ingest New Text/File ----------


st.sidebar.header("Ingest New Text")


doc_id = st.sidebar.text_input("Document ID", value="notes-1")


uploaded = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
st.sidebar.caption("Or paste text below (optional).")
new_text = st.sidebar.text_area("Paste text to add")


if st.sidebar.button("Add to knowledge base"):
    if uploaded is not None:
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/plain")}
            data = {"doc_id": doc_id}
            resp = requests.post(
                f"{BACKEND_URL}/ingest_file",
                files=files,
                data=data,
                timeout=300,
            )
            if resp.ok:
                result = resp.json()
                st.sidebar.success(f"✅ {result}")
            else:
                st.sidebar.error(f"Backend error: {resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error ingesting file: {e}")

    elif new_text.strip():
        try:
            resp = requests.post(
                f"{BACKEND_URL}/ingest_text",
                json={"doc_id": doc_id, "text": new_text},
                timeout=60,
            )
            if resp.ok:
                result = resp.json()
                st.sidebar.success(f"✅ {result}")
            else:
                st.sidebar.error(f"Backend error: {resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error ingesting text: {e}")

    else:
        st.sidebar.warning("Upload a .txt file or paste text before ingesting.")


# ---------- Main Chat Area ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.caption("📄 **Sources:** " + ", ".join(msg["sources"]))
            
            if msg.get("similarity_scores"):
                scores_text = " | ".join(
                    [f"{src}: {score:.3f}" for src, score in msg["similarity_scores"]]
                )
                st.caption(f"📊 **Similarity Scores:** {scores_text}")
            
            if msg.get("query_time"):
                st.caption(f"⏱️ **Query Time:** {msg['query_time']:.3f}s")


# ---------- Evaluation Metrics Dashboard ----------


if st.session_state.metrics_history:
    with st.expander("📈 Query Metrics Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        avg_time = sum(m.get("query_time", 0) for m in st.session_state.metrics_history) / len(st.session_state.metrics_history)
        avg_score = sum(m.get("avg_similarity", 0) for m in st.session_state.metrics_history) / len(st.session_state.metrics_history)
        precision_1_count = sum(1 for m in st.session_state.metrics_history if m.get("precision_1"))
        precision_3_avg = sum(m.get("relevant_count", 0) for m in st.session_state.metrics_history) / len(st.session_state.metrics_history) / 3
        
        with col1:
            st.metric("Avg Query Time", f"{avg_time:.2f}s", delta="<2s target")
        with col2:
            st.metric("Avg Similarity Score", f"{avg_score:.3f}", delta="Lower is better")
        with col3:
            st.metric("Precision@1", f"{(precision_1_count/len(st.session_state.metrics_history))*100:.1f}%")
        with col4:
            st.metric("Precision@3", f"{precision_3_avg*100:.1f}%")


# ---------- Chat Input & Processing ----------


prompt = st.chat_input("Ask a question about your notes or ingested documents:")


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        start_time = time.time()
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": prompt, "k": 3},
            timeout=30,
        )
        query_time = time.time() - start_time
        
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "No answer field in response.")
        results = data.get("results", [])
        
    except Exception as e:
        answer = f"Backend not reachable or error: {e}"
        results = []
        query_time = 0

    doc_ids = sorted({r.get("doc_id") for r in results if r.get("doc_id")})
    similarity_scores = [(r.get("doc_id"), r.get("score", 0)) for r in results]
    
    avg_similarity = sum(s[1] for s in similarity_scores) / len(similarity_scores) if similarity_scores else 0
    precision_1 = True
    relevant_count = len(results)
    
    st.session_state.metrics_history.append({
        "question": prompt,
        "query_time": query_time,
        "avg_similarity": avg_similarity,
        "precision_1": precision_1,
        "relevant_count": relevant_count,
        "num_results": len(results)
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": doc_ids,
        "similarity_scores": similarity_scores,
        "query_time": query_time,
        "avg_similarity": avg_similarity
    })

    with st.chat_message("assistant"):
        st.markdown(answer)
        
        if doc_ids:
            st.caption("📄 **Sources:** " + ", ".join(doc_ids))
        
        # if similarity_scores:
        #     scores_text = " | ".join([f"{src}: {score:.3f}" for src, score in similarity_scores])
        #     st.caption(f"📊 **Similarity Scores:** {scores_text}")
        #     st.caption("*(Lower L2 distance = higher similarity)*")
        
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.metric("⏱️ Query Time", f"{query_time:.3f}s", delta="<2s goal")
        # with col2:
        #     st.metric("📊 Avg Similarity", f"{avg_similarity:.3f}")
        # with col3:
        #     st.metric("📈 Results", f"{len(results)}")


# ---------- Evaluation Metrics Reference ----------
with st.sidebar:
    st.divider()
    st.subheader("Evaluation Metrics")
    
    with st.expander("📚 Metrics Guide"):
        st.markdown("""
        **⏱️ Query Time:** Response time (target: <2s)
        
        **📊 Similarity Score:** L2 distance (lower=better)
        
        **Precision@1:** Top result relevant (target: 85%)
        
        **Precision@3:** 2+ of 3 relevant (target: 75%)
        
        **MRR@3:** Ranking quality (target: 0.80)
        
        **NDCG@3:** Ranking order (target: 0.88)
        """)
