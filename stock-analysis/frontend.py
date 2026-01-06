import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import base64
import pandas as pd
import time

# ==========================================
# 1. 配置
# ==========================================
BACKEND_STREAM_URL = "http://localhost:8000/analyze_stream"

st.set_page_config(page_title="AgentCube Analyst", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .worker-box {
        background-color: #1e293b; color: #e2e8f0; padding: 12px;
        margin-bottom: 8px; border-radius: 6px; border-left: 5px solid #64748b;
    }
    .worker-box.running { border-left-color: #f59e0b; }
    .worker-box.done { border-left-color: #10b981; }
    .terminal-log {
        background: #0f172a; color: #22c55e; font-family: monospace;
        font-size: 11px; padding: 8px; border-radius: 4px;
        max-height: 150px; overflow-y: auto; white-space: pre-wrap;
    }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; color: #333;
    }
</style>
""", unsafe_allow_html=True)

def render_mermaid(code: str, height=350):
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">{code}</div>
    <script type="module">
        import mermaid from '[https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs](https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs)';
        mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
    """
    components.html(html_code, height=height)

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.title("🎛️ Agent 控制台")
    render_mermaid("""
    graph TD
        User[Input] --> Planner
        Planner -->|Code| SB1[Sandbox]
        SB1 -->|List| Map((Map))
        Map --> W1[Worker]
        Map --> W2[Worker]
        W1 -->|Tools| SB2[Sandbox]
        SB2 --> Agg[Aggregator]
        Agg --> Result
        style Agg fill:#10b981,stroke:#fff
    """)
    st.info("Backend: FastAPI + LangGraph\nFrontend: Streamlit")

# ==========================================
# 3. 主逻辑
# ==========================================
st.title("⚡️ 并发量化分析 Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("指令: 分析 A 股最热门的 3 只银行股"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        monitor_expander = st.expander("👁️ 全链路监控", expanded=True)
        result_container = st.container()
        
        worker_state = {} 
        analysis_finished = False 
        final_advice = None
        final_details = []

        try:
            with monitor_expander:
                dashboard = st.empty()
                response = requests.post(BACKEND_STREAM_URL, json={"query": prompt}, stream=True, timeout=300)
                
                if response.status_code != 200:
                    st.error(f"Server Error: {response.status_code}")
                    st.stop()

                for line in response.iter_lines():
                    if not line: continue
                    try:
                        event = json.loads(line)
                    except: continue

                    evt_type = event.get("type")
                    ticker = event.get("ticker", "System") or "System"

                    if ticker not in worker_state:
                        worker_state[ticker] = {"status": "Waiting...", "logs": [], "class": ""}

                    # Update State
                    if evt_type == "node_status":
                        worker_state["System"]["status"] = f"Node: {event.get('node')}"
                    elif evt_type == "llm_stream":
                        worker_state[ticker]["status"] = "Thinking..."
                        worker_state[ticker]["class"] = "running"
                    elif evt_type == "code_input":
                        worker_state[ticker]["logs"].append(f"> CODE:\n{event.get('code','')[:100]}...")
                        worker_state[ticker]["status"] = "Running Code..."
                    elif evt_type == "tool_output":
                        clean_out = event.get("output","").split("###IMAGE")[0]
                        worker_state[ticker]["logs"].append(f"< OUT:\n{clean_out[:150]}...")
                        worker_state[ticker]["status"] = "Done"
                        worker_state[ticker]["class"] = "done"
                    elif evt_type == "final_result":
                        final_advice = event.get("advice")
                        final_details = event.get("details")
                        analysis_finished = True

                    # Render Dashboard
                    with dashboard.container():
                        if "System" in worker_state:
                            st.info(worker_state["System"]["status"])
                        
                        active = [k for k in worker_state if k != "System"]
                        if active:
                            cols = st.columns(min(len(active), 3))
                            for idx, t in enumerate(active):
                                info = worker_state[t]
                                with cols[idx % 3]:
                                    st.markdown(f"""
                                    <div class="worker-box {info['class']}">
                                        <b>{t}</b> <small>{info['status']}</small>
                                        <div class="terminal-log">{info['logs'][-1] if info['logs'] else ''}</div>
                                    </div>""", unsafe_allow_html=True)

            if not analysis_finished:
                st.error("❌ 连接中断，未收到最终结果。")
            else:
                with result_container:
                    st.success("✅ 分析完成")
                    st.markdown(final_advice)
                    
                    if final_details:
                        st.divider()
                        st.subheader("📊 详情看板")
                        
                        # Chart
                        chart_data = [{"Ticker": d["ticker"], "Score": d["score_data"].get("score", 0)} for d in final_details if d.get("score_data")]
                        if chart_data:
                            st.bar_chart(pd.DataFrame(chart_data).set_index("Ticker"), color="#10b981")

                        # Cards
                        cols = st.columns(len(final_details))
                        for idx, d in enumerate(final_details):
                            with cols[idx]:
                                sc = d.get("score_data", {})
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{d['ticker']}</h4>
                                    <h2 style="color:#10b981">{sc.get('score', 0)}</h2>
                                    <small>Mom: {sc.get('momentum',0):.2f} | Vol: {sc.get('volatility',0):.2f}</small>
                                </div>
                                """, unsafe_allow_html=True)
                                if d.get("image_base64"):
                                    st.image(base64.b64decode(d["image_base64"]), use_column_width=True)

                    st.session_state.messages.append({"role": "assistant", "content": final_advice})

        except Exception as e:
            st.error(f"Client Error: {e}")