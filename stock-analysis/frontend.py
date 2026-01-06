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

# CSS 优化：增强 Worker Box 的展示效果
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* Worker 状态卡片 */
    .worker-box {
        background-color: #1e293b; 
        color: #e2e8f0; 
        padding: 15px;
        margin-bottom: 10px; 
        border-radius: 8px; 
        border-left: 5px solid #64748b;
        min-height: 200px; /* 增加最小高度，防止卡片太扁 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .worker-box.running { border-left-color: #f59e0b; }
    .worker-box.done { border-left-color: #10b981; }
    
    /* 头部信息 */
    .worker-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 1.1em;
    }
    
    /* 终端日志区域 */
    .terminal-log {
        background: #0f172a; 
        color: #22c55e; 
        font-family: 'Courier New', monospace;
        font-size: 12px; 
        line-height: 1.4;
        padding: 10px; 
        border-radius: 4px;
        max-height: 250px; /* 增加日志最大高度 */
        overflow-y: auto; 
        white-space: pre-wrap;       /* 自动换行 */
        word-break: break-all;       /* 强制长单词换行，防止撑开容器 */
        border: 1px solid #334155;
    }

    /* 结果指标卡片 */
    .metric-card {
        background-color: white; 
        padding: 20px; 
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        border: 1px solid #e2e8f0; 
        color: #333;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def render_mermaid(code: str, height=650):
    # 修复：移除了 import URL 周围的 Markdown 语法 []()
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">
        {code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
    """
    components.html(html_code, height=height)

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.title("🎛️ Agent 控制台")
    # 
    render_mermaid("""
    graph TD
        User[Input] --> Planner
        Planner -->|Code| SB1[Sandbox]
        SB1 -->|List| Map((Map))
        Map --> W1[Worker]
        Map --> W2[Worker]
        W1 -->|Tools| SB2[Sandbox]
        W2 -->|Tools| SB2[Sandbox]
        SB2 --> Agg[Aggregator]
        Agg --> Result
        style Agg fill:#10b981,stroke:#fff
    """)
    st.info("后端Agent运行在 AgentCube平台，并发使用 CodeInterpreter工具进行量化分析")

# ==========================================
# 3. 主逻辑
# ==========================================
st.title("⚡️ 股票并发量化分析 Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("指令: 分析 A 股最热门的 3 只银行股"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        monitor_expander = st.expander("👁️ Agent思维链追踪", expanded=True)
        result_container = st.container()
        
        worker_state = {} 
        analysis_finished = False 
        final_advice = None
        final_details = []

        try:
            with monitor_expander:
                dashboard = st.empty()
                # 发起流式请求
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

                    # 初始化状态
                    if ticker not in worker_state:
                        worker_state[ticker] = {"status": "Waiting...", "logs": [], "class": ""}

                    # Update State
                    if evt_type == "node_status":
                        worker_state["System"]["status"] = f"Node: {event.get('node')}"
                    elif evt_type == "llm_stream":
                        worker_state[ticker]["status"] = "🧠 Thinking..."
                        worker_state[ticker]["class"] = "running"
                    elif evt_type == "code_input":
                        worker_state[ticker]["logs"].append(f"> CODE SENT:\n{event.get('code','')[:150]}...")
                        worker_state[ticker]["status"] = "⚙️ Running Code..."
                    elif evt_type == "tool_output":
                        # 清洗输出，防止 Base64 刷屏
                        clean_out = event.get("output","").split("###IMAGE")[0]
                        worker_state[ticker]["logs"].append(f"< OUTPUT:\n{clean_out[:200]}...")
                        worker_state[ticker]["status"] = "✅ Done"
                        worker_state[ticker]["class"] = "done"
                    elif evt_type == "final_result":
                        final_advice = event.get("advice")
                        final_details = event.get("details")
                        analysis_finished = True

                    # 实时渲染 Dashboard
                    with dashboard.container():
                        # System 全局状态
                        if "System" in worker_state:
                            st.info(f"System Status: {worker_state['System']['status']}")
                        
                        # Worker 卡片渲染
                        active = [k for k in worker_state if k != "System"]
                        if active:
                            # 动态列数：如果少于3个就按实际数量，否则一行3个
                            cols = st.columns(min(len(active), 3))
                            for idx, t in enumerate(active):
                                info = worker_state[t]
                                with cols[idx % 3]:
                                    # 优化后的 HTML 结构
                                    st.markdown(f"""
                                    <div class="worker-box {info['class']}">
                                        <div class="worker-header">
                                            <span>{t}</span>
                                            <span style="font-size:0.8em; opacity:0.8">{info['status']}</span>
                                        </div>
                                        <div class="terminal-log">{info['logs'][-1] if info['logs'] else 'Waiting for logs...'}</div>
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
                        
                        # 1. 柱状图对比
                        chart_data = [{"Ticker": d["ticker"], "Score": d["score_data"].get("score", 0)} for d in final_details if d.get("score_data")]
                        if chart_data:
                            # 修复：use_container_width
                            st.bar_chart(pd.DataFrame(chart_data).set_index("Ticker"), color="#10b981", use_container_width=True)

                        # 2. 详情卡片
                        cols = st.columns(len(final_details))
                        for idx, d in enumerate(final_details):
                            with cols[idx]:
                                sc = d.get("score_data", {})
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{d['ticker']}</h4>
                                    <h2 style="color:#10b981">{sc.get('score', 0):.4f}</h2>
                                    <div style="font-size:0.8em; color:#666; margin-top:5px;">
                                        <div>Momentum: {sc.get('momentum',0):.4f}</div>
                                        <div>Volatility: {sc.get('volatility',0):.4f}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if d.get("image_base64"):
                                    # 修复：use_container_width
                                    st.image(base64.b64decode(d["image_base64"]), use_container_width=True)

                    st.session_state.messages.append({"role": "assistant", "content": final_advice})

        except Exception as e:
            st.error(f"Client Error: {e}")