import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import base64
import pandas as pd

# ==========================================
# 1. 配置与样式
# ==========================================

BACKEND_STREAM_URL = "http://localhost:8000/analyze_stream"

st.set_page_config(
    page_title="AgentCube Stock Analyst",
    page_icon="🛡️", # 换个图标表示已加固
    layout="wide"
)

# 注入 CSS：增加了一些动画效果和状态颜色
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* Worker 状态卡片 */
    .worker-box {
        background-color: #2b2d42;
        color: #edf2f4;
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 6px;
        border-left: 5px solid #8d99ae;
        transition: transform 0.2s;
    }
    .worker-box.running { border-left-color: #ffa726; } /* 运行中显示橙色 */
    .worker-box.done { border-left-color: #4caf50; }    /* 完成显示绿色 */
    .worker-box.error { border-left-color: #ef5350; }   /* 错误显示红色 */

    .worker-header {
        display: flex;
        justify-content: space-between;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .worker-status { font-size: 12px; color: #bbb; }
    
    /* 终端日志 */
    .terminal-log {
        background: #000;
        color: #0f0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        padding: 8px;
        border-radius: 4px;
        max-height: 120px;
        overflow-y: auto;
        white-space: pre-wrap;
        margin-top: 5px;
    }

    /* 结果卡片 */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #eee;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 辅助函数
# ==========================================

def render_mermaid(code: str, height=400):
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">
        {code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'base' }});
    </script>
    """
    components.html(html_code, height=height)

# ==========================================
# 3. 侧边栏
# ==========================================

with st.sidebar:
    st.title("🎛️ 监控控制台")
    st.caption("Map-Reduce 架构可视化")
    render_mermaid("""
    graph TD
        User --> Planner
        Planner -->|Code| Sandbox1
        Sandbox1 -->|Tickers| Map
        Map --> W1[Worker A]
        Map --> W2[Worker B]
        W1 -->|Tools| SB1[Session A]
        W2 -->|Tools| SB2[Session B]
        SB1 --> Agg[Aggregator]
        SB2 --> Agg
        Agg --> Result
        style Agg fill:#bfb
    """, height=800)
    st.info("前端已启用：状态守卫 (State Guard) 与 异常熔断机制。")

# ==========================================
# 4. 主逻辑
# ==========================================

st.title("🛡️ 鲁棒版量化分析 Agent")
st.markdown("集成 **流中断检测** 与 **状态自动闭环** 机制")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("指令: 分析 A 股最热门的 3 只半导体股票"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # UI 容器
        monitor_expander = st.expander("👁️ 全链路执行监控 (Live)", expanded=True)
        result_container = st.container()
        
        # 状态变量
        worker_state = {}     # 存储 {"NVDA": {"status": "running", "log": "..."}}
        final_advice = None
        final_details = None
        
        # 🛡️ 守卫标志位：判断流程是否正常结束
        analysis_finished = False 

        try:
            with monitor_expander:
                dashboard = st.empty() # 用于不断刷新 Dashboard
                
                # 发起请求
                # timeout=300 设置 5 分钟超时，防止无限挂起
                response = requests.post(BACKEND_STREAM_URL, json={"query": prompt}, stream=True, timeout=300)
                
                # 检查 HTTP 状态码，如果一开始就 500，直接报错
                if response.status_code != 200:
                    st.error(f"服务器拒绝连接 (HTTP {response.status_code})")
                    raise Exception("Backend Error")

                # === 流式读取循环 ===
                for line in response.iter_lines():
                    if not line: continue
                    
                    try:
                        # JSON 容错解析
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue # 跳过坏数据，不崩溃

                    evt_type = event.get("type")
                    ticker = event.get("ticker", "System")
                    if not ticker: ticker = "System"

                    # 初始化 Ticker 状态
                    if ticker not in worker_state:
                        worker_state[ticker] = {"status_text": "Waiting...", "logs": [], "state_class": "running"}

                    # --- 事件处理 ---
                    
                    if evt_type == "node_status":
                        node = event.get("node")
                        if node == "planner": worker_state["System"]["status_text"] = "🧠 Planner 正在解析..."
                        elif node == "worker": worker_state["System"]["status_text"] = "⚡️ 并发分发中..."
                        elif node == "aggregator": worker_state["System"]["status_text"] = "🏆 正在汇总..."

                    elif evt_type == "llm_stream":
                        worker_state[ticker]["status_text"] = f"✍️ Writing Code... {event.get('chunk','')[-10:]}"
                        worker_state[ticker]["state_class"] = "running"

                    elif evt_type == "code_complete":
                        worker_state[ticker]["status_text"] = "⏳ 代码编写完成，等待沙箱..."

                    elif evt_type == "code_input":
                        code = event.get("code","")
                        worker_state[ticker]["logs"].append(f">>> CODE SENT:\n{code[:100]}...")
                        worker_state[ticker]["status_text"] = "🚀 Running in Sandbox..."

                    elif evt_type == "tool_output":
                        out = event.get("output","")
                        clean = out.split("###IMAGE")[0]
                        worker_state[ticker]["logs"].append(f"<<< OUTPUT:\n{clean[:200]}...")
                        worker_state[ticker]["status_text"] = "✅ Execution Done"
                        worker_state[ticker]["state_class"] = "done"

                    elif evt_type == "final_result":
                        final_advice = event.get("advice")
                        final_details = event.get("details")
                        # 🛡️ 收到此信号，才算真正成功
                        analysis_finished = True 

                    # --- 实时渲染 Dashboard ---
                    # 每次循环都重绘一次，实现动画效果
                    with dashboard.container():
                        # System 状态
                        if "System" in worker_state:
                            st.info(f"System: {worker_state['System']['status_text']}")
                        
                        # Worker 卡片网格
                        active_workers = [k for k in worker_state if k != "System"]
                        if active_workers:
                            cols = st.columns(min(len(active_workers), 3))
                            for idx, t in enumerate(active_workers):
                                info = worker_state[t]
                                with cols[idx % 3]:
                                    # 动态生成 HTML class (running/done/error)
                                    css_class = info.get("state_class", "")
                                    last_log = info["logs"][-1] if info["logs"] else ""
                                    
                                    st.markdown(f"""
                                    <div class="worker-box {css_class}">
                                        <div class="worker-header">
                                            <span>{t}</span>
                                            <span>{info['status_text'][:15]}..</span>
                                        </div>
                                        <div class="terminal-log">{last_log}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

            # === 🛡️ 守卫逻辑：循环结束后的检查 ===
            
            if not analysis_finished:
                # 情况 A: 流中断了，但没有收到 final_result
                st.error("❌ **连接异常中断**：后端服务未返回最终结果就断开了连接。")
                st.warning("可能原因：后端报错崩溃、Tushare 接口超时或网络不稳定。请查看后端终端日志。")
                
                # 强制更新 UI 状态为“错误”，让用户知道不用等了
                with dashboard.container():
                    st.error("SYSTEM HALTED")
            
            else:
                # 情况 B: 正常结束
                with result_container:
                    # 1. 结论
                    st.success("✅ 分析任务完成")
                    st.markdown(final_advice)
                    st.divider()

                    # 2. 可视化
                    if final_details:
                        st.subheader("📊 结果看板")
                        
                        # 横向对比
                        chart_data = []
                        for item in final_details:
                            s = item.get("score_data") or {}
                            chart_data.append({
                                "Name": item.get("ticker"),
                                "Score": s.get("score", 0)
                            })
                        if chart_data:
                            df = pd.DataFrame(chart_data).set_index("Name")
                            st.bar_chart(df["Score"], color="#4CAF50")

                        st.divider()

                        # 详情卡片
                        cols = st.columns(len(final_details))
                        for idx, item in enumerate(final_details):
                            with cols[idx]:
                                ticker = item.get("ticker")
                                score = (item.get("score_data") or {}).get("score", 0)
                                report = item.get("report", "")[:100]
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{ticker}</h3>
                                    <div style="font-size:24px; color:#2e7d32; font-weight:bold;">{score}</div>
                                    <div style="font-size:12px; color:#666;">{report}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                img_b64 = item.get("image_base64")
                                if img_b64:
                                    try:
                                        st.image(base64.b64decode(img_b64), use_container_width=True)
                                    except:
                                        st.caption("无图表")

                    # 记录历史
                    st.session_state.messages.append({"role": "assistant", "content": final_advice})

        except requests.exceptions.ConnectionError:
            st.error("❌ 无法连接到后端服务器。请确认 `python backend_parallel.py` 正在运行。")
        except requests.exceptions.ReadTimeout:
            st.error("❌ 请求超时。Agent 思考或运行代码时间过长。")
        except Exception as e:
            st.error(f"❌ 发生未知错误: {e}")