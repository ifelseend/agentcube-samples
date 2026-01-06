import streamlit as st
import requests
import uuid

# ==========================================
# 1. 配置页面
# ==========================================
st.set_page_config(
    page_title="Agent Game Studio",
    layout="wide",  # 使用宽屏模式，方便左右分栏
    initial_sidebar_state="expanded"
)

# 后端 API 地址
#API_URL = "http://124.70.75.33"
API_URL = "http://127.0.0.1:8080"

# ==========================================
# 2. 初始化 Session State (状态管理)
# ==========================================
# 生成唯一的会话 ID，保证 Agent 记住上下文
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{str(uuid.uuid4())[:8]}"

# 聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的游戏开发 Agent。想做什么游戏？\n试试说：'帮我做一个贪吃蛇游戏'。"}
    ]

# 当前的游戏代码 HTML
if "game_html" not in st.session_state:
    st.session_state.game_html = None

# ==========================================
# 3. 页面布局
# ==========================================

st.title("🎮 Agent Game Studio")

# 创建左右两列：左边聊天(40%)，右边预览(60%)
col_chat, col_preview = st.columns([0.4, 0.6], gap="medium")

# --- 左侧：聊天区域 ---
with col_chat:
    st.subheader("💬 对话指令")
    
    # 创建一个容器来显示聊天历史，防止输入框把消息顶出屏幕
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # 输入框 (Streamlit 默认固定在底部)
    if prompt := st.chat_input("描述你的游戏需求..."):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # 2. 调用后端 API
        with chat_container:
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown("🤖 *Agent 正在思考并编写代码...*")
                
                try:
                    response = requests.post(API_URL, json={
                        "message": prompt,
                        "thread_id": st.session_state.session_id
                    }, timeout=1200) # Agent 写代码可能比较慢，设置较长超时
                    
                    if response.status_code == 200:
                        data = response.json()
                        text_reply = data.get("text_response", "")
                        
                        # 显示文本回复
                        status_placeholder.write(text_reply)
                        st.session_state.messages.append({"role": "assistant", "content": text_reply})
                        
                        # 3. 核心逻辑：检查是否有新游戏代码生成
                        file_data = data.get("generated_file")
                        if file_data and file_data.get("content"):
                            st.session_state.game_html = file_data["content"]
                            st.toast(f"🎉 成功生成/更新游戏：{file_data.get('filename')}", icon="🚀")
                            # 强制刷新以更新右侧预览
                            st.rerun()
                    else:
                        error_msg = f"❌ API 错误: {response.status_code} - {response}"
                        status_placeholder.error(error_msg)
                        
                except Exception as e:
                    status_placeholder.error(f"❌ 连接失败: {str(e)}")

# --- 右侧：游戏预览区域 ---
with col_preview:
    st.subheader("🖥️ 实时预览")
    
    if st.session_state.game_html:
        # 使用 Streamlit 组件渲染 HTML iframe
        # scrolling=False 且 height 设置大一点，模拟全屏游戏体验
        st.components.v1.html(
            st.session_state.game_html, 
            height=600, 
            scrolling=False
        )
        st.caption("提示：点击上方游戏区域以获取键盘焦点")
        
        # 下载按钮
        st.download_button(
            label="📥 下载 HTML 文件",
            data=st.session_state.game_html,
            file_name="my_agent_game.html",
            mime="text/html"
        )
    else:
        # 空状态占位符
        st.info("👈 在左侧告诉 Agent 你想做什么游戏，预览将在这里出现。")
        st.markdown(
            """
            <div style='background: #f0f2f6; height: 500px; display: flex; align-items: center; justify-content: center; color: #888; border-radius: 10px; border: 2px dashed #ccc;'>
                Waiting for Code Generation...
            </div>
            """, 
            unsafe_allow_html=True
        )

# 侧边栏说明
with st.sidebar:
    st.markdown("### 关于")
    st.markdown("这是一个 **CodeAct Agent** 的演示前端。")
    st.markdown("后端运行在 **AgentCube** 上，Agent具备调用Code Interpreter执行Python代码的能力。")
    st.markdown("---")
    if st.button(" 新游戏 / 清除历史"):
        st.session_state.messages = []
        st.session_state.game_html = None
        st.session_state.session_id = f"user_{str(uuid.uuid4())[:8]}"
        st.rerun()
    
