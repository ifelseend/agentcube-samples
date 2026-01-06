import os
import sys
import io
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agentcube import CodeInterpreterClient
# ==========================================
# 1. 环境配置
# ==========================================

# 定义沙箱内的工作目录 (容器内路径)
# 生产环境通常为 /tmp/sandbox 或 /app/data
WORK_DIR = os.getenv("WORK_DIR", "/tmp/sandbox")
os.makedirs(WORK_DIR, exist_ok=True)

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
api_key = os.getenv("OPENAI_API_KEY", "")
api_base_url = os.getenv("OPENAI_API_BASE", "")
model_name = os.getenv("OPENAI_MODEL", "")

# ==========================================
# 1. 通信协议定义 (Tags)
# ==========================================
# 用于在 stdout 杂乱的日志中精准提取 HTML 代码
HTML_START_TAG = ""
HTML_END_TAG = ""

# ==========================================
# 2. System Prompt (核心指令)
# ==========================================
SYSTEM_PROMPT = f"""
### 1. IDENTITY & OBJECTIVE
You are the **"AI Game Studio Architect"**. 
Your mission is to design, code, and deploy playable, single-file HTML5/Canvas games instantly.
You operate in a **Dual-Engine Architecture**:
1. **Python Engine**: For logic generation, complex math, procedural level design, and data processing.
2. **Web Engine**: For rendering and user interaction (HTML/JS).

### 2. CORE WORKFLOW: "COMPUTE FIRST, THEN RENDER"
To create high-quality games, you must follow this strict process:
1. **Compute**: Use Python to calculate game assets (mazes, physics, sudoku grids).
2. **Inject**: Embed this calculated data into the JS logic.
3. **Render**: Output the final HTML.

### 3. SECURITY & ENVIRONMENT PROTOCOLS
- **NO DISK ACCESS**: You are running in a stateless container. **DO NOT** use `open()`, `write()`, or `save()` to disk.
- **NO FILESYSTEM**: Do not assume files persist. 

### 4. ATOMIC EXECUTION PROTOCOL (CRITICAL)
**To prevent `NameError`, you must execute EVERYTHING in ONE SINGLE Python block.**
- **DO NOT** split logic into multiple tool calls (e.g., do not calculate data in one turn and print HTML in the next).
- **Variable Persistence is NOT guaranteed.** If you define `json_data` in step 1 and try to use it in step 2, it will fail.
- **RULE**: Imports -> Data Calculation -> HTML Definition -> Injection -> Print. **ALL IN ONE SCRIPT.**

### 5. DATA INJECTION & SYNTAX RULES (ANTI-CRASH)
**Follow this exact template to avoid `NameError: name 'json_data_str' is not defined`:**

1. **NO F-STRINGS FOR HTML**: Do **NOT** use `f\"\"\"...\"\"\"` for the HTML template. It conflicts with JS syntax (`${{var}}`).
2. **SERIALIZE FIRST**: You MUST define the JSON string variable **immediately** after imports.

**FOOLPROOF CODE TEMPLATE (Copy this structure exactly):**
```python
import json
import random

# --- STEP 1: PREPARE DATA (Define variables FIRST) ---
# Generate your python data
game_logic = {{"speed": 10, "map": [[1,0],[0,1]]}}

# CRITICAL: Serialize to string immediately. Name it 'data_payload'.
data_payload = json.dumps(game_logic)

# --- STEP 2: DEFINE TEMPLATE (Standard String, NO f-string) ---
# Use a unique placeholder like __DATA_HERE__
html_template = \"\"\"
<!DOCTYPE html>
<html>
<body>
<script>
    // Inject data here
    const gameData = __DATA_HERE__; 
    console.log(gameData);
</script>
</body>
</html>
\"\"\"

# --- STEP 3: INJECT & PRINT (Safe Replacement) ---
# Use the variable 'data_payload' defined in Step 1
final_html = html_template.replace("__DATA_HERE__", data_payload)

# Print using the mandatory tags
print(f"{HTML_START_TAG}")
print(final_html)
print(f"{HTML_END_TAG}")

### 6. ERROR RECOVERY (CRITICAL)
If the Python execution returns an "Error" or "Traceback":
1. **DO NOT** tell the user you finished.
2. **DO NOT** apologize profusely.
3. **IMMEDIATELY** analyze the error, fix the code, and **RUN IT AGAIN**.
4. You function is ONLY complete when the `` tag is successfully printed.
"""

# ==========================================
# 3. 工具定义
# ==========================================

# 全局上下文，保持变量状态 (模拟 Jupyter)
_GLOBAL_PYTHON_CONTEXT = {"WORK_DIR": WORK_DIR}

ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")

@tool
def agentcube_code_interpreter(code: str) -> str:
    """
    Executes python code in a secure sandbox.
    Use this tool to perform calculations, analyze data, or run any python script.
    Input: Valid Python code string.
    Output: Standard Output (stdout) or Error (stderr).
    Context: The code is executed in a persistent session, so variables defined in previous calls are available.
    """
    try:
        # Run the code
        return ci_client.run_code("python", code)

    except Exception as e:
        error_msg = f"Error executing Python code: Could not connect to Code Interpreter backend. Details: {e}"
        print(f"ERROR: {error_msg}")
        return error_msg


# ==========================================
# 4. 图构建 (Graph Construction)
# ==========================================


llm = init_chat_model(
        model_name,
        model_provider="openai",
        base_url=api_base_url,
        api_key=api_key,
        temperature=0.1
    )

tools = [agentcube_code_interpreter]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState):
    messages = state["messages"]
    
    # 【关键步骤】确保 SystemMessage 永远在消息列表的最前面
    # 如果历史消息里没有 SystemMessage，我们手动插入一个
    if not isinstance(messages[0], SystemMessage):
        sys_msg = SystemMessage(content=SYSTEM_PROMPT)
        # 注意：这里我们构造一个新的 list 传给 LLM，但不修改 state 里的历史，
        # 避免 LangGraph 每次都 append 重复的 SystemPrompt
        all_messages = [sys_msg] + messages
    else:
        # 如果第一条已经是，可能需要更新 Prompt (比如 WORK_DIR 变了)，这里简单处理直接用
        all_messages = messages

    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# 定义 Checkpointer (内存记忆)
memory = MemorySaver()

# 构建工作流
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 编译应用
graph_app = workflow.compile(checkpointer=memory)