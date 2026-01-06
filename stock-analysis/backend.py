import os
import re
import json
import base64
import random
import io
import operator
import langchain
import traceback
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send
from agentcube import CodeInterpreterClient

langchain.debug = True
langchain.verbose = True

load_dotenv()
# Configuration from environment variables
api_key = os.getenv("OPENAI_API_KEY", "")
api_base_url = os.getenv("OPENAI_API_BASE", "")
model_name = os.getenv("OPENAI_MODEL", "")

app = FastAPI(title="Concurrent Stock Agent")
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"后端崩溃: {str(exc)}\n详细堆栈:\n{traceback.format_exc()}"
    print(f"❌ CRITICAL ERROR:\n{error_msg}")
    # 返回 200 或 400，把错误信息传给前端显示，而不是直接 500
    return JSONResponse(
        status_code=200, 
        content={
            "advice": f"### ⚠️ 系统错误\n\n程序遇到了未处理的异常，请检查后端日志。\n\n```text\n{str(exc)}\n```",
            "details": []
        }
    )
# ==========================================
# 1. 模拟环境 (Mock Tools & LLM)
# ==========================================
# 实际项目中请设置 USE_MOCK_MODE = False 并配置 Keys
USE_MOCK_MODE = False

class MockAgentCubeClient:
    def run_code(self, runtime, code, session_id):
        # 简单解析代码以获取 ticker
        ticker = session_id.replace("sess_", "") if "sess_" in session_id else "UNKNOWN"
        
        # 模拟生成图片
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 4))
        plt.plot(np.cumsum(np.random.randn(50)) + 100)
        plt.title(f"{ticker} Trend (Parallel Worker)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 模拟数据
        score = round(random.uniform(0.5, 3.0), 2)
        json_data = json.dumps({"ticker": ticker, "score": score, "momentum": 0.5, "volatility": 0.2})
        
        stdout = (
            f"Analyzing {ticker} in session {session_id}...\n"
            f"###IMAGE_BASE64_START###{img_b64}###IMAGE_BASE64_END###\n"
            f"###JSON_DATA_START###{json_data}###JSON_DATA_END###\n"
            "Done."
        )
        return {"stdout": stdout, "stderr": ""}

if USE_MOCK_MODE:
    ci_client = MockAgentCubeClient()
    # 如果没 Key，LangChain 初始化会报错，这里假设环境有 Key 或使用 MockLLM
    # 实际运行请确保环境变量 OPENAI_API_KEY 存在 (即便是乱填的在 Mock 模式下也能跑，只要不真正 invoke)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
else:
    llm = init_chat_model(
        model_name,
        model_provider="openai",
        base_url=api_base_url,
        api_key=api_key,
        temperature=0.1
    )

# ==========================================
# 2. 工具定义 (Worker 专用)
# ==========================================

@tool
def agentcube_code_interpreter(code: str, session_id: str) -> str:
    """
    Executes Python code. 
    Args:
        code: Python script.
    """
    ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")
    try:
        # Run the code
        return ci_client.run_code("python", code)

    except Exception as e:
        error_msg = f"Error executing Python code: Could not connect to Code Interpreter backend. Details: {e}"
        print(f"ERROR: {error_msg}")
        return error_msg
    finally:
        ci_client.stop()


# 绑定工具
tools = [agentcube_code_interpreter]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 子图构建 (Worker Subgraph: 负责单只股票)
# ==========================================

class WorkerState(TypedDict):
    ticker: str                             # 输入：股票代码
    messages: Annotated[list[BaseMessage], add_messages] # 内部对话历史
    final_report: str                       # 输出：最终报告

def worker_agent_node(state: WorkerState):
    """子图的思考节点"""
    messages = state["messages"]
    ticker = state["ticker"]
    
    # 1. 从环境变量获取 Token (确保 .env 里配置了 TUSHARE_TOKEN)
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    
    # 2. 构造 System Prompt
    system_msg = SystemMessage(content=f"""
    You are a Quantitative Analyst analyzing stock: {ticker}.
    
    ### 1. EXECUTION PROTOCOL (CRITICAL)
    **ATOMIC EXECUTION REQUIRED**: You are running in a stateless environment. Variables are NOT saved between code blocks.
    - You MUST write **ONE SINGLE Python script** that does EVERYTHING: Imports -> Init API -> Fetch Data -> Calculate -> Print.
    - **DO NOT** try to use variables defined in previous turns (like `pro` or `df`). Redefine them every time.
    
    ### 2. CODE TEMPLATE (Use Tushare Pro)
    You MUST use this exact structure to avoid "NameError" and "Pandas Version" errors:
    
    ```python
    import tushare as ts
    import pandas as pd
    import numpy as np
    import io
    import base64
    import matplotlib.pyplot as plt
    import json

    # 1. Initialize Pro API (Must happen INSIDE this script)
    ts.set_token('{tushare_token}')
    pro = ts.pro_api()

    # 2. Fetch Data (Use pro.daily ONLY. Do NOT use ts.get_k_data which fails on Pandas 2.0)
    # Tushare daily data is descending. We MUST sort it.
    df = pro.daily(ts_code='{ticker}', start_date='20230101', end_date='20240101')
    
    if df is None or df.empty:
        print("Error: No data found for {ticker}")
    else:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 3. Calculate Indicators (Momentum-Volatility)
        # Convert percent to decimal: 1.5 -> 0.015
        df['pct_chg'] = df['pct_chg'] / 100 
        
        # Momentum (60 days)
        limit = 60
        if len(df) > limit:
            df['momentum'] = df['close'] / df['close'].shift(limit) - 1
            # Volatility (Annualized)
            df['volatility'] = df['pct_chg'].rolling(limit).std() * np.sqrt(252)
            
            # Score
            df['score'] = df['momentum'] / (df['volatility'] + 1e-6)
            
            # Get latest values
            last = df.iloc[-1]
            score_val = last['score']
            mom_val = last['momentum']
            vol_val = last['volatility']
        else:
            score_val, mom_val, vol_val = 0, 0, 0

        # 4. Plotting (Matplotlib -> Base64)
        plt.figure(figsize=(10, 5))
        plt.plot(df['trade_date'], df['close'], label='Close')
        plt.title(f"Analysis: {ticker} (Score: {{score_val:.2f}})")
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 5. Output (Tags for parsing)
        print(f"###IMAGE_BASE64_START###{{img_b64}}###IMAGE_BASE64_END###")
        
        json_res = {{
            "ticker": "{ticker}",
            "score": round(float(score_val), 4),
            "momentum": round(float(mom_val), 4),
            "volatility": round(float(vol_val), 4)
        }}
        print(f"###JSON_DATA_START###{{json.dumps(json_res)}}###JSON_DATA_END###")
    ```
    """)
    
    # 确保 SystemPrompt 在最前
    if not isinstance(messages[0], SystemMessage):
        messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def worker_should_continue(state: WorkerState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "__end__"

# 构建子图
worker_graph = StateGraph(WorkerState)
worker_graph.add_node("agent", worker_agent_node)
worker_graph.add_node("tools", ToolNode(tools)) # 使用预置 ToolNode 执行代码

worker_graph.add_edge(START, "agent")
worker_graph.add_conditional_edges("agent", worker_should_continue)
worker_graph.add_edge("tools", "agent")

# 编译子图
stock_analyst_app = worker_graph.compile()

# ==========================================
# 4. 主图构建 (Main Graph: Map-Reduce)
# ==========================================

class OverallState(TypedDict):
    query: str
    ticker_list: List[str]
    # 收集所有 worker 的输出
    # operator.add 用于将列表合并: [res1] + [res2] = [res1, res2]
    reports: Annotated[List[Dict], operator.add] 
    final_advice: str

def planner_node(state: OverallState):
    """
    通用目标解析节点：统一处理“显式股票名称”和“隐式选股条件”。
    输出：一个确定的 ticker_list 供后续 Worker 并发分析。
    """
    user_query = state["query"]
    print(f"--- [Universal Planner] Resolving targets for: '{user_query}' ---")
    
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")

    # ============================================================
    # 统一 System Prompt：教会 LLM 何时提取，何时写代码
    # ============================================================
    system_prompt = f"""
    ### ROLE
    You are a **Stock Target Resolver**. Output a definitive list of stock tickers.

    ### DATA DICTIONARY (Tushare API)
    1. **`pro.daily_basic`** (Valuation Data):
       - Fields: `ts_code`, `pe`, `pb`, `circ_mv` (Market Cap), `turnover_rate`, `dv_ratio`.
       - ⚠️ **WARNING**: This table does **NOT** contain the `name` column.
    
    2. **`pro.stock_basic`** (Metadata):
       - Fields: `ts_code`, `name`, `industry`.

    ### COMMON PATTERNS & CORRECTIONS
    - **Scenario: "Remove ST stocks" or "Value Investing"**
      - ❌ WRONG: Filtering `name` directly on `daily_basic` -> `KeyError: 'name'`
      - ✅ RIGHT: You MUST fetch `stock_basic` and **MERGE** it with `daily_basic` before filtering names.
      
      ```python
      # Correct Logic:
      df_daily = pro.daily_basic(trade_date='20231220', fields='ts_code,pe,pb')
      df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
      
      # MERGE IS REQUIRED before filtering by name
      df = pd.merge(df_daily, df_info, on='ts_code', how='inner')
      
      # Now you can safely filter ST
      df = df[~df['name'].str.contains('ST')]
      ```

    - **Scenario: "Most Valuable / Best"**
      - Sort by `pe` (ascending) or `dv_ratio` (descending).

    ### EXECUTION PROTOCOL
    1. Init Tushare with token '{tushare_token}'.
    2. Use `pro.daily_basic` with a recent fixed date (e.g., '20231220') to avoid empty data.
    3. Output the tickers list wrapped in tags.

    ### CODE TEMPLATE
    ```python
    import tushare as ts
    import pandas as pd
    import json
    
    ts.set_token('{tushare_token}')
    pro = ts.pro_api()
    
    # 1. Fetch Data
    df_daily = pro.daily_basic(ts_code='', trade_date='20231220', fields='ts_code,pe,pb,circ_mv')
    df_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    
    # 2. Merge (CRITICAL STEP)
    df = pd.merge(df_daily, df_info, on='ts_code', how='inner')
    
    # 3. Filter & Sort (Based on user query: "{user_query}")
    # Example: Remove ST and Sort by PE
    df = df[~df['name'].str.contains('ST')]
    df = df[df['pe'] > 0]
    df = df.sort_values('pe', ascending=True).head(5)
    
    tickers = df['ts_code'].tolist()
    
    # 4. Output
    print(f"###TICKERS_START###{{json.dumps(tickers)}}###TICKERS_END###")
    ```
    """

    # 1. LLM 思考与代码生成
    # 这里我们不用 bind_tools，而是让 LLM 直接生成代码，我们在节点内执行
    # 这样可以统一处理“提取”和“筛选”两种逻辑，因为本质上“提取”也可以看作是一段简单的 Python 代码（直接 print list）
    generation = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Resolve targets for: {user_query}")
    ])

    # 2. 提取代码
    code = ""
    match = re.search(r"```python(.*?)```", generation.content, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # 容错：如果 LLM 没写 markdown，假设全是代码
        code = generation.content

    print(f"    (Generated Resolution Code, Length: {len(code)})")

    # 3. 执行代码 (AgentCube)
    # 使用 'planner_session'，保持独立
    ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")
    try:
        
        result = ci_client.run_code("python", code)
         # 解析 tickers
        tickers = []
        t_match = re.search(r"###TICKERS_START###(.*?)###TICKERS_END###", result, re.DOTALL)
        if t_match:
            tickers = json.loads(t_match.group(1))
        else:
            print(f"    [Warning] No tickers found in stdout. Stdout: {stdout[:100]}...")

    except Exception as e:
        print(f"    [Planner Error]: {e}")
        tickers = []
    finally:
        ci_client.stop()

    print(f"    (Resolved Targets): {tickers}")
    return {"ticker_list": tickers, "reports": []}

def output_parser_node(state: WorkerState):
    """
    桥接节点：位于子图结束之后，主图汇总之前。
    负责从 Worker 的对话历史中提取结构化数据 (Artifacts)。
    """
    messages = state["messages"]
    last_msg = messages[-1].content
    ticker = state["ticker"]
    
    # 提取 Artifacts
    score_data = {}
    img_b64 = ""
    
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # 提取 JSON
            json_match = re.search(r"###JSON_DATA_START###(.*?)###JSON_DATA_END###", msg.content, re.DOTALL)
            if json_match:
                try: score_data = json.loads(json_match.group(1))
                except: pass
            
            # 提取 图片
            img_match = re.search(r"###IMAGE_BASE64_START###(.*?)###IMAGE_BASE64_END###", msg.content, re.DOTALL)
            if img_match:
                img_b64 = img_match.group(1).strip()
            
            if score_data or img_b64:
                break
    
    if not score_data and not img_b64:
        # 记录错误日志
        print(f"    [Warning] Worker for {ticker} failed. Last msg: {last_msg[-100:]}")
        return {
            "reports": [{
                "ticker": ticker,
                "report": f"⚠️ 分析失败: 未能从代码执行结果中提取有效数据。\n可能原因: 代码报错或格式不符。",
                "score_data": {"score": -999}, # 给一个极低分，避免排到第一
                "image_base64": "" # 空图片
            }]
        }
    
    # 返回给 Main Graph 的 reports 列表
    return {
        "reports": [{
            "ticker": ticker,
            "report": last_msg,
            "score_data": score_data,
            "image_base64": img_b64
        }]
    }

def aggregator_node(state: OverallState):
    """汇总所有报告，生成建议"""
    print(f"--- [Aggregator] Combining {len(state['reports'])} reports ---")
    reports = state['reports']
    if not reports:
        return {
            "final_advice": "⚠️ **分析中断**：未能获取任何有效的股票分析报告。\n\n可能原因：\n1. Planner 未能识别出有效的股票代码。\n2. Tushare 接口调用失败（如 Token 无效或参数错误）。\n3. 所有 Worker 任务均执行失败。",
            "reports": []
        }
    
    try:
        # 过滤掉 score_data 为空的无效报告
        valid_reports = [r for r in reports if r.get('score_data')]
        
        if not valid_reports:
             return {"final_advice": "⚠️ **分析完成但无数据**：所有股票均未能计算出有效评分。", "reports": reports}

        best_stock = max(valid_reports, key=lambda x: x['score_data'].get('score', -999))
        
        final_advice = f"""
### 投资组合分析总结
共分析了 {len(valid_reports)} 只股票。

🥇 **最佳推荐**: **{best_stock['ticker']}** (Score: {best_stock['score_data'].get('score')})

**详细理由**: 根据动量-波动率模型，{best_stock['ticker']} 展现出了最优的风险回报比。
        """
        return {"final_advice": final_advice}
        
    except Exception as e:
        print(f"Aggregator Error: {e}")
        return {"final_advice": f"⚠️ 汇总阶段发生错误: {str(e)}", "reports": reports}
    
# --- 关键：使用 map_workflow 构建并发 ---

main_workflow = StateGraph(OverallState)
main_workflow.add_node("planner", planner_node)
main_workflow.add_node("aggregator", aggregator_node)

# 定义 Map 逻辑: 将 planner 的结果分发给 subgraph
# 我们在这里不直接把 subgraph 当节点，而是用 wrapper
# 也可以直接注册 subgraph, 但为了提取数据方便，我们可以在 Send 的目标处定义处理函数

# 注册 worker 节点（这里我们其实是把 compiled subgraph 当作一个函数调用）
# LangGraph 允许节点是一个 CompiledGraph
async def worker_wrapper(state: WorkerState):
    ticker = state["ticker"]
    # 运行子图
    result = await stock_analyst_app.ainvoke(state)
    return await stock_analyst_app.ainvoke(
        state, 
        config={"tags": [f"ticker:{ticker}"]}
    )

main_workflow.add_node("worker", worker_wrapper)

main_workflow.add_edge(START, "planner")

# Conditional Edge (Map Step)
main_workflow.add_conditional_edges(
    "planner",
    lambda state: [
        Send("worker", {"ticker": t, "messages": [HumanMessage(content="Start Analysis")]}) 
        for t in state["ticker_list"]
    ],
    ["worker"]
)

main_workflow.add_edge("worker", "aggregator")
main_workflow.add_edge("aggregator", END)

app_graph = main_workflow.compile()

# ==========================================
# 5. API 接口
# ==========================================
class QueryRequest(BaseModel):
    query: str

async def event_generator(query: str):
    """
    核心生成器：监听 LangGraph 事件并转换为前端可读的 SSE 数据流。
    包含全链路异常捕获，防止中途崩溃导致前端卡死。
    """
    inputs = {"query": query}
    print(f"🚀 [Backend Stream] Starting analysis for: {query}")

    try:
        # 使用 astream_events v2 获取细粒度事件
        async for event in app_graph.astream_events(inputs, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            
            # --- 1. 尝试从 Tags 中提取当前是哪个 Ticker (用于并发区分) ---
            # 默认为 System，如果 tag 里有 "ticker:NVDA"，则提取出 "NVDA"
            current_ticker = "System"
            for tag in tags:
                if tag.startswith("ticker:"):
                    current_ticker = tag.split(":")[1]
                    break
            
            # --- 2. 捕获流式 Token (消除卡顿感) ---
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield json.dumps({
                        "type": "llm_stream",
                        "ticker": current_ticker,
                        "chunk": content
                    }) + "\n"

            # --- 3. 捕获代码编写结束 (状态流转优化) ---
            # 当 LLM 写完代码时，通知前端更新状态，避免一直卡在"正在编写..."
            elif kind == "on_chat_model_end":
                output = event["data"].get("output")
                # 只有当 output 包含 tool_calls 时，才算代码生成完毕
                if output and hasattr(output, 'tool_calls') and output.tool_calls:
                     yield json.dumps({
                        "type": "code_complete", # 新增事件：代码写完了
                        "ticker": current_ticker
                    }) + "\n"

            # --- 4. 捕获工具输入 (可视化代码输入) ---
            elif kind == "on_tool_start":
                tool_name = event["name"]
                # 过滤掉系统内部工具，只看 Code Interpreter
                if "code_interpreter" in tool_name or tool_name == "agentcube_code_interpreter":
                    tool_input = event["data"].get("input")
                    # 兼容 input 是字典或字符串的情况
                    code_content = ""
                    if isinstance(tool_input, dict):
                        code_content = tool_input.get("code", str(tool_input))
                    else:
                        code_content = str(tool_input)
                    
                    yield json.dumps({
                        "type": "code_input",
                        "ticker": current_ticker,
                        "code": code_content
                    }) + "\n"

            # --- 5. 捕获工具输出 (可视化沙箱日志) ---
            elif kind == "on_tool_end":
                tool_output = str(event["data"].get("output"))
                yield json.dumps({
                    "type": "tool_output",
                    "ticker": current_ticker,
                    "output": tool_output
                }) + "\n"

            # --- 6. 节点状态流转 ---
            elif kind == "on_chain_start":
                node_name = event["name"]
                # 只关注核心大节点的流转
                if node_name in ["planner", "worker", "aggregator", "screener"]:
                    yield json.dumps({
                        "type": "node_status",
                        "node": node_name,
                        "ticker": current_ticker
                    }) + "\n"

        # === 循环结束，开始获取最终结果 ===
        print("🏁 [Backend Stream] Stream events finished. Fetching final state...")
        
        # 再次调用 ainvoke 获取最终的 return value
        # 注意：在生产环境中建议使用 Checkpointer 读取状态，避免重复执行
        # 但在这个 Demo 中，因为图是有向无环的，ainvoke 会直接返回结果（如果已经缓存）或快速重算
        final_state = await app_graph.ainvoke(inputs)
        
        # 安全获取，防止 KeyError
        advice = final_state.get("final_advice", "⚠️ 分析完成，但未生成总结建议。")
        details = final_state.get("reports", [])
        
        yield json.dumps({
            "type": "final_result",
            "advice": advice,
            "details": details
        }) + "\n"

    except Exception as e:
        # === 核心防崩溃机制 ===
        # 如果流过程中任何地方报错（比如 Planner 写错代码，或者 Tushare 连不上）
        # 捕获异常，并伪造一个 "final_result" 发给前端
        # 这样前端就会停止转圈，并显示错误信息
        print(f"❌ [Backend Crash]: {str(e)}")
        traceback.print_exc()
        
        error_msg = f"💥 **系统运行中断**\n\n错误详情: `{str(e)}`\n\n请检查后端日志以获取更多信息。"
        
        yield json.dumps({
            "type": "final_result",
            "advice": error_msg,
            "details": [] # 空详情
        }) + "\n"

@app.post("/analyze_stream")
async def analyze_stream_endpoint(req: QueryRequest):
    return StreamingResponse(event_generator(req.query), media_type="application/x-ndjson")

@app.post("/analyze")
async def analyze_endpoint(req: QueryRequest):
    inputs = {"query": req.query}
    # 运行主图
    final_state = await app_graph.ainvoke(inputs)
    
    return {
        "advice": final_state["final_advice"],
        "details": final_state["reports"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)