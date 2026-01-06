import os
import re
import json
import base64
import ast
import operator
import traceback
import logging
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# LangChain / LangGraph Imports
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send

from agentcube import CodeInterpreterClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 0. 配置与全局变量
# ==========================================
api_key = os.getenv("OPENAI_API_KEY", "")
api_base_url = os.getenv("OPENAI_API_BASE", "")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
tushare_token = os.getenv("TUSHARE_TOKEN", "")

# 确保关键环境变量存在
if not tushare_token:
    logger.warning("⚠️ TUSHARE_TOKEN 未配置，股票数据获取可能会失败。")

app = FastAPI(title="Concurrent Stock Agent (Real)")

class QueryRequest(BaseModel):
    query: str

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"后端崩溃: {str(exc)}\n详细堆栈:\n{traceback.format_exc()}"
    logger.error(f"❌ CRITICAL ERROR:\n{error_msg}")
    return JSONResponse(
        status_code=200, 
        content={
            "advice": f"### ⚠️ 系统错误\n\n程序遇到了未处理的异常，请检查后端日志。\n\n```text\n{str(exc)}\n```",
            "details": []
        }
    )

# ==========================================
# 1. 初始化 LLM (仅真实模式)
# ==========================================
llm = init_chat_model(
    model_name,
    model_provider="openai",
    base_url=api_base_url,
    api_key=api_key,
    temperature=0.1
)

# ==========================================
# 2. 工具定义 (真实 Code Interpreter)
# ==========================================

@tool
def agentcube_code_interpreter(code: str) -> str:
    """
    Executes Python code in a sandboxed environment using AgentCube.
    Args:
        code: The full Python script to execute.
    """
    # 实例化真实 Client
    # 注意：如果需要隔离环境，请确保 AgentCube 后端能根据请求自动隔离，或在此处根据上下文动态生成 name
    ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")
    try:
        logger.info(f"Running code via tool...")
        print(f" (code to run in tool): {code}")
        result = ci_client.run_code("python", code)
        #print(f" (code result in tool): {result}")
        
        return str(result)

    except Exception as e:
        error_msg = f"Error executing Python code: {str(e)}"
        logger.error(error_msg)
        return error_msg
    finally:
        ci_client.stop()

tools = [agentcube_code_interpreter]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 子图构建 (Worker Subgraph)
# ==========================================

class WorkerState(TypedDict):
    ticker: str
    messages: Annotated[list[BaseMessage], add_messages]


def worker_agent_node(state: WorkerState):
    """
    Worker 思考节点 (V4 - 确定性熔断版)：
    核心逻辑：一旦检测到工具输出包含了有效数据，直接在 Python 层强制结束，
    不再调用 LLM。这彻底解决了因 Token 瘦身导致的 LLM 犹豫和死循环问题。
    """
    messages = state["messages"]
    ticker = state["ticker"]
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    
    # ============================================================
    # 🛑 1. 确定性熔断 (Deterministic Stop) - 关键修改！
    # ============================================================
    # 检查最近的一条消息是否是工具输出
    if messages and isinstance(messages[-1], ToolMessage):
        last_output = messages[-1].content
        
        # 只要检测到 JSON 数据的开始标记，说明代码已经运行成功
        # 我们不需要 LLM 再废话了，直接人工构造一个结束信号
        if "###JSON_DATA_START###" in last_output:
            logger.info(f"✅ [Worker:{ticker}] Success detected. Enforcing HARD STOP.")
            
            # 直接返回一个 AIMessage（不带 tool_calls）
            # graph 里的 worker_should_continue 检测到没 tool_calls 就会直接结束
            return {"messages": [AIMessage(content="Analysis completed. Data generated successfully.")]}

    # ============================================================
    # 2. System Prompt (保持不变)
    # ============================================================
    system_msg = SystemMessage(content=f"""
    You are a Quantitative Analyst analyzing stock: {ticker}.
    
    ### TASK
    Write Python code to analyze the stock using `tushare`.
    
    ### PROTOCOL
    1. **Environment**: You have `tushare`, `pandas`, `numpy`, `matplotlib`.
    2. **Token**: Use `ts.set_token('{tushare_token}')`.
    3. **One-Shot**: Write ONE single, complete script.
    
    ### CODE STRUCTURE
    1. Init Tushare Pro API.
    2. Fetch `daily` data (past 1 year). **Sort by date ascending**.
    3. Calculate:
       - Momentum (Current Close / Close 60 days ago - 1)
       - Volatility (StdDev of daily returns * sqrt(252))
       - Score (Momentum / Volatility)
    4. Plot trend chart -> Convert to Base64.
    
    5. **OUTPUT FORMATTING (CRITICAL)**:
       - You MUST convert all Numpy types to Python native types (float/int) before printing.
       - Use this exact print statement structure:
       
       ```python
       import json
       import numpy as np
       
       class NpEncoder(json.JSONEncoder):
           def default(self, obj):
               if isinstance(obj, np.integer): return int(obj)
               if isinstance(obj, np.floating): return float(obj)
               if isinstance(obj, np.ndarray): return obj.tolist()
               return super(NpEncoder, self).default(obj)
               
       res = {{
           "ticker": "{ticker}",
           "score": latest_score,
           "momentum": latest_momentum,
           "volatility": latest_volatility
       }}
       
       # Print Image Tag
       print(f"###IMAGE_BASE64_START###{{plot_base64}}###IMAGE_BASE64_END###")
       
       # Print JSON Tag (Use cls=NpEncoder)
       print(f"###JSON_DATA_START###{{json.dumps(res, cls=NpEncoder)}}###JSON_DATA_END###")
       ```
    """)
    
    full_history = [system_msg] + messages if not isinstance(messages[0], SystemMessage) else messages
    
    # ============================================================
    # 3. Context Sanitization (Token 瘦身 - 保持开启，省钱)
    # ============================================================
    sanitized_history = []
    
    for msg in full_history:
        if isinstance(msg, ToolMessage):
            # 替换 Base64，防止 Context Window 爆炸
            clean_content = re.sub(
                r"###IMAGE_BASE64_START###(.*?)###IMAGE_BASE64_END###", 
                "###IMAGE_BASE64_START###[IMAGE_HIDDEN]###IMAGE_BASE64_END###", 
                msg.content, 
                flags=re.DOTALL
            )
            # 防止 JSON 过长
            if len(clean_content) > 10000:
                clean_content = clean_content[:5000] + "\n...[TRUNCATED]...\n" + clean_content[-1000:]
            
            sanitized_history.append(ToolMessage(content=clean_content, tool_call_id=msg.tool_call_id))
        else:
            sanitized_history.append(msg)

    # ============================================================
    # 4. 调用 LLM
    # ============================================================
    # 只有在没有检测到成功标记时，才走到这一步
    response = llm_with_tools.invoke(sanitized_history)
    return {"messages": [response]}

def worker_should_continue(state: WorkerState) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "__end__"

worker_graph = StateGraph(WorkerState)
worker_graph.add_node("agent", worker_agent_node)
worker_graph.add_node("tools", ToolNode(tools))

worker_graph.add_edge(START, "agent")
worker_graph.add_conditional_edges("agent", worker_should_continue)
worker_graph.add_edge("tools", "agent")

stock_analyst_app = worker_graph.compile()

# ==========================================
# 4. 主图构建 (Main Graph)
# ==========================================

class OverallState(TypedDict):
    query: str
    ticker_list: List[str]
    reports: Annotated[List[Dict], operator.add] 
    final_advice: str

def planner_node(state: OverallState):
    """Planner: 解析查询并生成股票列表 (增强版：修复单引号JSON报错 + 去重)"""
    user_query = state["query"]
    tushare_token = os.environ.get("TUSHARE_TOKEN", "")
    
    if not tushare_token:
        logger.error("❌ TUSHARE_TOKEN is missing!")
        return {"ticker_list": [], "reports": []}

    print(f"--- [Planner] Resolving: '{user_query}' ---")

    system_prompt = f"""
    You are a Stock Target Resolver. 
    Write Python code using `tushare` to find stock tickers based on: "{user_query}".
    
    ### RULES
    1. Use `ts.set_token('{tushare_token}')`.
    2. **Retry Logic**: Use a loop to retry API calls (max 3 times) to handle timeouts.
    3. **Merge Strategy**: 
       - Fetch `pro.stock_basic` (name, symbol) AND `pro.daily_basic` (pe, pb, mv).
       - MERGE them on `ts_code`.
    4. **Deduplication**: Use `df.drop_duplicates(subset=['ts_code'])` to avoid repeated tickers.
    5. **Output**: Print the list using `json.dumps()` (Double Quotes required!).
    
    ### CODE TEMPLATE
    ```python
    import tushare as ts
    import pandas as pd
    import json
    import time

    ts.set_token('{tushare_token}')
    pro = ts.pro_api()

    def call_api(func, **kwargs):
        for i in range(3):
            try:
                return func(**kwargs)
            except:
                time.sleep(1)
        return pd.DataFrame()

    # 1. Fetch Basic Info (Name lookup)
    df_info = call_api(pro.stock_basic, exchange='', list_status='L', fields='ts_code,name,industry')
    
    # 2. Fetch Valuation (Optional but good for ranking)
    df_daily = call_api(pro.daily_basic, trade_date='20240105', fields='ts_code,pe,pb,total_mv')

    tickers = []
    if not df_info.empty:
        # Merge if we have daily data, otherwise just use info
        if not df_daily.empty:
            df = pd.merge(df_info, df_daily, on='ts_code', how='inner')
        else:
            df = df_info

        # --- FILTERING LOGIC START ---
        # (Example: Find by name)
        # df = df[df['name'].str.contains('keyword')]
        # --- FILTERING LOGIC END ---
        
        # Deduplicate is CRITICAL
        df = df.drop_duplicates(subset=['ts_code'])
        
        # Limit to top 5
        tickers = df.head(5)['ts_code'].tolist()

    # STRICTLY use json.dumps for double quotes
    print(f"###TICKERS_START###{{json.dumps(tickers)}}###TICKERS_END###")
    ```
    """
    
    generation = llm.invoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=f"Find tickers for: {user_query}")
    ])
    
    code = generation.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()

    # 执行代码
    ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")
    tickers = []
    try:
        print(f" (code): {code}")
        result = ci_client.run_code("python", code)
        print(f" (code result): {result}")

        stdout = result.get("stdout", "") if isinstance(result, dict) else str(result)
        
        t_match = re.search(r"###TICKERS_START###(.*?)###TICKERS_END###", stdout, re.DOTALL)
        if t_match:
            content = t_match.group(1).strip()
            try:
                # 优先尝试标准 JSON (双引号)
                tickers = json.loads(content)
            except json.JSONDecodeError:
                # 【关键修复】如果 JSON 失败（因为单引号），尝试使用 ast 解析 Python 列表
                try:
                    logger.warning(f"JSON decode failed for '{content}', trying AST eval...")
                    tickers = ast.literal_eval(content)
                except Exception as e:
                    logger.error(f"AST eval failed: {e}")
        else:
            logger.warning(f"No tickers tag found. Output: {stdout[:200]}")

    except Exception as e:
        logger.error(f"Planner Execution Failed: {e}")
    finally:
        ci_client.stop()

    # 再次去重，防止代码没写好
    if tickers:
        tickers = list(set(tickers))
        
    print(f"    (Targets): {tickers}")
    return {"ticker_list": tickers, "reports": []}

def aggregator_node(state: OverallState):
    """Aggregator: 汇总报告"""
    reports = state.get('reports', [])
    print(f"--- [Aggregator] Reports count: {len(reports)} ---")
    
    if not reports:
        return {"final_advice": "⚠️ 未能生成任何有效报告。请检查 Tushare Token 或 Planner 筛选逻辑。", "reports": []}
    
    valid_reports = [r for r in reports if r.get('score_data') and r['score_data'].get('score') is not None]
    
    if not valid_reports:
        return {"final_advice": "⚠️ Worker 执行完成，但未能提取到有效评分数据。", "reports": reports}

    try:
        best_stock = max(valid_reports, key=lambda x: float(x['score_data'].get('score', -999)))
        advice = f"""
### ✅ 分析完成
共筛选并分析了 **{len(valid_reports)}** 只股票。

🥇 **综合推荐**: **{best_stock['ticker']}**
- **评分**: {best_stock['score_data'].get('score')}
- **动量**: {best_stock['score_data'].get('momentum')}
- **波动率**: {best_stock['score_data'].get('volatility')}

详情请查看下方图表。
        """
        return {"final_advice": advice}
    except Exception as e:
        return {"final_advice": f"汇总数据时发生错误: {e}", "reports": reports}

async def worker_wrapper(state: WorkerState):
    """
    Wrapper to run the worker subgraph and EXTRACT the final report.
    [Fix]: Extracts image even if LLM embeds it INSIDE the JSON object.
    """
    ticker = state["ticker"]
    
    # 1. Run the subgraph
    final_state = await stock_analyst_app.ainvoke(
        state, 
        config={
            "tags": [f"ticker:{ticker}"], 
            "recursion_limit": 10
        }
    )
    
    # 2. Extract Artifacts
    messages = final_state["messages"]
    score_data = {}
    img_b64 = ""
    final_text = ""
    
    for msg in reversed(messages):
        # Extract JSON (Score Data)
        if isinstance(msg, ToolMessage):
            # 尝试提取 JSON
            jm = re.search(r"###JSON_DATA_START###(.*?)###JSON_DATA_END###", msg.content, re.DOTALL)
            if jm:
                raw_content = jm.group(1).strip()
                # 清洗 numpy 格式
                clean_content = re.sub(r"np\.[a-zA-Z0-9_]+\((.*?)\)", r"\1", raw_content)
                
                try:
                    score_data = json.loads(clean_content)
                except json.JSONDecodeError:
                    try:
                        score_data = ast.literal_eval(clean_content)
                    except:
                        pass
                
                # 【关键修复】: 检查 JSON 内部是否包含图片数据
                if score_data:
                    # LLM 常用的字段名
                    possible_img_keys = ["plot_base64", "image_base64", "img_base64", "chart"]
                    for key in possible_img_keys:
                        if key in score_data and score_data[key]:
                            # 提取出图片，并从 score_data 中移除（以免 JSON 太大）
                            img_b64 = score_data.pop(key)
                            break

            # 尝试提取独立的 Image 标签 (Fallback)
            if not img_b64:
                im = re.search(r"###IMAGE_BASE64_START###(.*?)###IMAGE_BASE64_END###", msg.content, re.DOTALL)
                if im: img_b64 = im.group(1)

        # Extract Final Text Report
        if isinstance(msg, AIMessage) and not final_text:
            if msg.content and not msg.tool_calls:
                final_text = msg.content

    # 3. Construct Report
    report_item = {
        "ticker": ticker,
        "score_data": score_data,
        "image_base64": img_b64, # 现在这里会有值了
        "report": final_text or "Analysis completed."
    }
    
    return {"reports": [report_item]}


# 构建主图
main_workflow = StateGraph(OverallState)
main_workflow.add_node("planner", planner_node)
main_workflow.add_node("worker", worker_wrapper)
main_workflow.add_node("aggregator", aggregator_node)

main_workflow.add_edge(START, "planner")
main_workflow.add_conditional_edges(
    "planner",
    lambda state: [
        Send("worker", {"ticker": t, "messages": [HumanMessage(content="Start")]}) 
        for t in state["ticker_list"]
    ],
    ["worker"]
)
main_workflow.add_edge("worker", "aggregator")
main_workflow.add_edge("aggregator", END)

app_graph = main_workflow.compile()

# ==========================================
# 5. Event Generator (SSE Stream)
# ==========================================
async def event_generator(query: str):
    inputs = {"query": query}
    logger.info(f"🚀 Stream start: {query}")

    # 用于临时存储最终结果
    captured_final_output = {}

    try:
        async for event in app_graph.astream_events(inputs, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            node_name = event.get("name", "")
            
            # --- 1. 捕获最终输出 (关键修复) ---
            # 监听 LangGraph 的最终结束事件，或者 Aggregator 节点的结束事件
            if kind == "on_chain_end" and node_name == "LangGraph":
                # 这是整个图的最终输出
                output = event["data"].get("output")
                if output and "final_advice" in output:
                    captured_final_output = output
            
            # 备选：监听 Aggregator 的结束
            if kind == "on_chain_end" and node_name == "aggregator":
                output = event["data"].get("output")
                if output:
                    # 更新 capture，防止 LangGraph 根节点没捕获到
                    captured_final_output.update(output)

            # --- 2. 常规流式事件处理 ---
            
            # 提取 Ticker
            current_ticker = "System"
            for tag in tags:
                if tag.startswith("ticker:"):
                    current_ticker = tag.split(":")[1]
                    break
            
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield json.dumps({"type": "llm_stream", "ticker": current_ticker, "chunk": content}) + "\n"

            elif kind == "on_chat_model_end":
                output = event["data"].get("output")
                if output and hasattr(output, 'tool_calls') and output.tool_calls:
                     yield json.dumps({"type": "code_complete", "ticker": current_ticker}) + "\n"

            elif kind == "on_tool_start" and "code" in event["name"]:
                inp = event["data"].get("input", {})
                code = inp.get("code") if isinstance(inp, dict) else str(inp)
                yield json.dumps({"type": "code_input", "ticker": current_ticker, "code": code}) + "\n"

            elif kind == "on_tool_end":
                output = str(event["data"].get("output"))
                yield json.dumps({"type": "tool_output", "ticker": current_ticker, "output": output}) + "\n"

            elif kind == "on_chain_start":
                if node_name in ["planner", "worker", "aggregator"]:
                    yield json.dumps({"type": "node_status", "node": node_name, "ticker": current_ticker}) + "\n"

        # === 循环结束 ===
        logger.info("🏁 Stream finished. Processing final results...")
        
        # ❌ 删除这一行，它会导致重跑： final_state = await app_graph.ainvoke(inputs)
        
        # 直接使用我们在流中捕获的数据
        final_advice = captured_final_output.get("final_advice", "⚠️ 未能获取最终建议")
        reports_raw = captured_final_output.get("reports", [])
        
        # 数据后处理 (提取 Artifacts)
        clean_details = []
        for worker_report in reports_raw:
            # 现在 reports 里的结构已经是 worker_wrapper 处理过的了：
            # {'ticker': '...', 'score_data': {...}, 'image_base64': '...', 'report': '...'}
            if isinstance(worker_report, dict) and "score_data" in worker_report:
                 clean_details.append(worker_report)
            
            # 兼容：如果由于某种原因还是原始 state 格式 (防御性编程)
            elif isinstance(worker_report, dict) and "messages" in worker_report:
                # (这里保留之前的解析逻辑，防止旧数据结构遗留，但在新 worker_wrapper 下应该走不到这里)
                pass 

        yield json.dumps({
            "type": "final_result",
            "advice": final_advice,
            "details": clean_details
        }) + "\n"

    except Exception as e:
        logger.error(f"Stream Error: {e}")
        traceback.print_exc()
        yield json.dumps({"type": "final_result", "advice": f"❌ Error: {str(e)}", "details": []}) + "\n"

@app.post("/analyze_stream")
async def analyze_stream_endpoint(req: QueryRequest):
    return StreamingResponse(event_generator(req.query), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)