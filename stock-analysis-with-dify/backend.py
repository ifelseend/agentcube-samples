import os
import json
import base64
import logging
import io
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import operator # 1. 必须导入这个标准库
from typing import Annotated, List, TypedDict, Any,Optional
# --- 修复 Import ---
from langgraph.types import Send 
from langgraph.graph import StateGraph, END, START
from agentcube import CodeInterpreterClient

# 强制不使用任何系统代理
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ["no_proxy"] = "*"

# --- 初始化 API ---
app = FastAPI(title="Stock Analysis Agent API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend-api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- 1. 配置与凭证 ---
OBS_CONFIG = {
    "ak": os.getenv("OBS_ACCESS_KEY_ID"),
    "sk": os.getenv("OBS_SECRET_ACCESS_KEY"),
    "endpoint": os.getenv("OBS_ENDPOINT"),
    "bucket": os.getenv("BUCKET_NAME"),
    "object_key": os.getenv("OBJECT_KEY")
}

# --- 2. 状态定义 ---

# 单只股票的数据结构
class StockData(TypedDict):
    filename: str
    csv_content: str  # 关键：CSV 的纯文本内容
    score: float      # 计算后的分数
    error: str

# 全局状态
class OverallState(TypedDict):
    top_n: int
    raw_stocks: List[StockData]   # 存放从 Step 1 获取的所有原始数据
    analyzed_results: Annotated[List[dict], operator.add]
    final_output: dict            # 包含图表和最终建议

# Map 阶段的输入状态
class AnalysisState(TypedDict):
    stock_data: StockData

# --- 3. 工具函数 ---

def run_sandbox_code(code: str, description: str) -> str:
    """通用沙箱执行器"""
    ci_client = CodeInterpreterClient(name="simple-codeinterpreter-python")
    try:
        logger.info(f"Running Sandbox Task: {description}")
        # 仅返回 stdout
        result = ci_client.run_code("python", code, timeout=120)
        return str(result)
    except Exception as e:
        logger.error(f"Sandbox Error ({description}): {e}")
        return json.dumps({"error": str(e)})
    finally:
        ci_client.stop()

# --- 4. 节点逻辑 ---

def fetch_and_extract_node(state: OverallState):
    """
    Step 1: 下载、解压、读取内容。
    注意：这里不保存文件，而是将文件内容读取为字符串返回。
    """
    # 注入 OBS 凭证
    code = f"""
import tarfile
import io
import json
import base64
# 模拟/使用 OBS SDK
from obs import ObsClient

ak = "{OBS_CONFIG['ak']}"
sk = "{OBS_CONFIG['sk']}"
server = "{OBS_CONFIG['endpoint']}"
bucket = "{OBS_CONFIG['bucket']}"
key = "{OBS_CONFIG['object_key']}"

try:
    # 1. 下载到内存 (BytesIO)
    client = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
    resp = client.getObject(bucket, key, loadStreamInMemory=True)
    
    if resp.status < 300:
        # 获取内存流
        content_stream = io.BytesIO(resp.body.buffer)
        
        extracted_data = []
        
        # 2. 内存解压
        with tarfile.open(fileobj=content_stream, mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.csv'):
                    f = tar.extractfile(member)
                    if f:
                        # 读取 CSV 文本内容
                        csv_text = f.read().decode('utf-8')
                        extracted_data.append({{
                            "filename": member.name,
                            "csv_content": csv_text
                        }})
        
        # 输出 JSON
        print(json.dumps(extracted_data))
    else:
        print(json.dumps({{"error": "Download failed"}}))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result_str = run_sandbox_code(code, "Fetch & Extract")
    try:
        data = json.loads(result_str)
        if isinstance(data, dict) and "error" in data:
            return {"raw_stocks": []}
        
        # 限制演示数量，防止大并发
        # 真实场景中，如果 CSV 很大，这里会占用大量内存，需要考虑分页处理
        return {"raw_stocks": data[:10]} 
    except Exception as e:
        logger.error(f"Parse error in fetch: {e}")
        return {"raw_stocks": []}

def map_analysis_node(state: AnalysisState):
    """
    Step 2: 独立的 CI 分析。
    策略：将 CSV 内容直接作为字符串嵌入代码。
    """
    stock = state["stock_data"]
    csv_content = stock["csv_content"]
    fname = stock["filename"]

    # 为了避免 CSV 内容中的引号破坏代码结构，使用原始字符串或 Base64 传递数据更安全
    # 这里演示 Base64 传递数据进沙箱
    b64_csv = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')

    code = f"""
import pandas as pd
import io
import base64
import json

try:
    # 1. 解码嵌入的数据
    csv_str = base64.b64decode("{b64_csv}").decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_str))
    
    # 2. 计算逻辑
    if len(df) < 2:
        score = 0
    else:
        closes = df['Close'].values
        momentum = (closes[-1] - closes[0]) / closes[0]
        volatility = df['Close'].pct_change().std()
        score = momentum / (volatility + 1e-9) if volatility else 0

    print(json.dumps({{"filename": "{fname}", "score": score}}))

except Exception as e:
    print(json.dumps({{"filename": "{fname}", "score": -999, "error": str(e)}}))
"""
    result_str = run_sandbox_code(code, f"Analyze {fname}")
    try:
        res = json.loads(result_str)
        return {"analyzed_results": [res]}
    except:
        return {"analyzed_results": []}

def reduce_and_plot_node(state: OverallState):
    """
    Step 3: 汇总 + Top N 绘图。
    这里需要再次启动一个 CI，并将 Top N 的 CSV 数据传进去绘图。
    """
    raw_data_map = {item['filename']: item['csv_content'] for item in state['raw_stocks']}
    results = state['analyzed_results']
    
    # 1. 排序取 Top N
    sorted_res = sorted(results, key=lambda x: x.get('score', -999), reverse=True)
    top_n = state['top_n']
    top_stocks = sorted_res[:top_n]
    
    # 2. 准备绘图所需的数据包 (只传输 Top N 的数据)
    plot_payload = []
    for stock in top_stocks:
        fname = stock['filename']
        if fname in raw_data_map:
            plot_payload.append({
                "id": fname,
                "score": stock['score'],
                "csv_content": raw_data_map[fname] # 再次取出内容
            })
    
    # 将数据包序列化为 Base64 以嵌入代码
    payload_json = json.dumps(plot_payload)
    payload_b64 = base64.b64encode(payload_json.encode('utf-8')).decode('utf-8')
    
    # 3. 生成绘图代码
    code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import io
import json
import base64

# 解包数据
input_data = json.loads(base64.b64decode("{payload_b64}").decode('utf-8'))
output_data = []

for item in input_data:
    try:
        csv_str = item['csv_content']
        df = pd.read_csv(io.StringIO(csv_str))
        
        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df['Close'], label='Close Price')
        plt.title(f"{{item['id']}} (Score: {{item['score']:.2f}})")
        plt.legend()
        plt.grid(True)
        
        # 内存绘图 -> Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        output_data.append({{
            "id": item['id'],
            "score": item['score'],
            "advice": "Strong Buy" if item['score'] > 0 else "Hold",
            "image": img_b64
        }})
    except Exception as e:
        output_data.append({{"id": item['id'], "error": str(e)}})

print(json.dumps(output_data))
"""
    result_str = run_sandbox_code(code, "Visualize Top N")
    try:
        final_data = json.loads(result_str)
        return {"final_output": {"data": final_data}}
    except Exception as e:
        return {"final_output": {"error": str(e)}}

# --- 5. 构建图 ---

def map_router(state: OverallState):
    # 为每个获取到的 CSV 原始数据启动一个 Map 任务
    return [Send("map_analysis", {"stock_data": s}) for s in state["raw_stocks"]]

workflow = StateGraph(OverallState)

workflow.add_node("fetch", fetch_and_extract_node)
workflow.add_node("map_analysis", map_analysis_node)
workflow.add_node("reduce_plot", reduce_and_plot_node)

workflow.add_edge(START, "fetch")
workflow.add_conditional_edges("fetch", map_router, ["map_analysis"])
workflow.add_edge("map_analysis", "reduce_plot")
workflow.add_edge("reduce_plot", END)

agent_app = workflow.compile()

# --- 6. 定义 API 接口模型 ---
class AnalysisRequest(BaseModel):
    top_n: int = 3

class AnalysisResponse(BaseModel):
    status: str
    data: Optional[List[dict]] = None
    error: Optional[str] = None

# --- 7. HTTP Endpoint ---
@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_stocks(request: AnalysisRequest):
    """
    触发 LangGraph 工作流
    """
    logger.info(f"Received request for Top {request.top_n} stocks")
    
    initial_state = {
        "top_n": request.top_n, 
        "raw_stocks": [], 
        "analyzed_results": [],
        "final_output": {}
    }
    
    try:
        # 这里的 invoke 是同步阻塞的，FastAPI 会在线程池中运行它
        # 如果是极高并发，建议将 agent 改写为 async 节点并使用 ainvoke
        result = agent_app.invoke(initial_state)
        
        final_output = result.get("final_output", {})
        
        if "error" in final_output:
            return {"status": "error", "error": final_output["error"]}
        
        return {
            "status": "success", 
            "data": final_output.get("data", [])
        }
        
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动服务器，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)