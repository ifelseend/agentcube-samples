import os
import re
import time
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage

# 导入业务逻辑
from agent_core import graph_app, HTML_START_TAG, HTML_END_TAG

# ==========================================
# 1. 基础配置 (日志 & 目录)
# ==========================================
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 本地游戏文件存储目录
OUTPUT_DIR = "./generated_games"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. FastAPI 应用初始化
# ==========================================
app = FastAPI(title="Agent Game Studio API")

# 允许跨域 (CORS) - 方便前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. 数据模型
# ==========================================
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class FileData(BaseModel):
    filename: str
    content: str
    media_type: str = "text/html"
    local_path: str

class ChatResponse(BaseModel):
    text_response: str
    generated_file: FileData | None = None

# ==========================================
# 4. API 接口
# ==========================================

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok", 
        "service": "Agent Game Studio",
        "timestamp": time.time()
    }

@app.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"📥 收到请求: {request.message} (Thread: {request.thread_id})")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # --- 1. 执行 Agent (耗时操作) ---
        # invoke 会运行直到图执行完毕 (__end__)
        final_state = graph_app.invoke(inputs, config=config)
    except Exception as e:
        logger.error(f"❌ Agent 执行崩溃: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # --- 2. 获取文本回复 ---
    messages = final_state["messages"]
    last_message = messages[-1].content
    file_data = None
    
   # --- 3. 解析协议 & 本地落盘 ---
    # 倒序遍历消息
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            
            # 🔍 调试日志
            logger.info(f"🔍 Inspecting Tool Output (Len: {len(content)})")

            html_content = ""
            extraction_method = "none"

            # -------------------------------------------------------
            # 策略 A: 优先尝试标准协议 (Tags)
            # -------------------------------------------------------
            pattern = re.escape(HTML_START_TAG) + r"(.*?)" + re.escape(HTML_END_TAG)
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 50: # 稍微严格一点的长度检查
                    html_content = extracted
                    extraction_method = "protocol_tags"
                else:
                    logger.warning(f"⚠️ 捕获到协议标签，但内容过短 ({len(extracted)} chars)。尝试兜底策略...")

            # -------------------------------------------------------
            # 策略 B: 兜底策略 - 直接正则匹配 HTML 结构 (Smart Fallback)
            # -------------------------------------------------------
            if not html_content:
                # 匹配以 <!DOCTYPE html> 开头，以 </html> 结尾的内容
                # re.IGNORECASE 忽略大小写
                fallback_match = re.search(r'(<!DOCTYPE html>.*?</html>)', content, re.DOTALL | re.IGNORECASE)
                if fallback_match:
                    html_content = fallback_match.group(1).strip()
                    extraction_method = "fallback_regex"
                    logger.info("✅ 启用兜底策略：直接从输出中提取到了 HTML 代码。")

            # -------------------------------------------------------
            # 4. 执行落盘
            # -------------------------------------------------------
            if html_content and len(html_content) > 50:
                # A. 生成文件名
                file_id = f"game_{int(time.time())}_{str(uuid.uuid4())[:4]}"
                filename = f"{file_id}.html"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # B. 写入文件
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    
                    logger.info(f"💾 文件落盘成功 ({extraction_method}): {filepath}")
                    
                    file_data = FileData(
                        filename=filename,
                        content=html_content,
                        local_path=filepath
                    )
                    break # 成功找到并保存，跳出循环
                except Exception as io_err:
                    logger.error(f"❌ 文件写入失败: {io_err}")
            else:
                if extraction_method == "none":
                    logger.warning("⚠️ 未检测到有效 HTML 代码 (Tags 或 Raw HTML 均未找到)")

    return ChatResponse(
        text_response=last_message,
        generated_file=file_data
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 服务启动中... 存储目录: {os.path.abspath(OUTPUT_DIR)}")
    uvicorn.run(app, host="0.0.0.0", port=8080)