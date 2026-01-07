import streamlit as st
import pandas as pd
import base64
import requests # 新增

# 配置后端地址
BACKEND_URL = "http://127.0.0.1:8000/api/v1/analyze"

st.set_page_config(page_title="选股 Agent", layout="wide")

st.title("⚡ Stock Analysis Agent")
#st.markdown(f"连接后端 API: `{BACKEND_URL}`")

with st.sidebar:
    top_n = st.number_input("Top N股票量化分析", 1, 5, 2)
    run_btn = st.button("开始分析", type="primary")

if run_btn:
    # 状态展示
    status_text = st.empty()
    status_text.info("正在请求后端 API，请稍候...")
    
    try:
        # 发送 HTTP POST 请求
        payload = {"top_n": top_n}
        response = requests.post(BACKEND_URL, json=payload, timeout=300) # 设置较长超时，因为分析耗时
        
        if response.status_code == 200:
            resp_json = response.json()
            
            if resp_json["status"] == "success":
                status_text.success("分析成功！")
                data_list = resp_json["data"]
                
                # 表格展示
                df = pd.DataFrame(data_list).drop(columns=["image"], errors="ignore")
                st.table(df)
                
                # 图片展示
                cols = st.columns(len(data_list))
                for idx, item in enumerate(data_list):
                    with cols[idx]:
                        st.markdown(f"### {item['id']}")
                        st.caption(f"Score: {item['score']:.4f}")
                        if "image" in item and item["image"]:
                            try:
                                img_bytes = base64.b64decode(item['image'])
                                st.image(img_bytes, use_container_width=True)
                            except:
                                st.error("图片解码失败")
            else:
                status_text.error(f"后端返回业务错误: {resp_json.get('error')}")
        else:
            status_text.error(f"HTTP 请求失败: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        status_text.error("无法连接到后端服务器，请确保 backend.py 正在运行。")
    except Exception as e:
        status_text.error(f"发生未知错误: {str(e)}")