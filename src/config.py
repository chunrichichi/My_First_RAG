import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

# 只有用户显式开启在线模式并执行操作时才初始化客户端。
def get_clients():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    if not API_KEY:
        raise ValueError("未读取 DASHSCOPE_API_KEY，请检查项目根目录 .env")
    if not BASE_URL:
        raise ValueError("未读取 DASHSCOPE_BASE_URL，请检查项目根目录 .env")
    
    llm = ChatOpenAI(model=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0, timeout=30, max_retries=2)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL,
        check_embedding_ctx_length=False, chunk_size=10, request_timeout=30, max_retries=2,
    )
    return llm, embeddings
