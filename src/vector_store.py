import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = str(ROOT / "vector_db")
COLLECTION_NAME = "local_pdf_knowledge"
BATCH_SIZE = 10

def get_vector_store(embeddings):
    # 离线页面不加载 Chroma 原生运行时。
    from langchain_chroma import Chroma
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                  persist_directory=PERSIST_DIRECTORY)

def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    return splitter.split_documents(docs)

def _chunk_id(file_hash, index, text):
    return hashlib.sha256(f"{file_hash}_{index}_{text}".encode("utf-8")).hexdigest()

def is_file_exists(store, file_hash):
    return bool(store.get(where={"file_hash": file_hash}, limit=1).get("ids"))

def add_documents_to_vector_store(documents, embeddings):
    store = get_vector_store(embeddings)
    groups = {}
    for doc in documents:
        digest = doc.metadata.get("file_hash")
        if not digest:
            raise ValueError("Document 缺少 file_hash，请检查 PDF 解析模块")
        groups.setdefault(digest, []).append(doc)

    added, skipped, count = [], [], 0
    for digest, pages in groups.items():
        name = pages[0].metadata.get("source", "未知文件")
        if is_file_exists(store, digest):
            skipped.append(name)
            continue
        chunks = split_documents(pages)
        if not chunks:
            raise ValueError(f"{name} 没有可入库的文本块")
        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            ids.append(_chunk_id(digest, i, chunk.page_content))
        # 全部批次成功才算完成。失败时回滚本文件，避免重试误判为重复文件。
        try:
            for start in range(0, len(chunks), BATCH_SIZE):
                store.add_documents(documents=chunks[start:start+BATCH_SIZE],
                                    ids=ids[start:start+BATCH_SIZE])
        except Exception as exc:
            store.delete(where={"file_hash": digest})
            raise
        added.append(name)
        count += len(chunks)
    return {"added_files": added, "skipped_files": skipped, "added_chunks": count}

def list_indexed_files(embeddings):
    result = get_vector_store(embeddings).get(include=["metadatas"])
    files = {}
    for meta in result.get("metadatas") or []:
        if meta and meta.get("file_hash"):
            files[meta["file_hash"]] = meta.get("source", "未知文件")
    return files

def delete_pdf(file_hash, embeddings):
    store = get_vector_store(embeddings)
    if not is_file_exists(store, file_hash):
        return False
    store.delete(where={"file_hash": file_hash})
    return True

