import hashlib
import tempfile
from pathlib import Path

def parse_uploaded_pdfs(uploaded_files):
    all_docs, names = [], []
    if not uploaded_files:
        return all_docs, names
    with tempfile.TemporaryDirectory() as tmp:
        for uploaded in uploaded_files:
            data = uploaded.getvalue()
            if not data:
                continue
            digest = hashlib.sha256(data).hexdigest()
            name = Path(uploaded.name).name
            # 临时文件用哈希命名，避免不同上传文件同名覆盖。
            path = Path(tmp) / (digest + Path(name).suffix)
            path.write_bytes(data)
            if Path(name).suffix.lower() != ".pdf":
                raise ValueError("仅支持 PDF 文件")
            from langchain_community.document_loaders import PyPDFLoader
            docs = PyPDFLoader(str(path)).load()
            if not any(doc.page_content.strip() for doc in docs):
                raise ValueError(f"{name} 没有可提取文本；扫描 PDF 需要先做 OCR")
            for doc in docs:
                doc.metadata["source"] = name
                doc.metadata["file_hash"] = digest
                if "page" in doc.metadata:
                    doc.metadata["page"] = int(doc.metadata["page"])
            all_docs.extend(docs)
            names.append(name)
    return all_docs, names
