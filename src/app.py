"""最小界面：PDF 管理和多轮问答；聊天只保存在当前浏览器会话。"""
import streamlit as st
from src.config import get_clients
from src.pdf_service import parse_uploaded_pdfs
from src.vector_store import add_documents_to_vector_store, list_indexed_files, delete_pdf
from src.rag_qa import answer_question

st.set_page_config(page_title='本地 PDF 知识库问答',page_icon='📚')
st.title('📚 本地 PDF 知识库问答系统')

@st.cache_resource
def clients():
    # 打开页面不创建客户端；只有入库或提问时才读取 API 配置。
    return get_clients()

for key,value in {'messages':[], 'parsed_documents':[]}.items():
    if key not in st.session_state:st.session_state[key]=value

with st.sidebar:
    st.header('PDF 管理')
    uploads=st.file_uploader('选择 PDF',type=['pdf'],accept_multiple_files=True)
    if st.button('解析 PDF',disabled=not uploads):
        st.session_state.parsed_documents=[]
        try:
            docs,names=parse_uploaded_pdfs(uploads)
            st.session_state.parsed_documents=docs
            st.success(f'{len(names)} 个文件，共 {len(docs)} 页')
        except Exception as exc:st.error(f'解析失败：{exc}')
    if st.button('加入知识库',disabled=not st.session_state.parsed_documents):
        try:
            _,emb=clients()
            with st.spinner('正在向量化并入库……'):
                result=add_documents_to_vector_store(st.session_state.parsed_documents,emb)
            st.success(f"新增 {result['added_chunks']} 个文本块")
            if result['skipped_files']:st.info('重复文件已跳过：'+'、'.join(result['skipped_files']))
        except Exception as exc:st.error(f'入库失败：{exc}')
    # 列出、删除资料不需要 Embedding 请求，也不需要 API 密钥。
    try:
        indexed=list_indexed_files(None)
        if indexed:
            selected=st.selectbox('已入库文件',list(indexed),format_func=lambda h:indexed[h])
            if st.button('删除选中文件'):
                delete_pdf(selected,None);st.rerun()
        else:st.info('知识库为空')
    except Exception as exc:st.warning(f'读取知识库失败：{exc}')
    if st.button('清空聊天'):
        st.session_state.messages=[];st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('参考资料'):
                for source in message['sources']:
                    st.markdown(f"**[资料{source['index']}] {source['filename']} · 第 {source['page']} 页**")
                    st.write(source['content'])
        if message.get('rewritten_question'):
            with st.expander('改写后的检索问题'):st.write(message['rewritten_question'])

question=st.chat_input('请输入问题')
if question:
    # 先保存旧历史，再添加当前问题，避免问题在 Prompt 中重复出现。
    history=[m for m in st.session_state.messages if not m.get('error')]
    st.session_state.messages.append({'role':'user','content':question})
    try:
        llm,emb=clients()
        with st.spinner('正在检索并回答……'):
            result=answer_question(question,history,llm,emb)
        st.session_state.messages.append(dict(role='assistant',content=result['answer'],
            sources=result['sources'],rewritten_question=result['rewritten_question']))
    except Exception as exc:
        st.session_state.messages.append({'role':'assistant','content':f'问答失败：{exc}','error':True})
    st.rerun()
