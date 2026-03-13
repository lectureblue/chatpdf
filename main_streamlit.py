# 배포시---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic import hub
from dotenv import load_dotenv
from io import StringIO
import os
import streamlit as st 
import tempfile
# load_dotenv()

#OpenAI 키 입력
api_key = st.text_input('OpenAI_API_Key',type='password')

#제목
st.title("_ChatPDF_ 📑")
st.write("---")

#파일 업로드 
uploaded_file = st.file_uploader("PDF파일을 올려주세요",type=['pdf'])
st.write("---")
#업로드한 파일 불러오기 
def pdf_to_document(uploaded_file):
   temp_dir = tempfile.TemporaryDirectory() #임시폴더 생성
   temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
   with open(temp_filepath, "wb") as f:
       f.write(uploaded_file.getvalue())
   loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된pdf로딩
   pages = loader.load_and_split()
   return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
  
    #Splitter
    text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size=100,
      chunk_overlap=20,
      length_function=len,
      is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)
    # print(texts[0])
    # print(texts[1])

    #Embedding
    embeddings_model = OpenAIEmbeddings(api_key=api_key,model="text-embedding-3-large")

    #Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)
    
    #배포시---
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    #----
    
    #스트리밍 처리할 Handler 생성
    class StreamHandler(BaseCallbackHandler):
      def __init__(self, container, initial_text=""):
          self.container = container
          self.text=initial_text
      def on_llm_new_token(self, token: str, **kwargs) -> None:
          self.text+=token
          self.container.markdown(self.text)
    # User Input

    st.header('PDF에게 질문해보세요!!!')
    question = st.text_input('질문을 입력하세요')
    
    if st.button('질문하기'):
        with st.spinner("Wait for it..."):
        
            llm = ChatOpenAI(temperature=0,api_key=api_key)
            #Chroma 백터 저장소에 대한 Retriever 인스턴스 생성
            retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), llm=llm
            )
            #Prompt Template
            prompt = hub.pull('rlm/rag-prompt')

            #Generate
            chat_box = st.empty() # 출력공간을 생성
            stream_handler = StreamHandler(chat_box)
            generate_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, openai_api_key=api_key, streaming=True, callbacks=[stream_handler])
            def format_docs(docs):
                return '\n\n'.join(doc.page_content for doc in docs)

            #사용자 질문에 대한 연관정보 가져온다.
            # docs = retriever_from_llm.invoke(question)
            # print(len(docs))#검색기의 실행 결과인 docs의 개수
            # print(docs)

            rag_chain = (
                {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}
                | prompt
                | generate_llm
                | StrOutputParser()
            )

            #Question
            result = rag_chain.invoke(question)
            # st.write(result)
