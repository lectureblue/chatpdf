# 배포시 ---- (맨위 추가)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# 업로드한 파일 불러오기 
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from dotenv import load_dotenv
from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.callbacks.base import BaseCallbackHandler
import streamlit as st
import tempfile
import os

# 맨 처음에 위치해야 함
# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
api_key = st.text_input("OPENAI_API_KEY",type="password")

st.title("📝 _ChatPDF_")
st.write("---")

uploaded_file = st.file_uploader(
    "PDF 파일을 올려주세요", type="pdf"
)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() # 임시폴더 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된 pdf로딩
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)
  
  #텍스트 분할
  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100, #각 청크의 최대 길이
    chunk_overlap=20, #인접한 청크사이의 중복영역, 문장이 끊기는 문제 해결 20글자 겹침
    length_function=len,#청크길이 측정하는 함수
    is_separator_regex=False,#단순한 문자열로 해석
  )
  texts = text_splitter.split_documents(pages)
  # print(texts[0])
  # print(texts[1])

  #임베딩 객체 생성
  embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key= api_key)

  #벡터 DB 관리
  db = Chroma.from_documents(texts, embeddings_model)

# 캐싱 문제 발생시 아래 코드를 2줄을 주석해제 하여서 사용
  import chromadb
  chromadb.api.client.SharedSystemClient.clear_system_cache()

  class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)
  #user input
  st.header('PDF에게 질문해보세요!!!')
  question = st.text_input('질문을 입력하세요')

  if st.button('질문하기'):
     with st.spinner('Wait for it....'):
        #검색 객체 생성
        llm = init_chat_model("gpt-5.5", api_key=api_key)

        # Chroma 백터 저장소에 대한 Retriever 인스턴스 생성
        retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
        )

        #Generate 
        #hub에서 프롬프트 가져오기
        client = Client()
        prompt = client.pull_prompt('rlm/rag-prompt',dangerously_pull_public_prompt=True)

        # 출력공간 확보 stream 부분
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
            
        generate_llm = init_chat_model(model="gpt-5.5", openai_api_key=api_key, streaming=True, callbacks=[stream_handler])
        # generate_llm = init_chat_model(model="gpt-5.5", openai_api_key=api_key)

        #검색결과 format
        def format_docs(docs):
            return '\n\n'.join(doc.page_content for doc in docs)

        # 체인
        rag_chain = (
          {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}  #입력값 그대로 사용
          | prompt
          | generate_llm
          | StrOutputParser()
        )

        #실행
        result = rag_chain.invoke(question)
        # st.write(result)