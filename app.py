import streamlit as st
import pinecone
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_template import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Pinecone

FREE_RUN = load_dotenv("FREE_RUN")
PINECONE_API_KEY = load_dotenv("PINECONE_API_KEY")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # split text into chunks 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, # to avoid cutting off words
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# def get_vector_store(text_chunks):
#     # create vector store
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") if FREE_RUN else OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
def get_vector_store(text_chunks):
    # create vector store
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl") if FREE_RUN else OpenAIEmbeddings()

    # Use Pinecone as the vector store
    pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')
    index_name = "initial-index"  # replace with your desired index name
    index = pinecone.Index(index_name)
    
    vectorstore = Pinecone(
    index, embeddings.embed_query, text_chunks
)
    return vectorstore


def get_conversation_chain(vectorstore):
    # create conversation chain
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
                        "temperature": 0.5, "max_length": 512}) if FREE_RUN else ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="WhisperChain 🔗", page_icon=":link:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("WhisperChain")
    user_question = st.text_input("Ask a question about your documents.")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):

                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vector_store = get_vector_store(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
            else:
                st.error("Please upload at least one PDF")


if __name__ == '__main__':
    main()