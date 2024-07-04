import streamlit as st
import os
from pypdf import PdfReader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

contextualized_sys_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualized_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualized_sys_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1",
                     model_kwargs={'temperature': 0.1, "max_length": 500},
                     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                                  model_kwargs={"device": 'cpu'},
                                                  encode_kwargs={"normalize_embeddings": True})


sys_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_pdf_docs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(text_chunks, huggingface_embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}, search_type="similarity")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt)
    chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    return chunks      

def handle_user_input(user_question, conversational_rag_chain):
    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )
    return response["answer"]

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "abc123"

    st.header("Chat with PDFs")
    
    user_question = st.text_input("Ask a question about your pdf")

    if user_question and st.session_state.conversation:
        response = handle_user_input(user_question, st.session_state.conversation)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your documents"):
                raw_text = get_pdf_docs(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()




















