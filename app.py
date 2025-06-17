import streamlit as st
import PyPDF2
from typing import List
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="PDF QA System", layout="wide")

class PDFQASystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = OpenAIEmbeddings()
        # Update memory configuration with output_key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'  # Specify the output key
        )

    def extract_text_from_pdf(self, pdf_file) -> str:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def create_vectorstore(self, text_chunks) -> FAISS:
        vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=self.embeddings
        )
        return vectorstore

    def create_conversation_chain(self, vectorstore) -> ConversationalRetrievalChain:
        llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': self.get_custom_prompt()},
            chain_type="stuff"  # Use 'stuff' chain type for simpler processing
        )
        return chain

    def get_custom_prompt(self):
        from langchain.prompts import PromptTemplate
        
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        
        Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

def initialize_session_state():
    session_vars = [
        'pdf_qa_system',
        'conversation_chain',
        'chat_history',
        'processed_pdfs'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(user_question: str):
    if st.session_state.conversation_chain is None:
        st.error("Please upload a PDF file first!")
        return

    try:
        response = st.session_state.conversation_chain({
            "question": user_question
        })
        
        # Add messages to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

        # Display source documents
        with st.expander("View Source Documents"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)
                st.write("---")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    # Initialize session state
    initialize_session_state()

    # Create sidebar
    with st.sidebar:
        st.title("PDF QA System")
        # File upload
        pdf_docs = st.file_uploader(
            "Upload your PDF Documents",
            type="pdf",
            accept_multiple_files=True
        )
        
        if pdf_docs:
            if st.session_state.pdf_qa_system is None:
                st.session_state.pdf_qa_system = PDFQASystem()
            
            # Process PDFs button
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    try:
                        # Extract text from PDFs
                        raw_text = ""
                        for pdf in pdf_docs:
                            raw_text += st.session_state.pdf_qa_system.extract_text_from_pdf(pdf)
                        
                        # Split text into chunks
                        text_chunks = st.session_state.pdf_qa_system.text_splitter.split_text(raw_text)
                        
                        # Create vectorstore
                        vectorstore = st.session_state.pdf_qa_system.create_vectorstore(text_chunks)
                        
                        # Create conversation chain
                        st.session_state.conversation_chain = (
                            st.session_state.pdf_qa_system.create_conversation_chain(vectorstore)
                        )
                        
                        st.session_state.processed_pdfs = pdf_docs
                        st.success("PDFs processed successfully!")
                    
                    except Exception as e:
                        st.error(f"An error occurred while processing PDFs: {str(e)}")

    # Main chat interface
    st.title("Chat with your PDFs 📚")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if user_question := st.chat_input("Ask a question about your documents:"):
        handle_user_input(user_question)

if __name__ == "__main__":
    main()