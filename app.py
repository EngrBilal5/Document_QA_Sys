import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import time

# Set up Streamlit UI
st.title("Document-Based Question Answering")
st.write("Upload a PDF file and ask a question about it.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the uploaded PDF
    loader = PyPDFLoader("uploaded_file.pdf")
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Initialize the embedding model and FAISS
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.from_documents(chunks, embedding)
    
    # Set up HuggingFace LLM pipeline
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Load the QA chain
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    
    # Question input from user
    question = st.text_input("Ask a question:")
    
    if question:
        # Retrieve relevant documents and run the QA chain
        def ask_question_with_retry(question, retries=3, delay=5):
            for attempt in range(retries):
                try:
                    relevant_docs = faiss_index.similarity_search(question, k=3)  # Retrieve 3 most relevant docs
                    answer = qa_chain.run(input_documents=relevant_docs, question=question)
                    return answer
                except Exception as e:
                    print(f"Error: {str(e)}. Retrying... (Attempt {attempt + 1}/{retries})")
                    time.sleep(delay)
            return "Sorry, I couldn't get an answer."
        
        # Display the answer
        answer = ask_question_with_retry(question)
        st.write(f"Answer: {answer}")
