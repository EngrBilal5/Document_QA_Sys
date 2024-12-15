from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import streamlit as st
import time

# Set up Streamlit UI
st.set_page_config(
    page_title="Document QA System",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar configuration
with st.sidebar:
    st.title(":books: Document QA System")
    st.write("Upload a document (PDF or text), process it, and ask questions!")
    st.info(
        "✨ **Tip:** Use this app for quick insights from long documents."
    )
    st.markdown("---")

# Title and description
st.title(":mag_right: Document-Based Question Answering")
st.markdown(
    """
    **Upload a PDF or text file** and **ask questions** about its content. This app uses advanced language models to provide accurate answers.
    """
)

# File upload
uploaded_file = st.file_uploader("Choose a file to upload:", type=["pdf", "txt"])

if uploaded_file is not None:
    # Display file details
    st.success(f"**Uploaded:** {uploaded_file.name}")

    # Save uploaded file temporarily
    file_path = f"uploaded_file.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(10)

    # Load the uploaded document
    st.info("📚 Loading the document...")
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")  # Specify UTF-8 encoding for text files
    else:
        st.error("Unsupported file format. Please upload a PDF or text file.")
        st.stop()

    try:
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        st.stop()

    progress_bar.progress(30)

    # Split the document into chunks
    st.info("✂️ Splitting the document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    progress_bar.progress(60)

    # Initialize the embedding model and FAISS
    st.info("🔍 Creating embeddings for efficient search...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_index = FAISS.from_documents(chunks, embedding)

    progress_bar.progress(80)

    # Set up HuggingFace LLM pipeline
    st.info("🤖 Initializing the question-answering model...")
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Load the QA chain
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")

    progress_bar.progress(100)
    st.success("🎉 Document processed and ready for questions!")

    # Question input from user
    st.markdown("---")
    question = st.text_input("**Ask a question:**")

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

        with st.spinner("🔎 Finding the best answer..."):
            answer = ask_question_with_retry(question)

        # Display the answer
        st.markdown(f"### :speech_balloon: Answer:")
        st.success(answer)

        # Optionally display relevant chunks (debugging or additional info)
        if st.checkbox("Show relevant document chunks"):
            relevant_docs = faiss_index.similarity_search(question, k=3)
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Chunk {i + 1}:** {doc.page_content}")

else:
    st.warning("⚠️ Please upload a PDF or text file to get started.")
