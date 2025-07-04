import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import shutil


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    return texts

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, make a natural conversation and ask user if he/she needs any details from the pdf and reply should be much shorter.Don't make it up.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",  # or llama3-70b-8192 if your quota allows
        temperature=0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if not st.session_state.get("pdf_uploaded"):
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,k=13)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]

    # Update chat history
    st.session_state.chat_history.append({"question": user_question, "answer": answer})

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")
    user_question = st.chat_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")

        for chat in st.session_state.chat_history:
            # User bubble
            with st.chat_message("user"):
                st.markdown(chat['question'])

            # AI bubble
            with st.chat_message("assistant"):
                st.markdown(chat['answer'])
    with st.sidebar:
        st.title("📁 Upload & Controls")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.session_state.pdf_uploaded = True
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Now ask your question.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        if st.button("Clear Chat(Double Click)"):
            user_question = ""
            st.session_state.chat_history = []
            st.session_state.pdf_uploaded = False
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.success("🧹 Chat history and PDF data got cleared. Pls re-upload PDF")
        output = "\n\n".join([
            f"You: {c['question']}\nAI: {c['answer']}" for c in st.session_state.chat_history
        ])
        st.download_button("⬇️Download Chat History", output, file_name="chat_history.txt")

if __name__ == "__main__":
    main()
