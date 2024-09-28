import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables and configure Google Gemini API
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Google Gemini API key is missing. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)


def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split the text into smaller chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create a vector store for efficient similarity search."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")


def get_conversational_chain():
    """Create a QA chain using Google Gemini model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompts = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompts)
    return chain


def user_input(user_question):
    """Handle user input and generate a response from the model."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Model: ", response["output_text"])
    except Exception as e:
        st.error(f"Error generating response from model: {str(e)}")


def main():
    """Main function for the Streamlit app."""
    st.set_page_config("ChatPDF")
    st.header("Chat with PDFs")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")

        if pdf_docs:
            if any(pdf.size > 10 * 1024 * 1024 for pdf in pdf_docs):
                st.error("One or more files exceed the 10 MB size limit.")
            else:
                st.write(f"Uploaded {len(pdf_docs)} PDF file(s).")

            if st.button("Submit & Process"):
                with st.spinner(f"Processing {len(pdf_docs)} PDF file(s)..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("The uploaded PDFs contain no readable text.")
                            st.stop()

                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success(f"Processing complete! {len(pdf_docs)} file(s) processed successfully.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
        else:
            st.warning("Only PDF files are supported!", icon="⚠️")


if __name__ == "__main__":
    main()
