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
import time
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
TEMPERATURE = 0.3


class PDFProcessor:
    def __init__(self):
        self._load_api_key()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def _load_api_key(self):
        """Load and validate API key."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Gemini API key is missing. Please set GEMINI_API_KEY in your environment variables.")
        genai.configure(api_key=api_key)

    def extract_text(self, pdf_docs: List[object]) -> str:
        """Extract text from multiple PDF files."""
        combined_text = []
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text.append(text)
            except Exception as e:
                logger.error(f"Error processing PDF {pdf.name}: {str(e)}")
                raise ValueError(f"Error processing PDF {pdf.name}: {str(e)}")

        return "\n".join(combined_text)

    def create_text_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        return splitter.split_text(text)

    def create_vector_store(self, text_chunks: List[str]) -> None:
        """Create and save vector store."""
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local("faiss-index")

    def load_vector_store(self) -> FAISS:
        """Load the vector store from disk."""
        return FAISS.load_local("faiss-index", self.embeddings, allow_dangerous_deserialization=True)


class QASystem:
    def __init__(self):
        self.prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in the context,
        respond with "I apologize, but I cannot find the answer to your question in the provided documents."
        Please maintain a professional and helpful tone.

        Context: {context}

        Question: {question}

        Answer:
        """
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=TEMPERATURE)

    def get_qa_chain(self):
        """Create the question-answering chain."""
        prompt = PromptTemplate(template=self.prompt_template,
                                input_variables=["context", "question"])
        return load_qa_chain(self.model, chain_type="stuff", prompt=prompt)

    def generate_response(self, question: str, vector_store: FAISS) -> str:
        """Generate response for user question."""
        docs = vector_store.similarity_search(question)
        chain = self.get_qa_chain()

        try:
            response = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            return response["output_text"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise ValueError(f"Error generating response: {str(e)}")


class StreamlitApp:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.qa_system = QASystem()
        self.setup_streamlit()

    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="PDF Chat Assistant",
            page_icon="📚",
            layout="wide"
        )
        st.header("💬 Chat with Your PDF Documents")

    def validate_files(self, pdf_docs: List[object]) -> bool:
        """Validate uploaded PDF files."""
        if not pdf_docs:
            return False

        for pdf in pdf_docs:
            if pdf.size > MAX_FILE_SIZE:
                st.error(f"File {pdf.name} exceeds the 10MB size limit.")
                return False

            if not pdf.name.lower().endswith('.pdf'):
                st.error(f"File {pdf.name} is not a PDF.")
                return False

        return True

    def show_sidebar(self) -> Optional[List[object]]:
        """Display and handle sidebar components."""
        with st.sidebar:
            st.title("📁 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDF files (10MB max per file)",
                accept_multiple_files=True,
                type="pdf"
            )

            if pdf_docs and self.validate_files(pdf_docs):
                st.write(f"📎 {len(pdf_docs)} file(s) uploaded successfully")

                if st.button("🔍 Process Documents", type="primary"):
                    return pdf_docs

            st.markdown("---")
            st.markdown("### 📋 Instructions")
            st.markdown("""
            1. Upload one or more PDF files
            2. Click 'Process Documents'
            3. Ask questions about your documents
            """)

        return None

    def process_pdfs(self, pdf_docs: List[object]) -> None:
        """Process uploaded PDF files."""
        with st.spinner("Processing documents..."):
            try:
                # Extract text from PDFs
                raw_text = self.pdf_processor.extract_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDFs.")
                    return

                # Create text chunks and vector store
                text_chunks = self.pdf_processor.create_text_chunks(raw_text)
                self.pdf_processor.create_vector_store(text_chunks)

                st.success("✅ Documents processed successfully!")
                st.session_state.docs_processed = True

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.session_state.docs_processed = False

    def handle_user_input(self, user_question: str) -> None:
        """Process user questions and display responses."""
        if not hasattr(st.session_state, 'docs_processed') or not st.session_state.docs_processed:
            st.warning("Please upload and process documents first.")
            return

        try:
            vector_store = self.pdf_processor.load_vector_store()
            with st.spinner("Thinking..."):
                response = self.qa_system.generate_response(user_question, vector_store)

            st.write("🤖 Assistant:", response)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    def run(self):
        """Run the Streamlit application."""
        # Initialize session state
        if 'docs_processed' not in st.session_state:
            st.session_state.docs_processed = False

        # Handle file upload in sidebar
        pdf_docs = self.show_sidebar()
        if pdf_docs:
            self.process_pdfs(pdf_docs)

        # Main chat interface
        user_question = st.text_area("🤔 Ask a question about your documents:", height=100)
        if st.button("Send", type="primary"):
            if user_question:
                self.handle_user_input(user_question)
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")