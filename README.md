<div align="center"> <img src="Logo.jpg" alt="AskPDF logo" width="300"/> </div>
<div align="center">AskPDF: Chat with PDFs Using Langchain and Gemini API</div>
<p align="center"> <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python"> <img src="https://img.shields.io/badge/Streamlit-1.x-red" alt="Streamlit"> <img src="https://img.shields.io/badge/Langchain-0.0.x-brightgreen" alt="Langchain"> <img src="https://img.shields.io/badge/FAISS-v1.7-orange" alt="FAISS"> </p> 
🚀 Overview
AskPDF is a PDF-driven chatbot that allows users to upload multiple PDFs, process their contents, and chat with the extracted information. This project leverages Langchain for building a question-answering system, Google Gemini API for embeddings and conversational AI, and FAISS for vector-based document retrieval.

🌟 Key Features
<ul> <li>📄 <strong>Upload and Process PDFs:</strong> Extract text from PDFs and create searchable chunks.</li> <li>🔍 <strong>Vector-Based Search:</strong> Utilize FAISS to perform efficient similarity searches across documents.</li> <li>🤖 <strong>Chat Interface:</strong> Ask questions about the PDFs through a user-friendly chatbot interface.</li> <li>⚡ <strong>Google Gemini API:</strong> Leverage advanced embeddings and conversation models from Google's API.</li> </ul>
📂 Tech Stack
<table> <tr> <td><strong>Streamlit</strong></td> <td>Interactive UI framework for Python</td> </tr> <tr> <td><strong>Langchain</strong></td> <td>Framework for building language model-powered applications</td> </tr> <tr> <td><strong>Google Gemini API</strong></td> <td>Embedding and conversational model API</td> </tr> <tr> <td><strong>FAISS</strong></td> <td>Efficient vector search engine</td> </tr> <tr> <td><strong>PyPDF</strong></td> <td>PDF text extraction</td> </tr> </table>
🛠️ Installation
bash
Copy code
# 1. Clone the repository
git clone https://github.com/your-repo/AskPDF.git
cd AskPDF

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # For macOS/Linux
# or
venv\Scripts\activate     # For Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (create .env)
echo "GEMINI_API_KEY=your_google_gemini_api_key" > .env

# 5. Run the app
streamlit run app.py
🎮 Usage
<ol> <li><strong>Upload PDFs:</strong> Go to the sidebar and upload one or multiple PDFs. After upload, click <em>Submit & Process</em> to extract the text.</li> <li><strong>Ask Questions:</strong> In the main interface, type your question about the PDFs you uploaded. The bot will respond based on the content of the PDFs.</li> <li><strong>Receive Answers:</strong> The chatbot will answer based on the content available in the processed PDFs.</li> </ol>
📁 File Structure
plaintext
Copy code
.
├── app.py                    # Main Streamlit application
├── faiss_index/               # FAISS index directory
├── .env                       # Environment variables (not included in repo)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
🔧 Troubleshooting
🛑 Common Issues
FAISS Index Not Found:
Ensure that PDFs are uploaded and processed before you attempt to ask questions. FAISS index must be built before querying.

Google Gemini API Key Error:
Verify that your API key is set in the .env file. You can obtain a key from Google Gemini API.

🚀 Future Enhancements
<ul> <li>📁 <strong>File Support:</strong> Extend support to additional file types (Word, text, etc.).</li> <li>💡 <strong>UI/UX Enhancements:</strong> Add dark mode, custom themes, and layout improvements.</li> <li>⚙️ <strong>Model Customization:</strong> Allow users to adjust parameters for embedding models.</li> </ul>
