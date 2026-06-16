# 📄 RAG Chatbot — Chat with your PDF

A conversational AI chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, Groq, ChromaDB, and Streamlit.

## 🚀 Live Demo
[Click here to try it](https://rag-chatbot-f22tdnakktup8yqjmkv5yw.streamlit.app/)

## ✨ Features
- Upload any PDF and chat with it instantly
- Powered by LLaMA 3.3 70B via Groq (super fast)
- Shows source pages for every answer
- Switch between PDFs seamlessly
- Clean dark themed UI
- Clear chat history anytime

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline |
| Groq API | LLM inference (LLaMA 3.3 70B) |
| ChromaDB | Vector database |
| HuggingFace Embeddings | Text embeddings |
| Streamlit | Web UI |
| PyPDF | PDF loading |

## ⚙️ Run Locally

1. Clone the repo
   git clone https://github.com/HiraArif666/rag-chatbot.git
   cd rag-chatbot

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your Groq API key in a .env file
   GROQ_API_KEY=your_key_here

5. Run the app
   streamlit run app.py

## 📸 How it Works
1. Upload a PDF from the sidebar
2. The PDF is split into chunks and converted to embeddings
3. Embeddings are stored in ChromaDB
4. When you ask a question, relevant chunks are retrieved
5. Groq LLM generates an answer based on those chunks

## 🔑 Get a Free Groq API Key
Visit https://console.groq.com to get your free API key.
