# GPT Models – Retrieval-Augmented Generation (RAG)

## 📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents and receive context-aware answers using Large Language Models (LLMs).

Instead of relying entirely on the LLM's pre-trained knowledge, the application retrieves relevant information from uploaded documents and uses that information to generate accurate and relevant responses.

The system combines document processing, text chunking, semantic embeddings, vector search, and LLM-based response generation.

## 🎯 Key Features

- PDF document loading and text extraction.
- Intelligent text chunking using Recursive Character Text Splitter.
- Semantic embeddings using Hugging Face embedding models.
- Efficient vector storage and similarity search using FAISS.
- Context-aware answer generation using Google Gemini.
- Retrieval of relevant document chunks to answer user queries.
- Environment variable management for API key security.

## 🏗️ System Architecture

1. **Document Loading:** Extract text from PDF documents using PyPDFLoader.
2. **Text Splitting:** Divide extracted text into smaller chunks with overlapping content.
3. **Embedding Generation:** Convert text chunks into numerical vector representations.
4. **Vector Database:** Store embeddings in FAISS for efficient similarity search.
5. **Query Processing:** Convert user questions into embeddings.
6. **Similarity Search:** Retrieve the most relevant document chunks.
7. **Response Generation:** Pass retrieved context and user questions to Google Gemini to generate answers.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline orchestration |
| PyPDFLoader | PDF document loading |
| Hugging Face | Text embedding models |
| BAAI/bge-small-en-v1.5 | Semantic embedding generation |
| FAISS | Vector database and similarity search |
| Google Gemini | LLM-based answer generation |
| python-dotenv | Environment variable management |

## 📂 Project Structure

```text
RAG-Project/
│
├── rag.py
├── utils.py
├── embedding_download.py
├── gpt_models.pdf
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root directory.

Add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Never upload your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python rag.py
```

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| GOOGLE_API_KEY | API key used to access Google Gemini |

## 📚 How It Works

The RAG pipeline follows a retrieve-then-generate approach:

**User Question → Query Embedding → FAISS Similarity Search → Relevant Context Retrieval → Gemini LLM → Generated Answer**

This approach helps the model generate responses grounded in the information contained in the provided PDF document.

## 🚀 Future Improvements

- Support multiple PDF document uploads.
- Implement conversational memory.
- Add source citations for generated answers.
- Introduce hybrid search combining keyword and semantic retrieval.
- Implement RAG evaluation using metrics such as Context Precision, Context Recall, and Faithfulness.
- Deploy the application using Streamlit.

## 👨‍💻 Author

**Sai Sushanth Sailla**

- GitHub: https://github.com/saisushanthsailla
- LinkedIn: https://linkedin.com/in/sai-sushanth-sailla

---

⭐ If you find this project useful, consider giving it a star!