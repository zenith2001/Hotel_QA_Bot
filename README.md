# 🏨 Hotel Q&A Bot using RAG with Local LLMs 🔍🤖  

A **Retrieval-Augmented Generation (RAG)-based chatbot** that answers hotel policy-related queries using **local LLMs**. The bot retrieves hotel rules from **PDF documents**, processes them into vector embeddings using `all-mpnet-base-v2`, and provides **accurate, context-aware answers** with **Mistral-7B-Instruct** as the LLM.  

---

## 🚀 **Features**  
✅ **Offline & Local Execution** - No reliance on external APIs.  
✅ **Hotel-Specific Policy Retrieval** - Users can select a hotel to filter relevant rules.  
✅ **ChromaDB for Fast Searching** - Stores embeddings for efficient document retrieval.  
✅ **Few-Shot Learning for Accuracy** - Uses in-context examples to improve LLM responses.  
✅ **Modular & Scalable** - Well-structured project for future enhancements.  

---

## 🛠️ **Tech Stack**  
- **LangChain** - Retrieval & LLM integration  
- **Mistral-7B (Quantized)** - Local LLM for answering queries  
- **ChromaDB** - Vector storage for fast similarity search  
- **SentenceTransformers (`all-mpnet-base-v2`)** - Embedding model for retrieval  
- **PyMuPDF (`fitz`)** - Extracts text from PDFs  
- **Python 3.8+** - Core programming language  

---

## 📂 **Project Structure**  

Hotel_Q&A_bot/ │── models/ # Local Mistral-7B model  
│── Data/ # Hotel policy PDFs  
│── chroma_db/ # ChromaDB vector store  
│── src/ # Source code  
│ │── app.py # Main chatbot interface  
│ │── preprocess.py # Extracts text & stores embeddings  
│ │── retriever.py # Retrieves hotel policies from ChromaDB  
│ │── generate_answer.py # Uses LLM to generate responses  
│ │── config.py # Stores model paths & settings  
│── requirements.txt # Dependencies  
│── README.md # Project documentation  


---

## 📝 **Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/zenith2001/Hotel_QA_Bot.git
cd Hotel_QA_Bot

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2️⃣ Download & Place the Mistral-7B Model**  
Download the **Mistral-7B GGUF model** from Hugging Face: 

Move the downloaded file to the `models/` directory:  
```bash
mv mistral-7b-instruct-v0.2.Q4_K_M.gguf models/
```

---

## 📌 **Usage**  

 Run the Q&A Chatbot
```bash
 python src/app.py
```
Example Query
```bash
🏨 Enter the hotel name: Cloudcastle_resort
You: What is the cancellation policy?
🤖 Bot: If canceled within 7 days of arrival, a 100% retention charge applies. There will be no refund.
```
