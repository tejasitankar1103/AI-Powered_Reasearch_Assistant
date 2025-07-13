# 🧠 AI-Powered Research Assistant

An interactive research tool that lets users analyze web articles, generate summaries, and ask natural language questions using **transformers**, **FAISS**, and **Streamlit**.

Built using:
- 🔍 `MiniLM` for semantic embeddings
- 📦 `FAISS` for similarity search
- 🤖 `Flan-T5` for answer generation
- 🧹 `trafilatura` for article extraction

---

## 🚀 Features

- Extract text from multiple URLs
- Clean and structure content using `trafilatura`
- Embed chunks using `MiniLM`
- Store/retrieve embeddings with `FAISS`
- Answer user questions using `Flan-T5` (via Hugging Face)
- Streamlit-based UI with optional Hugging Face API integration

---

## 🛠️ Technologies Used

- Python 3.8+
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- LangChain (for Retriever support)
- Trafilatura (content cleaner)
- Streamlit (UI)

---

## ▶️ How to Run

### 💻 Locally

**1. Clone the Repository**

```bash
git clone https://github.com/tejasitankar1103/AI-Powered_Reasearch_Assistant.git
cd AI-Powered_Reasearch_Assistant
```
**2. Create Virtual Environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
**4. Run the App**
```bash
streamlit run app.py
```
## 📁 Folder Structure

```
AI-Powered_Reasearch_Assistant/
├── app.py                  # Streamlit UI
├── requirements.txt        # Dependencies
├── .env                    # (Optional) Hugging Face token
├── .gitignore              # Ignore venv/cache files
├── faiss_news_index.pkl    # FAISS vector index (auto-generated)
└── README.md               # You're reading it!
```
## 📄 License

[MIT License](LICENSE)

---

## 👤 Author

**Tejas Itankar**\
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/tejas-itankar/) or contribute to this project!


