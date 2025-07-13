import os
import pickle
import streamlit as st
from dotenv import load_dotenv
import trafilatura

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Constants
INDEX_FILE = "faiss_news_index.pkl"

# Streamlit layout
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("🧠 Research Assistant (Flan-T5 Edition)")
st.sidebar.title("🔗 Enter Article URLs")

# Input fields for up to 3 article URLs
article_urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Article URL {i + 1}")
    article_urls.append(url.strip())

# Button to trigger processing
start_processing = st.sidebar.button("⚙️ Process Articles")
status_box = st.empty()

# ✅ Load models and cache them
@st.cache_resource
def load_models():
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return qa_pipeline, embedding_model

qa_pipeline, embedding_model = load_models()

# ✅ Clean article extractor using Trafilatura
def load_clean_articles(urls):
    documents = []
    for url in urls:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_formatting=False,
                )
                if extracted:
                    documents.append(Document(page_content=extracted, metadata={"source": url}))
        except Exception as e:
            st.warning(f"⚠️ Failed to extract content from {url}: {e}")
    return documents

# ✅ Main processing logic
if start_processing:
    article_urls = [url for url in article_urls if url]
    if not article_urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        try:
            status_box.text("📥 Fetching and cleaning articles...")
            documents = load_clean_articles(article_urls)

            if not documents:
                st.error("❌ No readable content found.")
            else:
                status_box.text("🔍 Splitting into chunks...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = splitter.split_documents(documents)

                status_box.text("📊 Generating embeddings...")
                vector_index = FAISS.from_documents(split_docs, embedding_model)

                # ✅ Automatically delete old FAISS index before saving new one
                if os.path.exists(INDEX_FILE):
                    os.remove(INDEX_FILE)

                with open(INDEX_FILE, "wb") as f:
                    pickle.dump(vector_index, f)

                status_box.success("✅ Articles processed. You can now ask questions.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ✅ Q&A Section
query = status_box.text_input("❓ Ask a question about the articles:")

if query:
    if not os.path.exists(INDEX_FILE):
        st.error("⚠️ Please process some articles first.")
    else:
        try:
            with open(INDEX_FILE, "rb") as f:
                vector_index = pickle.load(f)

            # Retrieve top 3 chunks
            relevant_docs = vector_index.similarity_search(query, k=3)
            context = " ".join([doc.page_content for doc in relevant_docs])

            # Prompt construction
            prompt = (
                f"Answer the following question using the provided context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )

            result = qa_pipeline(prompt, max_length=512, do_sample=False)
            answer = result[0]["generated_text"]

            st.subheader("🧠 Answer")
            st.write(answer)

            st.subheader("📚 Sources")
            for doc in relevant_docs:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")

        except Exception as e:
            st.error(f"❌ Error during question answering: {e}")
