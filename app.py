from flask import Flask, render_template, request, jsonify
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Updated LangChain imports
from langchain_community.llms import Ollama
# from langchain.text_splitter import CharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Available Models
MODELS = ["gemma:2b", "llama2"]

vector_store = None


# ----------------------------
# Document Processing Function
# ----------------------------
def process_document(filepath):
    global vector_store

    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
        text = df.to_string()
        docs = [{"page_content": text}]
    else:
        return "Unsupported file"

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="gemma:2b")
    vector_store = FAISS.from_documents(texts, embeddings)


# ----------------------------
# Home Route
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html", models=MODELS)


# ----------------------------
# Upload Route
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    process_document(filepath)
    return jsonify({"message": "Document processed successfully!"})


# ----------------------------
# Ask Question Route
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    data = request.json
    question = data["question"]
    model_name = data["model"]

    llm = Ollama(model=model_name)

    if vector_store:
        docs = vector_store.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful AI assistant.
Answer the question ONLY using the context below.
If the answer is not found in the context, say "Answer not found in document."

Context:
{context}

Question:
{question}

Answer:
"""

        answer = llm.invoke(prompt)

    else:
        answer = llm.invoke(question)

    return jsonify({"response": answer})


# ----------------------------
# Web Scraping Route
# ----------------------------
@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.json
    url = data["url"]
    model_name = data["model"]

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])

    llm = Ollama(model=model_name)
    summary = llm.invoke("Summarize this content:\n" + paragraphs)

    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)