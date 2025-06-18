from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

def load_vectorstore(directory="data"):
    texts = []
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    full_text = "\n".join(texts)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts([full_text], embeddings)
    db.save_local("vectorstore")
    return db

def get_qa_chain():
    db = FAISS.load_local("vectorstore", OpenAIEmbeddings())
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.2)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
