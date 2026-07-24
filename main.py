from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

chat_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GEN-AI-API-KEY")
)

test_response = chat_llm.invoke([
    HumanMessage(content="Explain about the benefits of using Groq for AI workloads.")
])

from langchain_core.prompts import ChatPromptTemplate

alien_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Neil deGrasse Tyson."),
    ("human", "Tell me about alien life in the {location} and which celestial bodies are most likely to harbor it?")
])
alien_response = alien_prompt.invoke({"location": "the rocky planets of our solar system"})
parser = StrOutputParser()

chain = alien_prompt | chat_llm | parser
alien_response = chain.invoke({"location": "the rocky planets of our solar system"})

# RAG
# 1. Load text file
loader = TextLoader("low_fantasy_creature.txt", encoding="utf-8")
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
split_docs = text_splitter.split_documents(docs)

# 3. Create embeddings 
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build vector store
vectorstore = FAISS.from_documents(split_docs, embedder)
retriever = vectorstore.as_retriever()
# 6. RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me about a fantasy creature based on the following context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

rag_chain = rag_prompt | chat_llm | StrOutputParser()

def rag_answer(question: str) -> str:
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    return rag_chain.invoke({
        "question": question,
        "context": context
    })



# 8. Example queries
if __name__ == "__main__":
    print(test_response.content)
    print("\n" + "."*60 + "\n")
    print(alien_response)
    print(rag_answer("Describe a fantasy creature that lives in the mountains."))
    print("\n" + "."*60 + "\n")
    print(rag_answer("Explain the ecology of of small forest-dwelling creatures."))
