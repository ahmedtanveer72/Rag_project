from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate

import os
load_dotenv()


loader = PyPDFLoader(r"D:\Gen AI\rag1\documents\deepai.pdf")

documents = loader.load()

print(f"Number of pages: {len(documents)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)


chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")
print(chunks[0].page_content)

embedding = MistralAIEmbeddings(
    model = "mistral-embed"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)
print("Vector database created successfully!")



retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------
# LLM
# -----------------------
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# -----------------------
# Prompt
# -----------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""
)

# -----------------------
# Chat Loop
# -----------------------
print("\nRAG Chatbot Started")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("\nBot:", response.content)
    print("-" * 60)
