from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# step 1. Load your document
loader = TextLoader("my_document.txt")
docs = loader.load()

# step 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# step 3. Embed chunks and store in chroma vector database
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# step 4. Create a retriever from the vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# step 5. Set up the LLM (Ollama running llama3.2)
llm = ChatOllama(model="llama3.2", temperature=0)

# step 6. Build the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# step 7. Ask a question
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break

        result = qa_chain.invoke({"query": query})
        print("\n--- Answer ---")
        print(result["result"])
        print("\n--- Sources ---")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'unknown')}")