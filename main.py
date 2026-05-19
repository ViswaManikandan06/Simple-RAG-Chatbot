from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_ollama import ChatOllama

# Load PDF
loader = PyPDFLoader("VISWA_M_RESUME_MAY_2026_DATA_SCINETIST.pdf")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(
    model="llama3"
)

# Store in vector DB
vectorstore = Chroma.from_documents(
    docs,
    embeddings
)

# Create retriever
retriever = vectorstore.as_retriever()

# Load chat model
llm = ChatOllama(model="llama3")

print("PDF Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)

    context = "\n".join([
        doc.page_content for doc in relevant_docs
    ])

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\nBot:", response.content)
    print("\n" + "-"*50 + "\n")