import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load variables globally
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

openai_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def process_query(query: str):
    # 1. Initialize models inside the worker function to avoid Windows process deadlocks
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_rag",
        embedding=embedding_model,
    )

    print("Searching Chunks:", query)
    search_results = vector_db.similarity_search(query=query)
    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user queries based on the available
    context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the
    user to open the right page number to know more.

    Context:
    {context}
    """

    response = openai_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"🤖: {answer}")
    
    # 2. CRITICAL: Return the answer so RQ saves it to Redis for FastAPI to fetch later
    return answer
