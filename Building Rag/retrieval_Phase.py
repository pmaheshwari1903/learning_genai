from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag_genai"
)

# Taking User Input
user_query = input("Ask whatever you want : ")

# Relevant Chunks from vector db
search_result = vector_db.similarity_search(query=user_query)

context = [
    f"Page Content: {result.page_content}\n Page Number : {result.metadata['page_label']}\n File Location: {result.metadata['source']}" for result in search_result
]

system_prompt = f'''
    You are the helpfull assistant who answers user query based on the available context retrieved from the pdf file along with the page_contents and page number.
    
    You should only answer the user based on the following context and navigate the user to open the right page number to know more.
    
    context : {context}
'''


response = client.chat.completions.create(
    model = "openai/gpt-oss-120b",
    messages = [
        {'role' : 'system' , 'content' : system_prompt},
        {'role' : 'user', 'content' : user_query}
    ]
)


print(f"🤖 : {response.choices[0].message.content}")