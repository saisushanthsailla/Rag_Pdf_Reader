import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
from utils import format_docs


load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Add GEMINI_API_KEY to your .env file")

# Data Ingestion 
loader = PyPDFLoader('gpt_models.pdf')

docs = loader.load()



# Text Splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 60
)

chunks = splitter.split_documents(docs)

print(chunks[5])

# embeddings model

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

texts = [chunk.page_content for chunk in chunks]

# vector store
vector_store = FAISS.from_documents(
    documents= chunks,
    embedding= embeddings
)

print("FAISS vector store created successfully!")

vector_store.save_local("faiss_index")

print("FAISS index saved!")

# retriver

retriever = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {"k" : 3}
)

# Intialize gemini 
model="gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(
    model = model,
    temperature = 0.2
)



# prompt template
schema = [
    ResponseSchema(name = "fact_1",description = "fact1 about the topic"),
    ResponseSchema(name = "fact_2",description = "fact2 about the topic"),
    ResponseSchema(name = "fact_3",description = "fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question} \n {format_instruction}
    """,
    input_variables = ['context', 'question'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}


)

# question = "which company model is mentioned"
# retrieved_docs    = retriever.invoke(question)

# print(retrieved_docs[0].page_content)

# Building a Runnable Chain

parallel_chain =RunnableParallel({
    "context" : retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

rag_chain = parallel_chain | template | llm | parser  


question = "Which GPT model is designed for coding and professional work?"
response = rag_chain.invoke(question)


print("\nParsed output:")
print(response)

print("\nIndividual facts:")
print(response["fact_1"])
print(response["fact_2"])
print(response["fact_3"])



