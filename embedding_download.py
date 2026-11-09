
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Specify the Hugging Face model
model_name = "BAAI/bge-small-en-v1.5"

# 2. Configure the model
model_kwargs = {
    "device": "cpu"
}

encode_kwargs = {
    "normalize_embeddings": True
}

# 3. Load the model through LangChain
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Model loaded successfully!")

# 4. Convert sample text into an embedding
text = "Retrieval augmented generation uses relevant documents."

vector = embeddings.embed_query(text)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])
