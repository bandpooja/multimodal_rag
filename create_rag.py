import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from data_preprocessing import DataPreprocessor  # Import the DataPreprocessor class

# Initialize the DataPreprocessor and process the data
preprocessor = DataPreprocessor(csv_file="data.csv")
preprocessor.process_data()

# Now use the preprocessed data to create the RAG system
final_images = preprocessor.final_images
final_text = preprocessor.final_text

# Initialize Chroma vector store with OpenCLIP embeddings
vectorstore = Chroma(
    collection_name="mm_rag_clip_photos", embedding_function=OpenCLIPEmbeddings(), persist_directory="content"
)

# Add images and texts to the vector store
vectorstore.add_images(uris=final_images)
vectorstore.add_texts(texts=final_text)

# Create retriever
retriever = vectorstore.as_retriever()

# Now `retriever` can be used for querying or further operations
print("RAG system created successfully.")
