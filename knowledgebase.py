import fitz
import os
from pymongo import MongoClient

def load_pdf_chunks(pdf_path, chunk_size=1000, overlap=100, output_dir="chunks"):
    # Open PDF and extract all text
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text()
    doc.close()

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(all_text):
        end = min(len(all_text), start + chunk_size)
        chunk_text = all_text[start:end]
        chunks.append(chunk_text)

        # Save chunk to a text file
        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_index}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as f:
            f.write(chunk_text)

        start += chunk_size - overlap
        chunk_index += 1

    return chunks

def upload_chunks_to_mongodb(chunk_dir, mongo_uri, db_name, collection_name):
    # Connect to MongoDB Atlas
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Insert all chunk files
    for filename in sorted(os.listdir(chunk_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(chunk_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                chunk_number = int(filename.replace("chunk_", "").replace(".txt", ""))
                doc = {
                    "chunk_number": chunk_number,
                    "content": content
                }
                collection.insert_one(doc)

    print("All chunks uploaded to MongoDB.")

if __name__ == "__main__":
    # === Change these values as needed ===
    pdf_file = "your_document.pdf"  # Mention the path to your PDF
    chunk_directory = "chunks"
    chunk_size = 1000
    overlap = 100

    # MongoDB details
    mongo_uri = "YOUR_MONGODB_URI"
    database_name = "knowledge_base"  #You may change the database name as desired
    collection_name = "water_supply" #You may change the collection name as required 

    # Step 1: Load and save chunks
    chunks = load_pdf_chunks(pdf_file, chunk_size, overlap, chunk_directory)
    print(f"Total chunks created and saved: {len(chunks)}")

    # Step 2: Upload to MongoDB
    upload_chunks_to_mongodb(chunk_directory, mongo_uri, database_name, collection_name)
