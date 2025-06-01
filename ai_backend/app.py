from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
import threading
import queue

app = Flask(__name__)
CORS(app)

# Initialize the AI model
print("[Backend] Loading SentenceTransformer model... (this may take a while on first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[Backend] SentenceTransformer model loaded.")
print(json.dumps({"type": "backend_ready"}))

# Store for tab information and settings
tabs = {}
tab_embeddings = {}
CHUNK_SIZE = 1024
DEFAULT_SIMILARITY_THRESHOLD = 0.5

def get_embedding(text):
    return model.encode(text)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of specified size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def search_tabs(query, threshold=DEFAULT_SIMILARITY_THRESHOLD, top_k=5):
    print(f"[Backend] search_tabs called with query: {query} and threshold: {threshold}")
    query_embedding = get_embedding(query)
    similarities = {}
    
    for tab_id, chunk_embeddings in tab_embeddings.items():
        max_similarity = 0
        for chunk_embedding in chunk_embeddings:
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)
        
        if max_similarity >= threshold:
            similarities[tab_id] = max_similarity
    
    for tab_id, sim in similarities.items():
        tab_info = tabs.get(tab_id, {})
        title = tab_info.get('title', '')
        print(f"[Backend] Tab ID: {tab_id} | Title: {title} | Similarity: {sim:.4f}")
    
    # Sort by similarity and return top k results
    sorted_tabs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"tab_id": tab_id, "similarity": float(sim)} for tab_id, sim in sorted_tabs]

def process_message(message):
    try:
        print(f"[Backend] Received message: {message}")
        data = json.loads(message)
        if data['type'] == 'search':
            print(f"[Backend] Performing search for query: {data['query']}")
            threshold = float(data.get('threshold', DEFAULT_SIMILARITY_THRESHOLD))
            results = search_tabs(data['query'], threshold)
            print(f"[Backend] Search results: {results}")
            print(json.dumps({"type": "search_results", "results": results}))
        elif data['type'] == 'update_tab':
            tab_id = data['tab_id']
            title = data['title']
            url = data['url']
            content = data.get('content', '')
            print(f"[Backend] Updating tab {tab_id} with title: {title}, url: {url}, content length: {len(content)}")
            
            # Update tab information
            tabs[tab_id] = {"title": title, "url": url, "content": content}
            
            # Split content into chunks and create embeddings
            chunks = chunk_text(content)
            chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
            
            # Add title and URL as separate chunks
            title_embedding = get_embedding(title)
            url_embedding = get_embedding(url)
            
            # Store all embeddings for this tab
            tab_embeddings[tab_id] = [title_embedding, url_embedding] + chunk_embeddings
            
            print(f"[Backend] Updated embeddings for tab {tab_id}: {len(tab_embeddings[tab_id])} chunks")
            print(json.dumps({"type": "tab_updated", "tab_id": tab_id}))
        elif data['type'] == 'update_settings':
            global CHUNK_SIZE
            if 'chunk_size' in data:
                CHUNK_SIZE = int(data['chunk_size'])
                print(f"[Backend] Updated chunk size to: {CHUNK_SIZE}")
                # Re-process all tabs with new chunk size
                for tab_id, tab_info in tabs.items():
                    content = tab_info.get('content', '')
                    chunks = chunk_text(content)
                    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
                    title_embedding = get_embedding(tab_info['title'])
                    url_embedding = get_embedding(tab_info['url'])
                    tab_embeddings[tab_id] = [title_embedding, url_embedding] + chunk_embeddings
                    print(f"[Backend] Re-processed tab {tab_id} with new chunk size")
                print(json.dumps({"type": "settings_updated"}))
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}))
        print(f"[Backend][Error] {str(e)}")

def read_stdin():
    while True:
        line = sys.stdin.readline()
        if line:
            process_message(line.strip())

if __name__ == '__main__':
    # Start stdin reader in a separate thread
    stdin_thread = threading.Thread(target=read_stdin)
    stdin_thread.daemon = True
    stdin_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        sys.exit(0) 