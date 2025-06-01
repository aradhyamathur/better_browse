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

# Store for tab information
tabs = {}
tab_embeddings = {}

def get_embedding(text):
    return model.encode(text)

def search_tabs(query, top_k=5):
    print(f"[Backend] search_tabs called with query: {query}")
    query_embedding = get_embedding(query)
    similarities = {}
    
    for tab_id, embedding in tab_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities[tab_id] = similarity
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
            results = search_tabs(data['query'])
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
            # Update embedding using title, url, and content
            embedding_text = f"{title} {url} {content}"
            print(f"[Backend] Embedding text: {embedding_text}")
            tab_embeddings[tab_id] = get_embedding(embedding_text)
            print(f"[Backend] Updated embedding for tab {tab_id}: length={len(tab_embeddings[tab_id])}, sample={tab_embeddings[tab_id][:5]}")
            print(json.dumps({"type": "tab_updated", "tab_id": tab_id}))
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