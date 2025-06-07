from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
import threading
import queue
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)
CORS(app)

# Initialize the AI model
print("[Backend] Loading SentenceTransformer model... (this may take a while on first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[Backend] SentenceTransformer model loaded.")
print(json.dumps({"type": "backend_ready"}))

# Store for browser instances, tabs, and URL cache
browser_instances = {}  # {browser_id: {tabs: {}, tab_embeddings: {}}}
url_cache = {}  # {url_hash: {embeddings: [], content: "", last_indexed: timestamp}}
CHUNK_SIZE = 1024
DEFAULT_SIMILARITY_THRESHOLD = 0.5
MAX_WORKERS = 4

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def get_url_hash(url: str) -> str:
    """Generate a hash for URL to check if already indexed."""
    return hashlib.md5(url.encode()).hexdigest()

def get_embedding(text: str):
    """Get embedding for text."""
    return model.encode(text)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of specified size."""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size:
            if current_chunk:  # Only add if not empty
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_or_create_browser_instance(browser_id: str) -> Dict:
    """Get or create browser instance data."""
    if browser_id not in browser_instances:
        browser_instances[browser_id] = {
            'tabs': {},
            'tab_embeddings': {},
            'created_at': time.time()
        }
        print(f"[Backend] Created new browser instance: {browser_id}")
    return browser_instances[browser_id]

def process_tab_content_async(browser_id: str, tab_id: str, title: str, url: str, content: str):
    """Process tab content asynchronously."""
    try:
        print(f"[Backend] Starting async processing for tab {tab_id} in browser {browser_id}")
        
        browser_instance = get_or_create_browser_instance(browser_id)
        url_hash = get_url_hash(url)
        
        # Check if URL is already cached
        if url_hash in url_cache:
            cached_data = url_cache[url_hash]
            # Check if content is the same (simple comparison)
            if cached_data.get('content_hash') == hashlib.md5(content.encode()).hexdigest():
                print(f"[Backend] Using cached embeddings for URL: {url}")
                browser_instance['tab_embeddings'][tab_id] = cached_data['embeddings']
                browser_instance['tabs'][tab_id] = {
                    'title': title, 
                    'url': url, 
                    'content': content,
                    'indexed_at': time.time(),
                    'from_cache': True
                }
                print(json.dumps({
                    "type": "tab_updated", 
                    "browser_id": browser_id,
                    "tab_id": tab_id,
                    "from_cache": True
                }))
                return
        
        # Process content and create embeddings
        chunks = chunk_text(content)
        print(f"[Backend] Created {len(chunks)} chunks for tab {tab_id}")
        
        # Create embeddings for chunks
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only process non-empty chunks
                embedding = get_embedding(chunk)
                chunk_embeddings.append(embedding)
                if i % 10 == 0:  # Progress logging
                    print(f"[Backend] Processed chunk {i+1}/{len(chunks)} for tab {tab_id}")
        
        # Create embeddings for title and URL
        title_embedding = get_embedding(title) if title else None
        url_embedding = get_embedding(url) if url else None
        
        # Combine all embeddings
        all_embeddings = []
        if title_embedding is not None:
            all_embeddings.append(title_embedding)
        if url_embedding is not None:
            all_embeddings.append(url_embedding)
        all_embeddings.extend(chunk_embeddings)
        
        # Store in browser instance
        browser_instance['tabs'][tab_id] = {
            'title': title, 
            'url': url, 
            'content': content,
            'indexed_at': time.time(),
            'from_cache': False
        }
        browser_instance['tab_embeddings'][tab_id] = all_embeddings
        
        # Cache the results for this URL
        url_cache[url_hash] = {
            'embeddings': all_embeddings,
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'last_indexed': time.time(),
            'url': url
        }
        
        print(f"[Backend] Completed async processing for tab {tab_id}: {len(all_embeddings)} embeddings")
        print(json.dumps({
            "type": "tab_updated", 
            "browser_id": browser_id,
            "tab_id": tab_id,
            "embeddings_count": len(all_embeddings),
            "from_cache": False
        }))
        
    except Exception as e:
        print(f"[Backend] Error in async processing for tab {tab_id}: {str(e)}")
        print(json.dumps({
            "type": "tab_error", 
            "browser_id": browser_id,
            "tab_id": tab_id,
            "error": str(e)
        }))

def search_tabs(browser_id: str, query: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD, top_k: int = 5) -> List[Dict]:
    """Search tabs within a specific browser instance."""
    print(f"[Backend] Searching tabs in browser {browser_id} for query: '{query}' with threshold: {threshold}")
    
    # Ensure browser instance exists
    browser_instance = get_or_create_browser_instance(browser_id)
    
    query_embedding = get_embedding(query)
    similarities = {}
    
    tab_embeddings = browser_instance.get('tab_embeddings', {})
    tabs = browser_instance.get('tabs', {})
    
    print(f"[Backend] Found {len(tabs)} tabs and {len(tab_embeddings)} embeddings for browser {browser_id}")
    
    if not tab_embeddings:
        print(f"[Backend] No embeddings found for browser {browser_id}")
        return []
    
    for tab_id, chunk_embeddings in tab_embeddings.items():
        if not chunk_embeddings:  # Skip tabs with no embeddings
            print(f"[Backend] Skipping tab {tab_id} - no embeddings")
            continue
            
        max_similarity = 0
        for chunk_embedding in chunk_embeddings:
            try:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                max_similarity = max(max_similarity, similarity)
            except Exception as e:
                print(f"[Backend] Error calculating similarity for tab {tab_id}: {e}")
                continue
        
        if max_similarity >= threshold:
            similarities[tab_id] = max_similarity
            tab_info = tabs.get(tab_id, {})
            title = tab_info.get('title', 'Unknown')
            print(f"[Backend] Match found - Tab: {tab_id} | Title: {title} | Similarity: {max_similarity:.4f}")
    
    # Sort by similarity and return top k results
    sorted_tabs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for tab_id, sim in sorted_tabs:
        tab_info = tabs.get(tab_id, {})
        results.append({
            "tab_id": tab_id, 
            "similarity": float(sim),
            "title": tab_info.get('title', ''),
            "url": tab_info.get('url', ''),
            "indexed_at": tab_info.get('indexed_at', 0),
            "from_cache": tab_info.get('from_cache', False)
        })
    
    print(f"[Backend] Returning {len(results)} results for browser {browser_id}")
    return results

def process_message(message: str):
    """Process incoming messages from the browser."""
    try:
        print(f"[Backend] Received message: {message}")
        data = json.loads(message)
        message_type = data.get('type')
        browser_id = data.get('browser_id', 'default')
        
        # Auto-create browser instance if it doesn't exist
        if browser_id not in browser_instances:
            print(f"[Backend] Auto-creating browser instance: {browser_id}")
            get_or_create_browser_instance(browser_id)
        
        if message_type == 'search':
            query = data['query']
            threshold = float(data.get('threshold', DEFAULT_SIMILARITY_THRESHOLD))
            print(f"[Backend] Performing search for browser {browser_id}, query: '{query}' (threshold: {threshold})")
            
            # Ensure browser instance exists before searching
            browser_instance = get_or_create_browser_instance(browser_id)
            tab_count = len(browser_instance.get('tabs', {}))
            embedding_count = len(browser_instance.get('tab_embeddings', {}))
            
            print(f"[Backend] Browser {browser_id} has {tab_count} tabs and {embedding_count} embeddings")
            
            if tab_count == 0:
                print(f"[Backend] No tabs found for browser {browser_id}")
                print(json.dumps({
                    "type": "search_results", 
                    "browser_id": browser_id,
                    "results": [],
                    "query": query,
                    "message": "No tabs indexed yet. Please wait for tabs to be processed."
                }))
                return
            
            results = search_tabs(browser_id, query, threshold)
            print(f"[Backend] Search completed: {len(results)} results for browser {browser_id}")
            print(json.dumps({
                "type": "search_results", 
                "browser_id": browser_id,
                "results": results,
                "query": query
            }))
            
        elif message_type == 'update_tab':
            tab_id = data['tab_id']
            title = data.get('title', '')
            url = data.get('url', '')
            content = data.get('content', '')
            
            print(f"[Backend] Queuing tab update for browser {browser_id}, tab {tab_id}")
            print(f"[Backend] Title: {title}, URL: {url}, Content length: {len(content)}")
            
            # Process asynchronously
            future = executor.submit(process_tab_content_async, browser_id, tab_id, title, url, content)
            
        elif message_type == 'update_settings':
            global CHUNK_SIZE
            if 'chunk_size' in data:
                old_chunk_size = CHUNK_SIZE
                CHUNK_SIZE = int(data['chunk_size'])
                print(f"[Backend] Updated chunk size from {old_chunk_size} to {CHUNK_SIZE}")
                
                # Optionally re-process tabs with new chunk size
                if data.get('reprocess_tabs', False):
                    print(f"[Backend] Re-processing all tabs with new chunk size")
                    for browser_id, browser_data in browser_instances.items():
                        for tab_id, tab_info in browser_data['tabs'].items():
                            content = tab_info.get('content', '')
                            title = tab_info.get('title', '')
                            url = tab_info.get('url', '')
                            executor.submit(process_tab_content_async, browser_id, tab_id, title, url, content)
                
                print(json.dumps({"type": "settings_updated", "browser_id": browser_id}))
                
        elif message_type == 'get_stats':
            # Return statistics about the backend
            stats = {
                "type": "backend_stats",
                "browser_id": browser_id,
                "browser_instances": len(browser_instances),
                "cached_urls": len(url_cache),
                "total_tabs": sum(len(b['tabs']) for b in browser_instances.values()),
                "active_workers": executor._threads and len([t for t in executor._threads if t.is_alive()]) or 0
            }
            
            if browser_id in browser_instances:
                browser_data = browser_instances[browser_id]
                stats.update({
                    "instance_tabs": len(browser_data['tabs']),
                    "instance_embeddings": len(browser_data['tab_embeddings'])
                })
            
            print(json.dumps(stats))
            
        elif message_type == 'remove_tab':
            tab_id = data['tab_id']
            browser_instance = browser_instances.get(browser_id)
            if browser_instance:
                browser_instance['tabs'].pop(tab_id, None)
                browser_instance['tab_embeddings'].pop(tab_id, None)
                print(f"[Backend] Removed tab {tab_id} from browser {browser_id}")
                print(json.dumps({"type": "tab_removed", "browser_id": browser_id, "tab_id": tab_id}))
            
    except Exception as e:
        error_msg = f"[Backend] Error processing message: {str(e)}"
        print(error_msg)
        print(json.dumps({"type": "error", "message": str(e)}))

def read_stdin():
    """Read and process messages from stdin."""
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                process_message(line.strip())
        except Exception as e:
            print(f"[Backend] Error reading stdin: {e}")

def cleanup_old_cache():
    """Clean up old cache entries periodically."""
    current_time = time.time()
    expired_keys = []
    
    for url_hash, cache_data in url_cache.items():
        # Remove cache entries older than 24 hours
        if current_time - cache_data.get('last_indexed', 0) > 86400:
            expired_keys.append(url_hash)
    
    for key in expired_keys:
        del url_cache[key]
    
    if expired_keys:
        print(f"[Backend] Cleaned up {len(expired_keys)} expired cache entries")

def periodic_cleanup():
    """Run periodic cleanup tasks."""
    while True:
        time.sleep(3600)  # Run every hour
        cleanup_old_cache()

if __name__ == '__main__':
    # Start stdin reader in a separate thread
    stdin_thread = threading.Thread(target=read_stdin, daemon=True)
    stdin_thread.start()
    
    # Start periodic cleanup thread
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    print("[Backend] Multi-instance backend ready with async processing")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Backend] Shutting down...")
        executor.shutdown(wait=True)
        sys.exit(0)