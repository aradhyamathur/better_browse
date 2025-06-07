from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import sys
import threading
import queue
import hashlib
import time
from typing import Dict, List, Optional, Tuple
import torch
import gc
import psutil
import os
import logging
import datetime
import signal
from collections import defaultdict, deque
def setup_logging():
    log_filename = f"browser_giga_single_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
# Try to import GPU-accelerated libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[Backend] CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("[Backend] CuPy not available, using CPU for similarity calculations")

try:
    import faiss
    FAISS_AVAILABLE = True
    print("[Backend] FAISS available for fast similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    print("[Backend] FAISS not available, using standard similarity search")

app = Flask(__name__)
CORS(app)

# CUDA Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Backend] Using device: {DEVICE}")

if DEVICE == "cuda":
    print(f"[Backend] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Backend] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

# Configuration
CHUNK_SIZE = 1024
DEFAULT_SIMILARITY_THRESHOLD = 0.5
BATCH_SIZE = 64 if DEVICE == "cuda" else 32  # Larger batches for GPU
EMBEDDING_DIM = 384
MAX_QUEUE_SIZE = 1000
PROCESSING_BATCH_SIZE = 8  # Process multiple requests together

# Global state - simple and clean
browser_instances = {}  # {browser_id: {tabs: {}, tab_embeddings: {}, faiss_index: None}}
url_cache = {}  # {url_hash: {embeddings: [], content_hash: "", last_indexed: timestamp}}
request_queue = deque()  # Simple queue for incoming requests
processing_stats = {
    "total_processed": 0,
    "total_errors": 0,
    "avg_processing_time": 0.0,
    "queue_length": 0
}

# Initialize single model instance
print("[Backend] Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
model.eval()
logger.info('[Backend] SentenceTransformer model loaded on ' + DEVICE)
print(f"[Backend] SentenceTransformer model loaded on {DEVICE}")

# Warmup the model
if DEVICE == "cuda":
    print("[Backend] Warming up CUDA model...")
    with torch.no_grad():
        _ = model.encode(["warmup text"], show_progress_bar=False)
    torch.cuda.empty_cache()
    print("[Backend] CUDA model ready")

# Setup logging


def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    result = {
        "ram_gb": process.memory_info().rss / 1e9,
        "queue_length": len(request_queue)
    }
    
    if DEVICE == "cuda":
        result.update({
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9
        })
    
    return result

def get_url_hash(url: str) -> str:
    """Generate hash for URL caching."""
    return hashlib.md5(url.encode()).hexdigest()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks."""
    if not text or not text.strip():
        return []
    
    words = text.split()
    if not words:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1
        if current_size + word_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Generate embeddings for batch of texts."""
    if not texts:
        return np.array([])
    
    # Filter empty texts
    non_empty_texts = [text.strip() for text in texts if text.strip()]
    if not non_empty_texts:
        return np.array([])
    
    try:
        with torch.no_grad():
            # Process in batches for memory efficiency
            all_embeddings = []
            for i in range(0, len(non_empty_texts), BATCH_SIZE):
                batch_texts = non_empty_texts[i:i + BATCH_SIZE]
                batch_embeddings = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=len(batch_texts)
                )
                all_embeddings.append(batch_embeddings)
            
            if all_embeddings:
                result = np.vstack(all_embeddings)
                return result
            else:
                return np.array([])
                
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.array([])

def cosine_similarity_fast(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Fast cosine similarity calculation."""
    if CUPY_AVAILABLE and DEVICE == "cuda" and embeddings.shape[0] > 50:
        try:
            # Use GPU for large datasets
            query_gpu = cp.asarray(query_embedding)
            embeddings_gpu = cp.asarray(embeddings)
            
            # Normalize
            query_norm = query_gpu / cp.linalg.norm(query_gpu)
            embeddings_norm = embeddings_gpu / cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
            
            # Compute similarity
            similarities = cp.dot(embeddings_norm, query_norm)
            return cp.asnumpy(similarities)
            
        except Exception as e:
            logger.warning(f"GPU similarity failed, using CPU: {e}")
    
    # CPU fallback
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([query_embedding], embeddings)[0]

def create_faiss_index(embeddings: np.ndarray) -> Optional:
    """Create FAISS index for fast search."""
    if not FAISS_AVAILABLE or embeddings.shape[0] == 0:
        return None
    try:
        embeddings_normalized = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)
        if DEVICE == "cuda" and embeddings.shape[0] > 500:
            # Use GPU FAISS for large datasets
            try:
                res = faiss.StandardGpuResources()
                res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory
                index = faiss.IndexFlatIP(embeddings.shape[1])
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(embeddings_normalized)
                logger.info(f"Created GPU FAISS index with {embeddings.shape[0]} vectors")
                return gpu_index
            except Exception as gpu_error:
                logger.warning(f"GPU FAISS failed: {gpu_error}, using CPU")
        # CPU FAISS
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings_normalized)
        logger.info(f"Created CPU FAISS index with {embeddings.shape[0]} vectors")
        return index
    except Exception as e:
        logger.error(f"FAISS index creation failed: {e}")
        return None

def rebuild_search_index(browser_id: str):
    """Rebuild search index for browser instance."""
    if browser_id not in browser_instances:
        return
    
    browser_data = browser_instances[browser_id]
    
    try:
        all_embeddings = []
        tab_id_to_idx = {}
        current_idx = 0
        
        for tab_id, embeddings in browser_data['tab_embeddings'].items():
            if embeddings and len(embeddings) > 0:
                embeddings_array = np.array(embeddings)
                if embeddings_array.ndim == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)
                
                all_embeddings.append(embeddings_array)
                for i in range(embeddings_array.shape[0]):
                    tab_id_to_idx[current_idx + i] = tab_id
                current_idx += embeddings_array.shape[0]
        
        if all_embeddings:
            embedding_matrix = np.vstack(all_embeddings)
            browser_data['embedding_matrix'] = embedding_matrix
            browser_data['tab_id_to_embedding_idx'] = tab_id_to_idx
            browser_data['faiss_index'] = create_faiss_index(embedding_matrix)
            logger.debug(f"Rebuilt search index for {browser_id}: {embedding_matrix.shape[0]} embeddings")
        else:
            browser_data['embedding_matrix'] = None
            browser_data['tab_id_to_embedding_idx'] = {}
            browser_data['faiss_index'] = None
            
    except Exception as e:
        logger.error(f"Error rebuilding search index: {e}")

def process_tab_content(browser_id: str, tab_id: str, title: str, url: str, content: str) -> dict:
    """Process tab content and generate embeddings."""
    start_time = time.time()
    try:
        # Initialize browser instance if needed
        if browser_id not in browser_instances:
            browser_instances[browser_id] = {
                'tabs': {},
                'tab_embeddings': {},
                'faiss_index': None,
                'embedding_matrix': None,
                'tab_id_to_embedding_idx': {},
                'created_at': time.time()
            }
        browser_data = browser_instances[browser_id]
        url_hash = get_url_hash(url)
        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if url_hash in url_cache:
            cached_data = url_cache[url_hash]
            if cached_data.get('content_hash') == content_hash:
                logger.info(f"Using cached embeddings for tab {tab_id}")
                browser_data['tab_embeddings'][tab_id] = cached_data['embeddings']
                browser_data['tabs'][tab_id] = {
                    'title': title, 'url': url, 'content': content,
                    'indexed_at': time.time(), 'from_cache': True
                }
                rebuild_search_index(browser_id)
                return {
                    "type": "tab_updated",
                    "browser_id": browser_id,
                    "tab_id": tab_id,
                    "status": "success",
                    "from_cache": True,
                    "embeddings_count": len(cached_data['embeddings']),
                    "processing_time": time.time() - start_time
                }
        # Process content
        chunks = chunk_text(content)
        logger.info(f"Processing tab {tab_id}: {len(chunks)} chunks")
        # Prepare texts for embedding
        all_texts = []
        if title and title.strip():
            all_texts.append(title)
        if url and url.strip():
            all_texts.append(url)
        all_texts.extend([chunk for chunk in chunks if chunk.strip()])
        if not all_texts:
            return {
                "type": "tab_updated",
                "browser_id": browser_id,
                "tab_id": tab_id,
                "status": "error",
                "error": "No valid text content found"
            }
        # Generate embeddings
        embeddings = get_embeddings_batch(all_texts)
        if embeddings.size == 0:
            return {
                "type": "tab_updated",
                "browser_id": browser_id,
                "tab_id": tab_id,
                "status": "error",
                "error": "Failed to generate embeddings"
            }
        # Store results
        browser_data['tabs'][tab_id] = {
            'title': title, 'url': url, 'content': content,
            'indexed_at': time.time(), 'from_cache': False
        }
        browser_data['tab_embeddings'][tab_id] = embeddings.tolist()
        # Update cache
        url_cache[url_hash] = {
            'embeddings': embeddings.tolist(),
            'content_hash': content_hash,
            'last_indexed': time.time(),
            'url': url
        }
        # Rebuild search index
        rebuild_search_index(browser_id)
        processing_time = time.time() - start_time
        logger.info(f"Processed tab {tab_id} in {processing_time:.2f}s: {embeddings.shape[0]} embeddings")
        return {
            "type": "tab_updated",
            "browser_id": browser_id,
            "tab_id": tab_id,
            "status": "success",
            "from_cache": False,
            "embeddings_count": embeddings.shape[0],
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing tab {tab_id}: {e}")
        return {
            "type": "tab_updated",
            "browser_id": browser_id,
            "tab_id": tab_id,
            "status": "error",
            "error": str(e)
        }

def search_tabs(browser_id: str, query: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD, top_k: int = 5) -> List[Dict]:
    """Search tabs using FAISS or similarity calculation."""
    if browser_id not in browser_instances:
        return []
    
    browser_data = browser_instances[browser_id]
    tabs = browser_data.get('tabs', {})
    print(json.dumps({
        "type": "debug",
        "message": "Starting search",
        "browser_id": browser_id,
        "tabs_count": len(tabs)
    }))
    
    if not tabs:
        return []
    
    try:
        # Generate query embedding
        query_embedding = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        
        # Use FAISS if available
        faiss_index = browser_data.get('faiss_index')
        embedding_matrix = browser_data.get('embedding_matrix')
        tab_id_to_idx = browser_data.get('tab_id_to_embedding_idx', {})
        
        if faiss_index is not None and embedding_matrix is not None:
            print(json.dumps({
                "type": "debug",
                "message": "Using FAISS search",
                "browser_id": browser_id,
                "matrix_shape": embedding_matrix.shape
            }))
            # FAISS search
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            query_norm = query_norm.reshape(1, -1).astype(np.float32)
            
            k = min(top_k * 5, embedding_matrix.shape[0])
            scores, indices = faiss_index.search(query_norm, k)
            
            # Aggregate scores by tab
            tab_scores = defaultdict(list)
            for score, idx in zip(scores[0], indices[0]):
                if idx in tab_id_to_idx and score >= threshold:
                    tab_id = tab_id_to_idx[idx]
                    tab_scores[tab_id].append(float(score))
            
            tab_max_scores = {tab_id: max(scores) for tab_id, scores in tab_scores.items()}
            
        else:
            print(json.dumps({
                "type": "debug",
                "message": "Using standard similarity search",
                "browser_id": browser_id
            }))
            # Standard similarity search
            tab_max_scores = {}
            tab_embeddings = browser_data.get('tab_embeddings', {})
            
            for tab_id, embeddings in tab_embeddings.items():
                if not embeddings:
                    continue
                
                embeddings_array = np.array(embeddings)
                if embeddings_array.ndim == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)
                
                similarities = cosine_similarity_fast(query_embedding, embeddings_array)
                max_similarity = np.max(similarities)
                
                if max_similarity >= threshold:
                    tab_max_scores[tab_id] = float(max_similarity)
        
        # Sort and format results
        sorted_tabs = sorted(tab_max_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        
        print(json.dumps({
            "type": "debug",
            "message": "Processing search results",
            "browser_id": browser_id,
            "results_count": len(sorted_tabs)
        }))
        
        for tab_id, similarity in sorted_tabs:
            tab_info = tabs.get(tab_id, {})
            
            # Get content preview (first 200 characters)
            content = tab_info.get('content', '')
            content_preview = content[:200] + '...' if len(content) > 200 else content
            
            # Format indexed time
            indexed_at = tab_info.get('indexed_at', 0)
            indexed_time = datetime.datetime.fromtimestamp(indexed_at).strftime('%Y-%m-%d %H:%M:%S')
            
            result = {
                "tab_id": tab_id,
                "similarity": similarity,
                "title": tab_info.get('title', ''),
                "url": tab_info.get('url', ''),
                "content_preview": content_preview,
                "indexed_at": indexed_time,
                "from_cache": tab_info.get('from_cache', False),
                "content_length": len(content),
                "match_details": {
                    "similarity_score": f"{similarity:.4f}",
                    "threshold": threshold,
                    "is_cached": tab_info.get('from_cache', False)
                }
            }
            
            # Log detailed match information
            print(json.dumps({
                "type": "debug",
                "message": "Found match",
                "browser_id": browser_id,
                "tab_id": tab_id,
                "title": result['title'],
                "url": result['url'],
                "similarity": f"{similarity:.4f}",
                "indexed_at": indexed_time,
                "from_cache": result['from_cache']
            }))
            
            results.append(result)
        
        print(json.dumps({
            "type": "debug",
            "message": "Search completed",
            "browser_id": browser_id,
            "query": query[:50],
            "results_count": len(results)
        }))
        
        return results
        
    except Exception as e:
        error_msg = f"Error in search: {str(e)}"
        print(json.dumps({
            "type": "error",
            "message": error_msg,
            "browser_id": browser_id
        }))
        logger.error(error_msg)
        return []

def process_requests():
    """Main processing loop - handles requests from queue."""
    print(json.dumps({
        "type": "debug",
        "message": "Starting request processing loop"
    }))
    
    while True:
        try:
            # Check for requests in queue
            if not request_queue:
                time.sleep(0.01)  # Short sleep to prevent CPU spinning
                continue
                
            # Process batch of requests
            batch = []
            batch_size = min(PROCESSING_BATCH_SIZE, len(request_queue))
            for _ in range(batch_size):
                if request_queue:
                    batch.append(request_queue.popleft())
                    
            if not batch:
                continue
                
            # Process each request in the batch
            for request_data in batch:
                start_time = time.time()
                try:
                    request_type = request_data.get('type')
                    browser_id = request_data.get('browser_id', 'default')
                    
                    print(json.dumps({
                        "type": "debug",
                        "message": f"Processing {request_type} request",
                        "browser_id": browser_id
                    }))
                    
                    if request_type == 'update_tab':
                        result = process_tab_content(
                            browser_id,
                            request_data['tab_id'],
                            request_data.get('title', ''),
                            request_data.get('url', ''),
                            request_data.get('content', '')
                        )
                        print(json.dumps(result))
                        
                    elif request_type == 'search':
                        print(json.dumps({
                            "type": "debug",
                            "message": "Starting search",
                            "browser_id": browser_id,
                            "query": request_data.get('query', '')[:50]
                        }))
                        
                        results = search_tabs(
                            browser_id,
                            request_data['query'],
                            float(request_data.get('threshold', DEFAULT_SIMILARITY_THRESHOLD)),
                            int(request_data.get('top_k', 5))
                        )
                        
                        response = {
                            "type": "search_results",
                            "browser_id": browser_id,
                            "results": results,
                            "query": request_data['query']
                        }
                        
                        print(json.dumps(response))
                        
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    print(json.dumps({
                        "type": "error",
                        "message": error_msg,
                        "browser_id": request_data.get('browser_id', 'default')
                    }))
                    logger.error(error_msg)
                    
        except Exception as e:
            error_msg = f"Error in request processing loop: {str(e)}"
            print(json.dumps({
                "type": "error",
                "message": error_msg
            }))
            logger.error(error_msg)
            time.sleep(1)  # Sleep on error to prevent tight loop

def cleanup_cache():
    """Clean up old cache entries."""
    current_time = time.time()
    expired_keys = []
    
    for url_hash, cache_data in url_cache.items():
        if current_time - cache_data.get('last_indexed', 0) > 86400:  # 24 hours
            expired_keys.append(url_hash)
    
    for key in expired_keys:
        del url_cache[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

def add_request_to_queue(request_data: dict):
    """Add request to processing queue."""
    if len(request_queue) >= MAX_QUEUE_SIZE:
        logger.warning("Request queue full, dropping oldest request")
        request_queue.popleft()
    
    request_queue.append(request_data)

def read_stdin():
    """Read messages from stdin and add to queue."""
    while True:
        try:
            line = sys.stdin.readline()
            if line:
                try:
                    data = json.loads(line.strip())
                    add_request_to_queue(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Error reading stdin: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting single-process CUDA backend")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Max queue size: {MAX_QUEUE_SIZE}")
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_requests, daemon=True)
    processing_thread.start()
    
    # Start stdin reader thread
    stdin_thread = threading.Thread(target=read_stdin, daemon=True)
    stdin_thread.start()
    
    # Start periodic cleanup thread
    def cleanup_loop():
        while True:
            time.sleep(3600)
            cleanup_cache()

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    
    # Send backend ready message
    print(json.dumps({
        "type": "backend_ready",
        "status": "ready",
        "device": DEVICE,
        "model": "all-MiniLM-L6-v2",
        "gpu_available": DEVICE == "cuda",
        "faiss_available": FAISS_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE
    }))
    
    logger.info("Backend ready for processing")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Main thread error: {e}")
        signal_handler(signal.SIGTERM, None)