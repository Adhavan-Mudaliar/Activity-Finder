import os
import subprocess
import sys
import time
import socket
import webbrowser
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import shutil
import uvicorn

# Fix for OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import local modules
from src.retrieval.search import SearchEngine
from src.utils.video_processor import VideoProcessor
from src.models.clip_encoder import CLIPEncoder

class SearchRequest(BaseModel):
    query: str
    video_id: str = None
    top_k: int = 5

# --- APPLICATION LOGIC (from src/api/app.py) ---

clip_encoder = None
engine = None
processor = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_encoder, engine, processor
    device = "mps" if torch.mps.is_available() else "cpu"
    
    print(f"Initializing Engines (Device: {device}, Offline mode)...")
    try:
        # Load CLIP once
        print("📥 Loading CLIP Model (ViT-L-14)...")
        clip_encoder = CLIPEncoder(device=device)
        print("✅ CLIP Model loaded.")
        
        # Share it with both engines
        print("🤖 Initializing Search Engine...")
        engine = SearchEngine(clip_encoder=clip_encoder)
        print("📽️ Initializing Video Processor...")
        processor = VideoProcessor(device=device, clip_encoder=clip_encoder)
        
        print("\n✨ ALL ENGINES READY FOR RETRIEVAL ✨\n")
    except Exception as e:
        print(f"Error initializing engines: {e}")
    yield

app = FastAPI(
    title="Video Scene Retrieval API", 
    description="Offline Video Scene Retrieval System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_id(filename: str) -> str:
    """Removes special characters and spaces for safe file paths."""
    import re
    base = os.path.splitext(filename)[0]
    # Replace anything not alphanumeric, underscore, or hyphen with underscores
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
    # Avoid double underscores and trailing/leading underscores or hyphens
    clean = re.sub(r'_+', '_', clean).strip('_-')
    return clean

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if processor is None:
        raise HTTPException(status_code=503, detail="Video processor not initialized.")
    
    # Ensure directory exists
    os.makedirs("data/videos", exist_ok=True)
    
    video_id = sanitize_id(file.filename)
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join("data/videos", f"{video_id}{file_extension}")
    
    print(f"📥 RECEIVING VIDEO: {file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start processing in background
        background_tasks.add_task(processor.process_new_video, file_path, video_id)
        
        return {"message": "Video uploaded successfully. Processing started.", "video_id": video_id, "filename": file.filename}
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_scenes(request: SearchRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized.")
    
    print(f"🔍 SEARCH QUERY: \"{request.query}\" (Filter: {request.video_id})")
    try:
        results = engine.search(request.query, top_k=request.top_k, video_id=request.video_id)
        print(f"🎯 Found {len(results)} relevant scenes.")
        return {"query": request.query, "results": results}
    except Exception as e:
        print(f"❌ Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve uploaded videos
os.makedirs("data/videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")

# Mount static files last
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# --- LAUNCHER LOGIC (from run_app.py) ---

def run_command(command, description):
    print(f"\n🚀 {description}...")
    if command.startswith("python "):
        command = command.replace("python ", sys.executable + " ", 1)
    result = subprocess.run(command, shell=True, env={**os.environ, "PYTHONPATH": "."})
    return result.returncode == 0

def ensure_dependencies():
    print("🔍 Checking dependencies...")
    try:
        import multipart
    except ImportError:
        print("📦 python-multipart missing. Installing...")
        run_command(f"{sys.executable} -m pip install python-multipart", "Installing python-multipart")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def open_browser_delayed(url, delay=45):
    print(f"⏳ Waiting {delay}s for engines to initialize before opening browser...")
    time.sleep(delay)
    print(f"🌐 Opening Frontend: {url}")
    webbrowser.open(url)

def main():
    print("\n" + "="*40)
    print("✨  SNAPMOMENT: UNIFIED APP  ✨")
    print("="*40)
    
    ensure_dependencies()
    
    # Fix for OMP: Error #15
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Pre-flight checks
    video_path = "data/videos/testX.mp4"
    hdf5_path = "data/hdf5/embeddings.h5"
    if os.path.exists(video_path) and not os.path.exists(hdf5_path):
        if not run_command(f"python scripts/process_video.py --video {video_path} --interval 30", "Processing Baseline Video"):
            sys.exit(1)
        if not run_command(f"python scripts/extract_embeddings.py --video_id testX --mapping data/metadata/frame_mappings/testX_mapping.json --batch 16", "Extracting AI Embeddings"):
            sys.exit(1)
    
    index_path = "data/metadata/faiss.index"
    if not os.path.exists(index_path):
        if not run_command("python scripts/build_index.py", "Building Search Index"):
            sys.exit(1)
    
    if is_port_in_use(8000):
        print("\n⚠️  WARNING: Port 8000 is already in use!")
        print("Please stop any other instances of the app before starting.")
        sys.exit(1)

    # Set offline environment variables
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = "."

    # Start browser in background thread
    threading.Thread(target=open_browser_delayed, args=("http://127.0.0.1:8000",), daemon=True).start()

    # Start main server
    print("\n✅ Setup complete! Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
