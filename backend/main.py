import os
import cv2
import json
import uuid
import numpy as np
import logging
import threading
import time
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Needle Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# CONSTANTS & STORAGE
# ---------------------------------------------------------
DB_FILE = "db.json"
THUMBNAILS_DIR = "thumbnails"
MODEL_NAME = "Facenet512" # Upgraded for better separation
DETECTOR_BACKEND = "retinaface"
EMBEDDING_SIZE = 512

if not os.path.exists(THUMBNAILS_DIR):
    os.makedirs(THUMBNAILS_DIR)

DB = {
    "photos": {},
    "faces": {},
    "settings": {
        "folders": [],
        "gdrive_folder_id": None
    }
}

# In-memory status for background tasks
# task_id -> { status: "running"|"completed"|"failed", "progress": "...", "result": {} }
TASKS = {}
TASK_LOCK = threading.Lock()

def load_db():
    global DB
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                loaded = json.load(f)
                DB.update(loaded)
                # Schema migration / Calculation
                if "settings" not in DB:
                    DB["settings"] = {"folders": [], "gdrive_folder_id": None, "threshold": 0.40}
                if "threshold" not in DB["settings"]:
                    DB["settings"]["threshold"] = 0.40
                
                # Ensure all reference faces have norms for faster matching and correct model version
                updated = False
                for fid, face in DB.get("faces", {}).items():
                    # Check if we need to upgrade the model embedding
                    needs_model_upgrade = False
                    if "embedding" in face and face["embedding"]:
                        if len(face["embedding"]) != EMBEDDING_SIZE:
                            needs_model_upgrade = True
                    
                    if needs_model_upgrade:
                        logger.info(f"Upgrading reference face model: {face.get('name')}")
                        path = face.get("photo_path")
                        if path and os.path.exists(path):
                            new_data = get_embedding_data(path)
                            if new_data:
                                # Pick the first/best face
                                best = new_data[0]
                                face["embedding"] = best["embedding"]
                                face["norm"] = best["norm"]
                                face["bbox"] = best["bbox"]
                                updated = True
                    
                    if face.get("norm") is None and "embedding" in face and face["embedding"]:
                        vec = np.array(face["embedding"])
                        face["norm"] = float(np.linalg.norm(vec))
                        updated = True
                if updated:
                    save_db()
        except Exception as e:
            logger.error(f"Error loading DB: {e}")

def save_db():
    with open(DB_FILE, 'w') as f:
        json.dump(DB, f, indent=2)

# load_db() - Called later after DeepFace is ready

# Face Detection Engine (Now deprecated in favor of DeepFace Unified Pipeline)
# ---------------------------------------------------------

def extract_thumbnail(img_path, bbox, face_id):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        x, y, w, h = bbox
        h_img, w_img, _ = img.shape
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        crop = img[y1:y2, x1:x2]
        thumb_filename = f"{face_id}.jpg"
        thumb_path = os.path.join(THUMBNAILS_DIR, thumb_filename)
        cv2.imwrite(thumb_path, crop)
        return thumb_filename
    except Exception as e:
        logger.error(f"Error extracting thumbnail: {e}")
        return None
    
# ---------------------------------------------------------
# BACKGROUND WORKER
# ---------------------------------------------------------
def process_folder_task(task_id, folder_path):
    logger.info(f"Starting scan for {folder_path}")
    try:
        exts = ('.jpg', '.jpeg', '.png')
        count_new = 0
        total_files = 0
        
        # First pass to count
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(exts):
                    total_files += 1

        processed = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(exts):
                    processed += 1
                    
                    # Update Progress
                    with TASK_LOCK:
                        TASKS[task_id]["progress"] = f"{processed}/{total_files}"
                    
                    full_path = os.path.join(root, file)
                    # Use Unified DeepFace Pipeline
                    faces_found = get_embedding_data(full_path)
                    
                    # Resolve unique matches
                    face_match_list = []
                    for face_data in faces_found:
                        matches = find_all_matches(face_data["embedding"])
                        face_match_list.append({
                            "data": face_data,
                            "matches": matches # list of (name, dist)
                        })

                    # Assign names greedily
                    used_names = set()
                    
                    all_possibilities = []
                    for idx, fml in enumerate(face_match_list):
                        for name, dist, fid in fml["matches"]:
                            all_possibilities.append((idx, name, dist, fid))
                    
                    all_possibilities.sort(key=lambda x: x[2]) # sort by distance
                    
                    final_assignments = {} # face_idx -> {name, fid}
                    for f_idx, name, dist, fid in all_possibilities:
                        if f_idx not in final_assignments and name not in used_names:
                            final_assignments[f_idx] = {"name": name, "fid": fid}
                            used_names.add(name)
                            logger.info(f"Greedy Match: Photo={file} Name={name} dist={dist:.4f}")

                    faces_metadata = []
                    for idx, fml in enumerate(face_match_list):
                        face_data = fml["data"]
                        bbox = face_data["bbox"]
                        assignment = final_assignments.get(idx, {})
                        
                        faces_metadata.append({
                            "bbox": bbox,
                            "name": assignment.get("name", "Unknown"),
                            "matched_id": assignment.get("fid"),
                            "embedding": face_data["embedding"],
                            "norm": face_data.get("norm")
                        })

                    DB["photos"][full_path] = {
                        "filename": file,
                        "path": full_path,
                        "faces": faces_metadata
                    }
                    if faces_metadata:
                        count_new += 1
        
        save_db()
        with TASK_LOCK:
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["result"] = {"new": count_new, "total": len(DB["photos"])}
            
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        with TASK_LOCK:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
class ScanRequest(BaseModel):
    folder_path: str

class UpdateFaceRequest(BaseModel):
    name: str

class FaceData(BaseModel):
    id: Optional[str] = None
    bbox: List[int]
    name: str
    thumbnail: Optional[str] = None
    photo_path: Optional[str] = None
    matched_id: Optional[str] = None
    embedding: Optional[List[float]] = None

class PhotoMetadata(BaseModel):
    filename: str
    path: str
    faces: List[FaceData]

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Smart Needle API v2"}

@app.post("/scan")
def scan_folder(request: ScanRequest, background_tasks: BackgroundTasks):
    folder_path = request.folder_path
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Directory not found")

    task_id = str(uuid.uuid4())
    with TASK_LOCK:
        TASKS[task_id] = {"status": "running", "progress": "Starting...", "type": "scan"}
    
    # Persist folder for auto-rescan
    if folder_path not in DB["settings"]["folders"]:
        DB["settings"]["folders"].append(folder_path)
    save_db()

    background_tasks.add_task(process_folder_task, task_id, folder_path)
    
    return {"task_id": task_id, "message": "Scan started in background"}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    with TASK_LOCK:
        task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/browse")
def browse_folder():
    """Opens a native folder dialog on the server (local machine) and returns the path."""
    import tkinter as tk
    from tkinter import filedialog
    
    # Create hidden root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Folder to Scan")
    root.destroy()
    
    if folder_path:
        return {"path": os.path.normpath(folder_path)}
    return {"path": ""}

@app.put("/faces/{face_id}")
def update_face(face_id: str, request: UpdateFaceRequest):
    if face_id not in DB["faces"]:
        raise HTTPException(status_code=404, detail="Face not found")
    DB["faces"][face_id]["name"] = request.name
    save_db()
    return {"message": "Updated"}

@app.get("/photos")
def get_photos(search: Optional[str] = None):
    result = []
    search_lower = search.lower() if search else None
    
    for path, meta in DB["photos"].items():
        faces_in_photo = []
        # Support new inline faces
        if "faces" in meta:
            for f in meta["faces"]:
                faces_in_photo.append(FaceData(
                    bbox=f["bbox"], 
                    name=f["name"],
                    photo_path=path,
                    matched_id=f.get("matched_id")
                ))
        # Support legacy face_ids
        elif "face_ids" in meta:
            for fid in meta["face_ids"]:
                if fid in DB["faces"]:
                    f = DB["faces"][fid]
                    faces_in_photo.append(FaceData(
                        id=f["id"], bbox=f["bbox"], name=f["name"], 
                        thumbnail=f.get("thumbnail"), photo_path=path
                    ))
        
        # Filter logic
        if search_lower:
            if any(search_lower in f.name.lower() for f in faces_in_photo):
                result.append(PhotoMetadata(filename=meta["filename"], path=path, faces=faces_in_photo))
        else:
            result.append(PhotoMetadata(filename=meta["filename"], path=path, faces=faces_in_photo))
            
    return result

@app.get("/faces")
def get_faces():
    """Returns only REFERENCE (ID Card) faces to keep the face database clean."""
    result = []
    for fid, data in DB["faces"].items():
        if data.get("type") == "reference":
            result.append(FaceData(
                id=data["id"], bbox=data["bbox"], name=data["name"], 
                thumbnail=data["thumbnail"], photo_path=data.get("photo_path")
            ))
    return result

@app.get("/image")
def get_image(path: str):
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/thumbnail/{filename}")
def get_thumbnail(filename: str):
    path = os.path.join(THUMBNAILS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Thumbnail not found")

# --- Google Drive Integration (Web Server Flow) ---
import pickle
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.requests import Request as StarletteRequest
import io

# Allow HTTP for local dev
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
REDIRECT_URI = "http://localhost:9091/auth/google/callback"

# Add Middleware for session
app.add_middleware(SessionMiddleware, secret_key="secret-key-walnut-saas")

def get_gdrive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials exist but expired, try to refresh
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        except Exception:
            # Refresh failed (e.g. revoked), user needs to login again
            return None
            
    if not creds or not creds.valid:
        return None

    return build('drive', 'v3', credentials=creds)

@app.get("/auth/login")
def login(request: StarletteRequest):
    if not os.path.exists(CREDS_FILE):
        raise HTTPException(status_code=500, detail="SaaS Server Credentials not configured.")

    try:
        flow = Flow.from_client_secrets_file(
            CREDS_FILE, 
            scopes=SCOPES, 
            redirect_uri=REDIRECT_URI
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        request.session['state'] = state
        return RedirectResponse(authorization_url)
    except Exception as e:
        logger.error(f"Auth Login Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate auth: {e}")

@app.get("/auth/google/callback")
async def auth_callback(request: StarletteRequest):
    state = request.session.get('state')
    
    # For local dev sometimes allow missing state if cookie issue, but better safe
    # Flow handles state verification usually
    
    try:
        flow = Flow.from_client_secrets_file(
            CREDS_FILE,
            scopes=SCOPES,
            state=state,
            redirect_uri=REDIRECT_URI
        )
        
        # Use the authorization server's response to fetch the OAuth 2.0 token.
        authorization_response = str(request.url)
        flow.fetch_token(authorization_response=authorization_response)

        creds = flow.credentials
        
        # Save credentials
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
            
        # Redirect back to frontend
        return RedirectResponse("http://localhost:5173/")
    except Exception as e:
         logger.error(f"Auth Callback Error: {e}")
         return JSONResponse({"error": f"Authentication failed: {str(e)}"}, status_code=400)

@app.get("/gdrive/list")
def list_gdrive_folders():
    try:
        service = get_gdrive_service()
        if not service:
             # Return mock if not configured, to prevent UI breakage during setup
             return [
                {"id": "mock_1", "name": "Set up credentials.json to see real folders"},
                {"id": "mock_2", "name": "Example School Folder"}
            ]
        
        # Query for folders
        results = service.files().list(
            q="mimeType='application/vnd.google-apps.folder'",
            pageSize=20, 
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])
        return items
    except Exception as e:
        logger.error(f"GDrive Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gdrive/scan")
def scan_gdrive_folder(folder_id: str = Query(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    with TASK_LOCK:
       TASKS[task_id] = {"status": "running", "progress": "Initializing GDrive Download...", "type": "gdrive_scan"}

    # Persist for auto-rescan
    DB["settings"]["gdrive_folder_id"] = folder_id
    save_db()

    if background_tasks:
        background_tasks.add_task(process_gdrive_task, task_id, folder_id)
    
    return {"task_id": task_id, "message": "GDrive Scan started"}

def process_gdrive_task(task_id, folder_id):
    logger.info(f"Starting GDrive scan for {folder_id}")
    try:
        service = get_gdrive_service()
        if not service:
            raise Exception("Google Drive Service not initialized")

        # Create local temp dir
        temp_dir = os.path.join("temp_gdrive", folder_id)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # List files in folder
        query = f"'{folder_id}' in parents and (mimeType = 'image/jpeg' or mimeType = 'image/png')"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        total = len(files)
        processed = 0
        
        for item in files:
            file_id = item['id']
            name = item['name']
            
            # Download
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Save to disk
            local_path = os.path.join(temp_dir, name)
            with open(local_path, "wb") as f:
                f.write(fh.getbuffer())
            
            # Use Unified DeepFace Pipeline
            faces_found = get_embedding_data(local_path)
            
            # Resolve unique matches
            face_match_list = []
            for face_data in faces_found:
                matches = find_all_matches(face_data["embedding"])
                face_match_list.append({"data": face_data, "matches": matches})

            used_names = set()
            all_possibilities = []
            for idx, fml in enumerate(face_match_list):
                for name_match, dist, fid in fml["matches"]:
                    all_possibilities.append((idx, name_match, dist, fid))
            
            all_possibilities.sort(key=lambda x: x[2])
            
            final_assignments = {}
            for f_idx, name_match, dist, fid in all_possibilities:
                if f_idx not in final_assignments and name_match not in used_names:
                    final_assignments[f_idx] = {"name": name_match, "fid": fid}
                    used_names.add(name_match)

            faces_metadata = []
            for idx, fml in enumerate(face_match_list):
                face_data = fml["data"]
                bbox = face_data["bbox"]
                assignment = final_assignments.get(idx, {})
                
                faces_metadata.append({
                    "bbox": bbox,
                    "name": assignment.get("name", "Unknown"),
                    "matched_id": assignment.get("fid"),
                    "embedding": face_data["embedding"]
                })

            DB["photos"][local_path] = {
                "filename": name, "path": local_path, "faces": faces_metadata
            }
            
            processed += 1
            with TASK_LOCK:
                TASKS[task_id]["progress"] = f"Downloaded & Scanned {processed}/{total}"
        
        save_db()
        with TASK_LOCK:
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["result"] = {"new": total, "total": len(DB["photos"])}

    except Exception as e:
        logger.error(f"GDrive task failed: {e}")
        with TASK_LOCK:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = str(e)


# ---------------------------------------------------------
# FACE RECOGNITION ENGINE (DeepFace)
# ---------------------------------------------------------
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

def get_embedding_data(img_path):
    """
    Returns a list of dicts: {'embedding': [...], 'bbox': [x, y, w, h]}
    Uses DeepFace internal detection + alignment for consistency.
    """
    if not DeepFace:
        return []
    try:
        # 'retinaface' is best for group photos and small faces.
        results = DeepFace.represent(
            img_path=img_path, 
            model_name=MODEL_NAME, 
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )
        
        parsed_results = []
        for res in results:
            area = res.get("facial_area", {})
            x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
            
            emb = res.get("embedding")
            norm = float(np.linalg.norm(emb)) if emb else 0
            
            parsed_results.append({
                "embedding": res["embedding"],
                "norm": norm,
                "bbox": [x, y, w, h]
            })
        return parsed_results
    except Exception as e:
        logger.error(f"DeepFace processing failed: {e}")
        return []

# Now that DeepFace is ready, we can load and upgrade the DB
load_db()


def rematch_all_photos():
    """Updates matching for all photos using stored embeddings. Very fast."""
    logger.info("Starting global re-match (fast)")
    for photo_path, meta in DB["photos"].items():
        # Auto-upgrade embeddings if they are from an older model version
        needs_repro = any(len(f.get("embedding", [])) != EMBEDDING_SIZE for f in meta["faces"])
        if needs_repro:
            logger.info(f"Upgrading embeddings for {photo_path}")
            new_faces = get_embedding_data(photo_path)
            meta["faces"] = [{
                "bbox": f["bbox"],
                "name": "Unknown",
                "matched_id": None,
                "embedding": f["embedding"],
                "norm": f.get("norm")
            } for f in new_faces]
            
        face_match_list = []
        for face_obj in meta["faces"]:
            emb = face_obj.get("embedding")
            if not emb: continue
            matches = find_all_matches(emb)
            face_match_list.append({
                "face_obj": face_obj,
                "matches": matches
            })
            
        if not face_match_list: continue
        
        used_names = set()
        all_possibilities = []
        for idx, fml in enumerate(face_match_list):
            for name, dist, fid in fml["matches"]:
                all_possibilities.append((idx, name, dist, fid))
        
        all_possibilities.sort(key=lambda x: x[2])
        
        final_assignments = {}
        for f_idx, name, dist, fid in all_possibilities:
            if f_idx not in final_assignments and name not in used_names:
                final_assignments[f_idx] = {"name": name, "fid": fid}
                used_names.add(name)

        for idx, fml in enumerate(face_match_list):
            assignment = final_assignments.get(idx, {"name": "Unknown", "fid": None})
            fml["face_obj"]["name"] = assignment["name"]
            fml["face_obj"]["matched_id"] = assignment["fid"]
            
    save_db()
    logger.info("Global re-match completed")

def run_auto_rescan(background_tasks: BackgroundTasks):
    """Triggered after new reference upload. Updates matching and re-scans sources."""
    # 1. Immediate rematch (fast)
    rematch_all_photos()
    
    # 2. Re-trigger full scans (slow, handles new files)
    for folder_path in DB["settings"].get("folders", []):
        if os.path.isdir(folder_path):
            task_id = str(uuid.uuid4())
            with TASK_LOCK:
                TASKS[task_id] = {"status": "running", "progress": "Auto-rescan...", "type": "scan"}
            background_tasks.add_task(process_folder_task, task_id, folder_path)
    
    g_id = DB["settings"].get("gdrive_folder_id")
    if g_id:
        task_id = str(uuid.uuid4())
        with TASK_LOCK:
            TASKS[task_id] = {"status": "running", "progress": "Auto-rescan GDrive...", "type": "gdrive_scan"}
        background_tasks.add_task(process_gdrive_task, task_id, g_id)

def find_all_matches(target_emb, threshold=None):
    """Returns a list of (name, distance, fid) for all matches under threshold."""
    if not target_emb: return []
    if threshold is None:
        threshold = DB["settings"].get("threshold", 0.40)
    
    target_vec = np.array(target_emb)
    if len(target_vec) != EMBEDDING_SIZE:
        return [] # Incompatible model embedding
        
    norm_target = np.linalg.norm(target_vec)
    if norm_target < 1e-6: return []
    
    matches = []
    for fid, face in DB["faces"].items():
        if face.get("type") == "reference" and face.get("embedding"):
            ref_vec = np.array(face["embedding"])
            norm_ref = face.get("norm") or np.linalg.norm(ref_vec)
            if norm_ref < 1e-6: continue
            
            sim = np.dot(target_vec, ref_vec) / (norm_target * norm_ref)
            dist = 1 - sim
            if dist < threshold:
                matches.append((face["name"], dist, fid))
    
    # Sort by best distance
    matches.sort(key=lambda x: x[1])
    return matches

@app.post("/upload/reference")
async def upload_reference(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Uploads ID cards. Uses DeepFace to detect & embed the single face."""
    uploaded_count = 0
    upload_dir = "uploads/reference"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    for file in files:
        try:
            safe_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(upload_dir, safe_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            name = os.path.splitext(file.filename)[0]
            
            # Use Unified Pipeline
            faces_found = get_embedding_data(file_path)
            
            if faces_found:
                # Assuming ID card has 1 face. Take the first (largest usually).
                # DeepFace sorts by confidence or size usually.
                face_data = faces_found[0]
                bbox = face_data["bbox"]
                emb = face_data["embedding"]
                norm = face_data.get("norm")
            else:
                bbox = [0,0,100,100]
                emb = None
                norm = 0
            
            fid = str(uuid.uuid4())
            thumb = extract_thumbnail(file_path, bbox, fid)
            
            DB["faces"][fid] = {
                "id": fid, "photo_path": file_path, "bbox": bbox, 
                "name": name, "thumbnail": thumb,
                "embedding": emb,
                "norm": norm,
                "type": "reference"
            }
            uploaded_count += 1
            
        except Exception as e:
            logger.error(f"Ref Upload failed {file.filename}: {e}")

    save_db()
    # Trigger Auto-Rescan
    run_auto_rescan(background_tasks) 
    
    return {"message": f"Registered {uploaded_count} ID Cards. Auto-rescan started.", "count": uploaded_count}

@app.post("/upload/event")
async def upload_event(files: List[UploadFile] = File(...)):
    """Uploads Event photos. Matches faces against Reference using Cosine Distance."""
    uploaded_count = 0
    matches_found = []
    
    upload_dir = "uploads/events"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    for file in files:
        try:
            safe_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(upload_dir, safe_filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Resolve unique matches
            faces_found = get_embedding_data(file_path)
            face_match_list = []
            for face_data in faces_found:
                matches = find_all_matches(face_data["embedding"])
                face_match_list.append({"data": face_data, "matches": matches})

            used_names = set()
            all_possibilities = []
            for idx, fml in enumerate(face_match_list):
                for name, dist, fid in fml["matches"]:
                    all_possibilities.append((idx, name, dist, fid))
            
            all_possibilities.sort(key=lambda x: x[2])
            
            final_assignments = {}
            for f_idx, name, dist, fid in all_possibilities:
                if f_idx not in final_assignments and name not in used_names:
                    final_assignments[f_idx] = {"name": name, "fid": fid}
                    used_names.add(name)
                    matches_found.append(name)

            faces_metadata = []
            for idx, fml in enumerate(face_match_list):
                face_data = fml["data"]
                bbox = face_data["bbox"]
                assignment = final_assignments.get(idx, {})
                
                faces_metadata.append({
                    "bbox": bbox,
                    "name": assignment.get("name", "Unknown"),
                    "matched_id": assignment.get("fid"),
                    "embedding": face_data["embedding"],
                    "norm": face_data.get("norm")
                })

            DB["photos"][file_path] = {
                "filename": file.filename,
                "path": file_path, 
                "faces": faces_metadata
            }
            uploaded_count += 1
            
        except Exception as e:
            logger.error(f"Event Upload failed {file.filename}: {e}")

    save_db()
    
    msg = f"Processed {uploaded_count} photos."
    if matches_found:
        unique_matches = list(set(matches_found))
        msg += f" Matched {len(unique_matches)} people: {', '.join(unique_matches)}"
    else:
        msg += " No known faces matched."
        
    return {"message": msg, "count": uploaded_count, "matches": matches_found}

@app.delete("/faces/{face_id}")
def delete_face(face_id: str):
    """Delete a specific face by ID"""
    if face_id not in DB["faces"]:
        raise HTTPException(status_code=404, detail="Face not found")
    
    face = DB["faces"][face_id]
    
    # Remove thumbnail if exists
    if face.get("thumbnail"):
        thumb_path = os.path.join(THUMBNAILS_DIR, face["thumbnail"])
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    
    # Remove from any photos
    for photo_path, photo_data in DB["photos"].items():
        if "faces" in photo_data:
            for face_entry in photo_data["faces"]:
                if face_entry.get("matched_id") == face_id:
                    face_entry["matched_id"] = None
                    face_entry["name"] = "Unknown"
    
    # Delete the face
    del DB["faces"][face_id]
    save_db()
    return {"message": f"Face {face_id} deleted successfully"}

@app.delete("/photos/{photo_path:path}")
def delete_photo(photo_path: str):
    """Delete a specific photo and its metadata"""
    if photo_path not in DB["photos"]:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # In the new system, we just delete the photo entry since faces are inline.
    # For legacy cleanup:
    photo_data = DB["photos"][photo_path]
    for fid in photo_data.get("face_ids", []):
        if fid in DB["faces"] and DB["faces"][fid].get("type") == "event":
            # Remove legacy thumbnail
            thumb = DB["faces"][fid].get("thumbnail")
            if thumb:
                tp = os.path.join(THUMBNAILS_DIR, thumb)
                if os.path.exists(tp): os.remove(tp)
            del DB["faces"][fid]

    del DB["photos"][photo_path]
    save_db()
    return {"message": "Photo deleted"}

@app.delete("/clear/references")
def clear_references():
    """Clear all reference (ID card) faces"""
    count = 0
    faces_to_delete = [fid for fid, face in DB["faces"].items() if face.get("type") == "reference"]
    
    for face_id in faces_to_delete:
        face = DB["faces"][face_id]
        # Remove thumbnail
        if face.get("thumbnail"):
            thumb_path = os.path.join(THUMBNAILS_DIR, face["thumbnail"])
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
        del DB["faces"][face_id]
        count += 1
    
    save_db()
    return {"message": f"Cleared {count} reference faces"}

@app.delete("/clear/events")
def clear_events():
    """Clear all event photos and any lingering event faces"""
    # Legacy Cleanup of event faces in DB["faces"]
    fids_to_del = [fid for fid, f in DB["faces"].items() if f.get("type") == "event"]
    for fid in fids_to_del:
        thumb = DB["faces"][fid].get("thumbnail")
        if thumb:
            tp = os.path.join(THUMBNAILS_DIR, thumb)
            if os.path.exists(tp): os.remove(tp)
        del DB["faces"][fid]
    
    count = len(DB["photos"])
    DB["photos"] = {}
    save_db()
    return {"message": f"Cleared {count} photos and {len(fids_to_del)} legacy faces"}

@app.delete("/reset")
def reset_database():
    global DB
    DB = {
        "photos": {}, 
        "faces": {}, 
        "settings": {"folders": [], "gdrive_folder_id": None}
    }
    
    # Clear persistence files
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    # Clear thumbnails
    for f in os.listdir(THUMBNAILS_DIR):
        os.remove(os.path.join(THUMBNAILS_DIR, f))
        
    save_db()
    return {"message": "Database cleared successfully"}

@app.post("/rematch")
def manual_rematch(background_tasks: BackgroundTasks):
    """Exposes the internal rematch function as an API endpoint."""
    background_tasks.add_task(rematch_all_photos)
    return {"message": "Global re-match started in background"}

@app.get("/settings")
def get_settings():
    return DB.get("settings", {})

@app.put("/settings")
def update_settings(settings: dict):
    if "settings" not in DB: DB["settings"] = {}
    DB["settings"].update(settings)
    save_db()
    return DB["settings"]
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9091)
