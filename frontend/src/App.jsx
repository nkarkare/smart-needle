import { useState, useEffect, useRef } from "react";
import "./App.css";
import logo from "./assets/logo.png";

function ImageWithOverlay({ src, faces }) {
  const imgRef = useRef(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });

  const handleLoad = (e) => {
    setDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  };

  // Calculate box style relative to displayed image
  const getBoxStyle = (bbox) => {
    if (!dims.w || !bbox || !Array.isArray(bbox) || bbox.length < 4) return { display: "none" };
    const [x, y, w, h] = bbox;
    return {
      left: `${(x / dims.w) * 100}%`,
      top: `${(y / dims.h) * 100}%`,
      width: `${(w / dims.w) * 100}%`,
      height: `${(h / dims.h) * 100}%`,
    };
  };

  return (
    <div
      className="img-wrapper-inner"
      style={{ position: "relative", width: "100%", height: "100%" }}
    >
      <img
        ref={imgRef}
        src={src}
        onLoad={handleLoad}
        style={{ width: "100%", height: "100%", objectFit: "contain" }}
        loading="lazy"
      />
      {faces && Array.isArray(faces) && faces.map((face, i) => (
        <div
          key={i}
          className={`face-box ${face.name === 'Unknown' ? 'unknown-rect' : ''}`}
          style={{
            ...getBoxStyle(face.bbox),
            display: (face.name === 'Unknown' && !window.showAllFaces) ? 'none' : 'flex'
          }}
          data-name={face.name}
        >
          <span className="face-mask-label">{face.name}</span>
        </div>
      ))}
    </div>
  );
}

function PhotoCard({ photo, onPhotoClick, onDelete }) {
  const [hiddenTags, setHiddenTags] = useState(new Set());

  const toggleTag = (e, name) => {
    e.stopPropagation();
    setHiddenTags(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const visibleFaces = photo.faces.filter(f => f.name === 'Unknown' || !hiddenTags.has(f.name));

  return (
    <div className="photo-item">
      <div className="img-wrapper" onClick={() => onPhotoClick(photo)} style={{ cursor: 'zoom-in' }}>
        <ImageWithOverlay
          src={`http://localhost:9091/image?path=${encodeURIComponent(photo.path)}`}
          faces={visibleFaces}
        />
      </div>
      <div className="photo-meta">
        <span className="filename" title={photo.filename}>
          {photo.filename}
        </span>
        <div className="tags">
          {photo.faces
            .filter((f) => f.name !== "Unknown")
            .map((f, i) => (
              <span
                key={i}
                className={`tag known clickable ${hiddenTags.has(f.name) ? 'disabled' : ''}`}
                onClick={(e) => toggleTag(e, f.name)}
                title={hiddenTags.has(f.name) ? 'Show tag' : 'Hide tag'}
              >
                {f.name}
              </span>
            ))}
        </div>
        <button 
          className="delete-btn" 
          onClick={() => onDelete(photo.path)}
          title="Delete this photo"
        >
          🗑️ Delete
        </button>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("scan"); // scan, gallery, faces
  const [folderPath, setFolderPath] = useState("");
  const [photos, setPhotos] = useState([]);
  const [faces, setFaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [banner, setBanner] = useState(null); // { type: 'success' | 'error', msg: '' }
  const [confirming, setConfirming] = useState(null); // { msg: '', onConfirm: () => {} }
  const [selectedFace, setSelectedFace] = useState(null); // { photo_path: '', bbox: [] }
  const [selectedPhoto, setSelectedPhoto] = useState(null); // meta
  const [modalHiddenTags, setModalHiddenTags] = useState(new Set());
  const [showUnknown, setShowUnknown] = useState(false);
  const [threshold, setThreshold] = useState(0.40);

  const notify = (msg, type = 'success') => {
    setBanner({ msg, type });
    setTimeout(() => setBanner(null), 4000);
  };

  // Load photos on mount and when tab changes
  useEffect(() => {
    loadSettings();
    if (activeTab === "gallery") loadPhotos();
    if (activeTab === "faces") loadFaces();
  }, [activeTab]);

  const loadSettings = async () => {
    try {
      const res = await fetch("http://localhost:9091/settings");
      const data = await res.json();
      if (data.threshold) setThreshold(data.threshold);
    } catch (e) {
      console.error("Failed to load settings", e);
    }
  };

  const loadPhotos = async (query = "") => {
    try {
      let url = "http://localhost:9091/photos";
      if (query) url += `?search=${encodeURIComponent(query)}`;
      const res = await fetch(url);
      const data = await res.json();
      setPhotos(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadFaces = async () => {
    try {
      const res = await fetch("http://localhost:9091/faces");
      const data = await res.json();
      setFaces(data);
    } catch (e) {
      console.error(e);
    }
  };

  const handleBrowse = async () => {
    try {
      const res = await fetch("http://localhost:9091/browse");
      const data = await res.json();
      if (data.path) {
        setFolderPath(data.path);
      }
    } catch (e) {
      console.error(e);
    }
  };
  const handleScan = async () => {
    console.log("Starting scan for:", folderPath);
    setLoading(true);
    try {
      const response = await fetch("http://localhost:9091/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: folderPath }),
      });
      const data = await response.json();
      console.log("Scan result:", data);
      notify(
        `✅ Scan complete! Found ${data.new_photos} new photos. Total: ${data.total_photos}`
      );
    } catch (err) {
      console.error("Scan error:", err);
      notify(`❌ Error scanning: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Helper to handle uploads
  const processUpload = async (files, type) => {
    if (!files || files.length === 0) return;
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    // Determine Endpoint
    const endpoint = type === "reference" ? "upload/reference" : "upload/event";

    setLoading(true);
    try {
      const res = await fetch(`http://localhost:9091/${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }

      const data = await res.json();

      // Show success message
      notify(
        `✅ SUCCESS! ${data.message}`
      );
      
      // Auto-switch to relevant tab
      if (type === "reference") {
        setActiveTab("faces");
      } else {
        setActiveTab("gallery");
      }
    } catch (err) {
      notify(
        `❌ Upload failed! ${err.message}`, 'error'
      );
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    processUpload(e.dataTransfer.files, type);
  };

  const handleFileSelect = (e, type) => {
    processUpload(e.target.files, type);
  };

  const handleUpdateName = async (faceId, newName) => {
    try {
      await fetch(`http://localhost:9091/faces/${faceId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
      });
      // Refresh
      if (activeTab === "faces") loadFaces();
      if (activeTab === "gallery") loadPhotos();
      notify("✅ Name updated");
    } catch {
      notify("❌ Failed to update name", 'error');
    }
  };

  const inspectFace = (face) => {
    setSelectedFace({
      photo_path: face.photo_path,
      bbox: face.bbox,
      name: face.name
    });
  };

  const deleteFace = (faceId) => {
    setConfirming({
      msg: "Delete this face? This action cannot be undone.",
      onConfirm: async () => {
        setConfirming(null);
        setLoading(true);
        try {
          const res = await fetch(`http://localhost:9091/faces/${faceId}`, {
            method: "DELETE"
          });
          const data = await res.json();
          notify(`✅ ${data.message}`);
          if (activeTab === "faces") loadFaces();
          if (activeTab === "gallery") loadPhotos();
        } catch (err) {
          notify(`❌ Delete failed: ${err.message}`, 'error');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const deletePhoto = (photoPath) => {
    setConfirming({
      msg: "Delete this photo and all its faces? This action cannot be undone.",
      onConfirm: async () => {
        setConfirming(null);
        setLoading(true);
        try {
          const res = await fetch(`http://localhost:9091/photos/${encodeURIComponent(photoPath)}`, {
            method: "DELETE"
          });
          const data = await res.json();
          notify(`✅ ${data.message}`);
          if (activeTab === "gallery") loadPhotos();
        } catch (err) {
          notify(`❌ Delete failed: ${err.message}`, 'error');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const clearReferences = () => {
    setConfirming({
      msg: "⚠️ Clear ALL ID Cards?\nKeep event photos but delete reference data.",
      onConfirm: async () => {
        setConfirming(null);
        setLoading(true);
        try {
          const res = await fetch("http://localhost:9091/clear/references", {
            method: "DELETE"
          });
          const data = await res.json();
          notify(`✅ ${data.message}`);
          if (activeTab === "faces") loadFaces();
          if (activeTab === "gallery") loadPhotos();
        } catch (err) {
          notify(`❌ Clear failed: ${err.message}`, 'error');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const clearEvents = () => {
    setConfirming({
      msg: "⚠️ Clear ALL Event Photos?\nKeep ID cards but delete all event data.",
      onConfirm: async () => {
        setConfirming(null);
        setLoading(true);
        try {
          const res = await fetch("http://localhost:9091/clear/events", {
            method: "DELETE"
          });
          const data = await res.json();
          notify(`✅ ${data.message}`);
          if (activeTab === "gallery") loadPhotos();
          if (activeTab === "faces") loadFaces();
        } catch (err) {
          notify(`❌ Clear failed: ${err.message}`, 'error');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const resetAll = () => {
    setConfirming({
      msg: "🚨 RESET EVERYTHING?\nDelete all data, thumbnails, and databases.",
      onConfirm: async () => {
        setConfirming(null);
        setLoading(true);
        try {
          const res = await fetch("http://localhost:9091/reset", {
            method: "DELETE"
          });
          const data = await res.json();
          notify(`✅ ${data.message}`);
          setPhotos([]);
          setFaces([]);
        } catch (err) {
          notify(`❌ Reset failed: ${err.message}`, 'error');
        } finally {
          setLoading(false);
        }
      }
    });
  };

  const handleRematch = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:9091/rematch", { method: "POST" });
      const data = await res.json();
      notify(`🔄 ${data.message}`);
      // Refresh after a delay since it's background task
      setTimeout(loadPhotos, 2000);
    } catch (err) {
      notify(`❌ Re-match failed: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateThreshold = async (val) => {
    setThreshold(val);
    try {
      await fetch("http://localhost:9091/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: val }),
      });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand-header">
          <img src={logo} alt="Smart Needle Logo" className="app-logo" />
          <h2>Smart Needle</h2>
        </div>
        <nav>
          <button
            className={activeTab === "scan" ? "active" : ""}
            onClick={() => setActiveTab("scan")}
          >
            📡 Scan & Import
          </button>
          <button
            className={activeTab === "gallery" ? "active" : ""}
            onClick={() => setActiveTab("gallery")}
          >
            🖼️ Photo Gallery
          </button>
          <button
            className={activeTab === "faces" ? "active" : ""}
            onClick={() => setActiveTab("faces")}
          >
            👥 Face Management
          </button>
          <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>
            ⚙️ Settings
          </button>
        </nav>
      </div>

      {/* Main Content */}
      <div className="main-content">
        
        {/* Visual Notification Banner */}
        {banner && (
          <div className={`notification-banner ${banner.type}`}>
            {banner.msg}
          </div>
        )}

        {/* Global Loading Overlay */}
        {loading && (
          <div className="global-loading-overlay">
            <div className="loader"></div>
            <p>Processing...</p>
          </div>
        )}

        {/* Face Inspection Modal */}
        {selectedFace && (
          <div className="inspect-modal-overlay" onClick={() => setSelectedFace(null)}>
            <div className="inspect-modal" onClick={e => e.stopPropagation()}>
              <div className="inspect-header">
                <h3>Face Inspector: {selectedFace.name}</h3>
                <button className="close-btn" onClick={() => setSelectedFace(null)}>✕</button>
              </div>
              <div className="inspect-body">
                <ImageWithOverlay 
                  src={`http://localhost:9091/image?path=${encodeURIComponent(selectedFace.photo_path || "")}`}
                  faces={[{ bbox: selectedFace.bbox, name: selectedFace.name }]}
                />
              </div>
              <p className="hint">Original Photo: {selectedFace.photo_path?.split(/[\\/]/).pop() || "N/A"}</p>
            </div>
          </div>
        )}

        {/* Full Photo Viewer Modal */}
        {selectedPhoto && (
          <div className="inspect-modal-overlay" onClick={() => { setSelectedPhoto(null); setModalHiddenTags(new Set()); }}>
            <div className="inspect-modal" onClick={e => e.stopPropagation()}>
              <div className="inspect-header">
                <h3>{selectedPhoto.filename}</h3>
                <button className="close-btn" onClick={() => { setSelectedPhoto(null); setModalHiddenTags(new Set()); }}>✕</button>
              </div>
              <div className="inspect-body">
                <ImageWithOverlay 
                  src={`http://localhost:9091/image?path=${encodeURIComponent(selectedPhoto.path)}`}
                  faces={selectedPhoto.faces.filter(f => f.name === 'Unknown' || !modalHiddenTags.has(f.name))}
                />
              </div>
              <div className="inspect-footer">
                <div className="tags">
                  {selectedPhoto.faces
                    .filter(f => f.name !== "Unknown")
                    .map((f, i) => (
                      <span 
                        key={i} 
                        className={`tag known clickable ${modalHiddenTags.has(f.name) ? 'disabled' : ''}`}
                        onClick={() => setModalHiddenTags(prev => {
                          const next = new Set(prev);
                          if (next.has(f.name)) next.delete(f.name);
                          else next.add(f.name);
                          return next;
                        })}
                      >
                        {f.name}
                      </span>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Custom Confirmation Modal */}
        {confirming && (
          <div className="confirm-modal-overlay">
            <div className="confirm-modal">
              <h3>Confirm Action</h3>
              <p>{confirming.msg}</p>
              <div className="confirm-actions">
                <button className="primary-btn pulse" onClick={confirming.onConfirm}>Confirm</button>
                <button className="secondary-btn" onClick={() => setConfirming(null)}>Cancel</button>
              </div>
            </div>
          </div>
        )}
        {activeTab === "scan" && (
          <div className="view-container">
            <h1>Import Photos</h1>

            {/* Double Drag & Drop Zones */}
            <div
              className="upload-flex-container"
              style={{ display: "flex", gap: "2rem", flexWrap: "wrap" }}
            >
              {/* ID Cards Zone */}
              <div className="scan-card upload-zone" 
                   style={{ flex: 1, minWidth: '300px', borderColor: '#00fff2' }}
                   onDragOver={(e) => e.preventDefault()}
                   onDrop={(e) => handleDrop(e, 'reference')}>
                <h3>🪪 Upload ID Cards</h3>
                <p>Filename will be used as Name (e.g. "John.jpg")</p>
                <div style={{ fontSize: '0.8rem', color: '#00fff2', marginBottom: '1rem' }}>Legacy Face Data will be created</div>
                <button className="secondary-btn" onClick={() => document.getElementById('ref-upload').click()} disabled={loading}>
                  {loading ? '⏳ Uploading...' : 'Select ID Cards'}
                </button>
                <input 
                  type="file" multiple accept="image/*" style={{ display: 'none' }} id="ref-upload"
                  onChange={(e) => handleFileSelect(e, 'reference')}
                />
              </div>

              {/* Event Photos Zone */}
              <div className="scan-card upload-zone" 
                   style={{ flex: 1, minWidth: '300px', borderColor: '#ffe600' }}
                   onDragOver={(e) => e.preventDefault()}
                   onDrop={(e) => handleDrop(e, 'event')}>
                <h3>📸 Upload Event Photos</h3>
                <p>System will match faces to ID cards</p>
                <div style={{ fontSize: '0.8rem', color: '#ffe600', marginBottom: '1rem' }}>Automatic Tagging & Grouping</div>
                <button className="secondary-btn" onClick={() => document.getElementById('evt-upload').click()} disabled={loading}>
                  {loading ? '⏳ Uploading...' : 'Select Event Photos'}
                </button>
                <input 
                  type="file" multiple accept="image/*" style={{ display: 'none' }} id="evt-upload"
                  onChange={(e) => handleFileSelect(e, 'event')}
                />
              </div>
            </div>

            <div className="divider">
              <span>OR</span>
            </div>

            <div className="scan-card">
              <label>Local Folder Path</label>
              <div className="input-group">
                <button
                  onClick={handleBrowse}
                  className="secondary-btn"
                  style={{ marginRight: "0.5rem" }}
                >
                  📂
                </button>
                <input
                  type="text"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="e.g. C:\Photos\Events"
                />
                <button
                  onClick={handleScan}
                  disabled={loading}
                  className="primary-btn"
                >
                  {loading ? "Scanning..." : "Start Scan"}
                </button>
              </div>
              <p className="hint">
                Supports .jpg, .png. Folders are scanned recursively.
              </p>
            </div>

            <div className="scan-card">
              <label>Google Drive Integration</label>
              <p className="hint">
                1. Place <code>credentials.json</code> in backend folder.
              </p>
              <p className="hint">2. Click Connect to authorize.</p>
              <button
                className="primary-btn"
                onClick={() =>
                  (window.location.href = "http://localhost:9091/auth/login")
                }
              >
                Connect Google Account
              </button>
            </div>
          </div>
        )}

        {/* GALLERY VIEW */}
        {activeTab === "gallery" && (
          <div className="view-container">
            <div className="header-actions">
              <div className="header-title-row">
                <h1>Gallery ({photos.length})</h1>
                <button 
                  className="secondary-btn rematch-btn" 
                  onClick={handleRematch}
                  title="Re-match all faces using current ID cards"
                >
                  🔄 Re-match All Faces
                </button>
              </div>
              <div className="search-bar">
                <input
                  type="text"
                  placeholder="Search by person..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && loadPhotos(searchQuery)
                  }
                />
                <button onClick={() => loadPhotos(searchQuery)}>Search</button>
              </div>
              
              <div className="gallery-controls-row">
                <div className="threshold-control">
                  <label>
                    Matching Sensitivity: <strong>{( (1 - threshold) * 100 ).toFixed(0)}%</strong>
                    <span className="hint-text">(Higher = catch more faces, Lower = fewer false matches)</span>
                  </label>
                  <input 
                    type="range" min="0.1" max="0.8" step="0.01" 
                    value={threshold} 
                    onChange={(e) => handleUpdateThreshold(parseFloat(e.target.value))}
                    onMouseUp={handleRematch} // Trigger rematch when user stops sliding
                  />
                </div>
                
                <div className="debug-toggle">
                  <label>
                    <input 
                      type="checkbox" 
                      checked={showUnknown} 
                      onChange={(e) => {
                        setShowUnknown(e.target.checked);
                        window.showAllFaces = e.target.checked;
                      }} 
                    />
                    Show all detected faces (including Unknowns)
                  </label>
                </div>
              </div>
            </div>

            <div className="gallery-grid">
              {photos.map((photo, i) => (
                <PhotoCard 
                  key={i} 
                  photo={photo} 
                  onPhotoClick={setSelectedPhoto} 
                  onDelete={deletePhoto} 
                />
              ))}
            </div>
          </div>
        )}

        {/* FACES VIEW */}
        {activeTab === "faces" && (
          <div className="view-container">
            <h1>Face Management</h1>
            <div className="faces-grid">
              {faces.map((face) => (
                <div key={face.id} className="face-card clickable" onClick={() => inspectFace(face)}>
                  {face.thumbnail ? (
                    <img
                      src={`http://localhost:9091/thumbnail/${face.thumbnail}`}
                      alt="Face"
                    />
                  ) : (
                    <div className="no-thumb">?</div>
                  )}
                  <div className="face-input" onClick={e => e.stopPropagation()}>
                    <input
                      defaultValue={face.name}
                      onBlur={(e) => handleUpdateName(face.id, e.target.value)}
                      placeholder="Enter Name"
                    />
                  </div>
                  <button 
                    className="delete-btn" 
                    onClick={(e) => { e.stopPropagation(); deleteFace(face.id); }}
                    title="Delete this face"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SETTINGS VIEW */}
        {activeTab === 'settings' && (
          <div className="view-container">
            <h1>⚙️ Settings & Data Management</h1>
            
            <div className="scan-card" style={{ marginBottom: '2rem' }}>
              <h3>🗑️ Quick Delete Shortcuts</h3>
              <p className="hint">Manage your uploaded data with these quick actions.</p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1.5rem' }}>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', background: 'rgba(0, 255, 242, 0.05)', borderRadius: '8px' }}>
                  <div style={{ flex: 1 }}>
                    <strong style={{ color: '#00fff2' }}>Clear All ID Cards</strong>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', opacity: 0.8 }}>Delete all uploaded ID card reference faces. Event photos will be kept.</p>
                  </div>
                  <button 
                    className="secondary-btn" 
                    onClick={clearReferences}
                    disabled={loading}
                    style={{ minWidth: '120px' }}
                  >
                    {loading ? 'Processing...' : '🪪 Clear ID Cards'}
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', background: 'rgba(255, 230, 0, 0.05)', borderRadius: '8px' }}>
                  <div style={{ flex: 1 }}>
                    <strong style={{ color: '#ffe600' }}>Clear All Event Photos</strong>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', opacity: 0.8 }}>Delete all event photos and their detected faces. ID cards will be kept.</p>
                  </div>
                  <button 
                    className="secondary-btn" 
                    onClick={clearEvents}
                    disabled={loading}
                    style={{ minWidth: '120px' }}
                  >
                    {loading ? 'Processing...' : '📸 Clear Events'}
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
                  <div style={{ flex: 1 }}>
                    <strong style={{ color: '#fff' }}>Re-scan & Re-match Faces</strong>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', opacity: 0.8 }}>Update recognition for all uploaded photos using current ID cards. Use this if recognition seems wrong.</p>
                  </div>
                  <button 
                    className="primary-btn" 
                    onClick={handleRematch}
                    disabled={loading}
                    style={{ minWidth: '120px', background: '#444' }}
                  >
                    {loading ? 'Processing...' : '🔄 Re-match All'}
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', background: 'rgba(255, 0, 0, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 0, 0, 0.3)' }}>
                  <div style={{ flex: 1 }}>
                    <strong style={{ color: '#ff4444' }}>⚠️ Reset Everything</strong>
                    <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', opacity: 0.8 }}>Delete ALL data: ID cards, event photos, faces, and thumbnails. This cannot be undone!</p>
                  </div>
                  <button 
                    className="primary-btn" 
                    onClick={resetAll}
                    disabled={loading}
                    style={{ minWidth: '120px', background: '#ff4444' }}
                  >
                    {loading ? 'Processing...' : '🚨 Reset All'}
                  </button>
                </div>
              </div>
            </div>

            <div className="scan-card">
              <h3>ℹ️ About</h3>
              <p>Smart Needle - Face Recognition System</p>
              <p className="hint">Version 2.0 with DeepFace Integration</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
