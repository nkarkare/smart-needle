# Smart Needle 📍

**AI-Powered Face Tagging for Schools. Private. Local. Fast.**

> Built in a 48-hour Christmas Sprint by [Nikhil Karkare](https://www.linkedin.com/in/nikhilkarkare/) & Antigravity (AI Agent).

![Smart Needle Demo](smart_needle_demo.gif)

**Smart Needle** is a 100% local, privacy-first face recognition engine that uses your laptop's power to index thousands of photos and tag students automatically.

## ✨ Key Features

- **🔒 Privacy First**: All facial recognition (detection & embedding) happens **locally on your device**. No photos are ever sent to external APIs.
- **🧠 High-Fidelity Recognition**: Powered by the `Facenet512` model for deep, 512-dimensional face embeddings, ensuring distinct separation even between similar-looking students.
- **👁️ RetinaFace Detection**: Uses the advanced `retinaface` backend to catch small faces in large group photos where other detectors fail.
- **🎚️ Sensitivity Slider**: A unique UI feature that lets you adjust the matching threshold in real-time. Turn it down for strict matching, or turn it up to find blurry/partial faces (Google Photos style).
- **🤝 Greedy Assignment**: In group photos, the engine mathematically ensures that the "best match" claims a student identity first, preventing duplicate tags.

## 🛠️ Tech Stack

- **Backend**: Python (FastAPI, DeepFace, Uvicorn)
- **Frontend**: React (Vite, CSS Modules)
- **Database**: efficient JSON-based flat-file DB (no heavy SQL setup required)
- **Storage**: Local filesystem + Google Drive API integration

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js & npm

### 1. clone the repo

```bash
git clone https://github.com/nkarkare/smart-needle.git
cd smart_needle
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate, Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py
```

_The backend will start on `http://localhost:9091`_

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

_The UI will open at `http://localhost:5173`_

## 📖 Usage Guide

1.  **Upload Reference**: Go to the **Faces** tab and upload a clear ID card photo of a student. Name the file carefully (e.g., `John_Doe.jpg`).
2.  **Scan Folder**: Go to the **Scan** tab and point the tool at a local folder (or Google Drive folder) containing your event photos.
3.  **Watch the Magic**: The system will index the photos in the background.
4.  **Gallery**: Go to the **Gallery** to see your matched photos! Use the **Sensitivity Slider** to fine-tune the results if you missed any faces.

## 🤝 Contributing

This project was built as a fun experiment to solve a real-world problem. Extensions, PRs, and suggestions are welcome!

## 📜 License

MIT License. Free to use for any school or parent who needs it.
