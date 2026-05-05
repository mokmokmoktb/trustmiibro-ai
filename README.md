# 🛡️ TrustMiiBro - AI Password Enhancer (Backend)

An intelligent backend service built with **FastAPI** and **Google Gemini 1.5 Flash**. This API takes simple, weak passwords or phrases and transforms them into highly secure, memorable, leetspeak-style passwords. 

Unlike standard AI generators that struggle with exact character counts, this project uses a **Hybrid Architecture**: AI handles the creativity, while strict Python logic enforces exact lengths, symbols, uppercase, and number constraints.

## ✨ Key Features

* **🧠 AI-Powered Creativity:** Uses Google's Gemini API to generate base passwords.
* **📏 100% Strict Formatting:** Python algorithms take the AI's output and strictly enforce the user's requested constraints (exact length, mandatory numbers, symbols, uppercase, etc.).
* **🛡️ Offline Fallback Mechanism:** If the Gemini API hits a rate limit (429 Quota Exceeded) or the internet drops, the system seamlessly falls back to a 100% offline, highly secure Python generation algorithm. The app never crashes.
* **⚡ Blazing Fast:** Replaced heavy machine learning libraries (like PyTorch) with lightweight API calls, making deployment incredibly fast and reducing server costs.
* **🛠️ Auto-Configuration:** Includes a `start.py` script that automatically bypasses Windows file-extension hiding by generating the required `.env` file on the fly.

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
* Python 3.8 or higher installed on your machine.

### 2. Installation
Clone this repository and install all required dependencies directly via pip:
```bash
pip install fastapi uvicorn pydantic google-generativeai python-dotenv
(Alternatively, if you have a requirements.txt file, you can run pip install -r requirements.txt)
```

### 3. Add your API Key
Open the start.py file and put your actual Google Gemini API Key on line 5:

Python
API_KEY = "YOUR_GEMINI_API_KEY_HERE" # (Line 5)

### 4. Connect your Frontend
To make sure your website can talk to this backend, open your frontend code (home.js) and check your URL and endpoint on line 144.

### 5. Run the Server
Forget about manually managing .env files or Uvicorn paths. Just run the unified startup script:

```bash

python start.py
This script will:

Auto-generate a perfectly formatted .env file.

Auto-detect the best available Gemini model to prevent 404 errors.

Start the FastAPI server at http://127.0.0.1:8000.

```
### 🏗️ Technologies Used
FastAPI: High-performance web framework for building APIs.

Uvicorn: Lightning-fast ASGI server.

Google Generative AI: SDK for interacting with Gemini 1.5 Flash.

Python Dotenv: For managing environment variables securely.
