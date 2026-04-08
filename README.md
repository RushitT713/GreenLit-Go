# 🎬 GreenLit GO — Movie Success Prediction Platform

An ML-powered full-stack web application that predicts movie box office success before release, using a Stacking Ensemble (RandomForest, XGBoost, LightGBM, KNN + RidgeCV Meta-learner) trained on **2,180 curated movies**, with SHAP explainability.

---

## ✨ Features

- **Revenue Prediction** — Predicts box office gross with R² = 0.7504 across 46 engineered features
- **Success Classification** — Categorizes movies (Blockbuster to Flop) with 62.39% Accuracy
- **Script Analysis (LLM)** — Upload screenplay PDFs for Generative AI (Google Gemini 2.5 Flash) insights on plot, tone, and pacing.
- **Explainable AI** — SHAP-powered feature attribution for every prediction
- **Insights Dashboard** — 8+ interactive charts for industry trend analysis
- **What-If Simulation** — Tweak inputs and see predictions change in real-time
- **Competition Analysis** — Assess threat from other movies in the same release window
- **Multi-Industry** — Hollywood, Bollywood, Tollywood, Kollywood, Mollywood

---

## 📁 Project Structure

```
ml-web-app-project/
├── client/              # React Frontend (Vite) — Port 5173
├── server/              # Node.js/Express Backend — Port 5000
├── ml-service/          # Python Flask ML API — Port 5001
├── data-collection/     # Data collection scripts (optional, one-time use)
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start (New Device Setup)

### Prerequisites

Make sure you have these installed:

| Tool | Version | Download |
|------|---------|----------|
| **Node.js** | 18+ | https://nodejs.org |
| **Python** | 3.10+ | https://python.org |
| **Git** | Latest | https://git-scm.com |

> **Note:** MongoDB is hosted on Atlas (cloud). No local MongoDB installation needed.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/RushitT713/GreenLit-Go.git
cd GreenLit-Go
```

---

### Step 2: Setup Environment Variables

#### Server (.env)

```bash
cd server
cp .env.example .env
```

Edit `server/.env` and fill in your API keys:

```env
PORT=5000
MONGODB_URI=your_mongodb_atlas_connection_string
TMDB_API_KEY=your_tmdb_api_key
OMDB_API_KEY=your_omdb_api_key
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
ML_SERVICE_URL=http://localhost:5001
```

> **How to get API keys:**
> - **TMDB:** Sign up at [themoviedb.org](https://www.themoviedb.org/settings/api) → Settings → API
> - **OMDB:** Get a free key at [omdbapi.com](http://www.omdbapi.com/apikey.aspx)
> - **YouTube:** Create a project at [Google Cloud Console](https://console.cloud.google.com) → Enable YouTube Data API v3 → Create API Key
> - **Gemini AI:** Get a free key from Google AI Studio at [aistudio.google.com](https://aistudio.google.com/)
> - **MongoDB Atlas:** Get connection string from [MongoDB Atlas](https://cloud.mongodb.com) → Database → Connect

---

### Step 3: Install Dependencies

Run these commands from the project root:

**Backend (Node.js):**
```bash
cd server
npm install
```

**Frontend (React):**
```bash
cd client
npm install
```

**ML Service (Python):**
```bash
cd ml-service
pip install -r requirements.txt
```

> **Tip:** If you're using a Python virtual environment:
> ```bash
> cd ml-service
> python -m venv venv
> venv\Scripts\activate       # Windows
> # source venv/bin/activate  # macOS/Linux
> pip install -r requirements.txt
> ```

---

### Step 4: Run the Application

You need **3 terminals** running simultaneously:

**Terminal 1 — Backend Server (Port 5000):**
```bash
cd server
npm run dev
```

**Terminal 2 — ML Service (Port 5001):**
```bash
cd ml-service
python -m app
```

**Terminal 3 — Frontend (Port 5173):**
```bash
cd client
npm run dev
```

---

### Step 4: Verification (Second Device Checklist)

To verify the setup on a second device:
1.  **IP Whitelisting**: Ensure your current IP is whitelisted in [MongoDB Atlas](https://cloud.mongodb.com/). 
    - Database Access -> Network Access -> Add IP Address.
2.  **ML Service Connectivity**:
    - Ensure `ml-service/app.py` is running.
    - Check that `server/.env` has the correct `ML_SERVICE_URL`.
3.  **Ports Check**: Ensure ports 5173 (Vite), 5000 (Express), and 5001 (Flask) are not blocked by the firewall.
4.  **Data Fetch**: Open the app and go to "Released Movies". If data appears, connection is successful.

---

### Step 5: Start All Servers (Development)

Visit: **http://localhost:5173**

🎉 You should see the GreenLit GO home page with the cinematic dark theme!

---

## 🔗 Port Reference

| Service | Port | URL |
|---------|------|-----|
| Frontend (React + Vite) | 5173 | http://localhost:5173 |
| Backend (Node.js + Express) | 5000 | http://localhost:5000 |
| ML Service (Flask) | 5001 | http://localhost:5001 |

---

## 🎯 API Endpoints

### Movies API (`/api/movies`)
- `GET /` — List movies with filters (industry, genre, category, sort, search)
- `GET /:id` — Get movie details by ID

### Predictions API (`/api/predictions`)
- `POST /predict` — Get revenue prediction + success classification
- `POST /explain` — Get SHAP explanation for a prediction
- `POST /optimal-release` — Get optimal release timing recommendation
- `POST /competition` — Analyze competition in release window
- `POST /whatif` — Run what-if simulation

### Scripts API (`/api/scripts`)
- `POST /analyze` — Upload PDF/TXT and extract script insights via Gemini 2.5 Flash

### Trends API (`/api/trends`)
- `GET /yearly` — Yearly trends
- `GET /genres` — Genre analysis
- `GET /seasonal` — Monthly/seasonal performance
- `GET /budget-revenue` — Budget vs revenue scatter analysis
- `GET /youtube-hype` — YouTube buzz vs box office correlation
- `GET /production-houses` — Top production house analysis
- `GET /opening-weekend` — Opening weekend vs total revenue
- `GET /critic-audience` — Critic vs audience score comparison

### Talents API (`/api/talents`)
- `GET /search?q=name` — Search directors/actors via TMDB

---

## 📊 Data Sources

| Source | What We Collect |
|--------|----------------|
| **TMDB API** | Movie metadata, cast, crew, posters, popularity |
| **OMDB API** | IMDb ratings, Metascore, Rotten Tomatoes scores |
| **YouTube API** | Trailer views, likes, comments, engagement |
| **Box Office Mojo** | Opening weekend grosses (web scraping) |
| **LLM (Gemini)** | Post-upload script analysis from native PDFs |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 19, Vite, React Router, ApexCharts, Framer Motion, Lucide Icons |
| **Backend** | Node.js, Express 4, Mongoose, Axios, Google Generative AI (Gemini), pdf-parse |
| **ML Service** | Python, Flask, Scikit-learn, XGBoost, LightGBM, Optuna, SHAP |
| **Database** | MongoDB Atlas (cloud) |
| **Data Collection** | Python, BeautifulSoup, Requests, Pandas |

---

## 🐛 Troubleshooting

### "Cannot connect to MongoDB"
→ Check your `MONGODB_URI` in `server/.env`. If using Atlas, make sure your IP is whitelisted in Atlas → Network Access → Add Current IP Address.

### "ML Service not responding"
→ Make sure the ML service is running on port 5001. Check that all Python dependencies are installed: `pip install -r requirements.txt`

### "Predictions return errors"
→ The ML models (`.pkl` files) must exist in `ml-service/models/`. These are included in the repo. If missing, you'll need to retrain by running the trainer.

### "TMDB/OMDB/YouTube API errors"
→ Verify your API keys in `server/.env`. Free TMDB/OMDB keys have rate limits (~40 requests/10 seconds for TMDB).

### "Frontend shows blank page"
→ Make sure the backend is running first (port 5000). The frontend proxies API calls to the backend.

---

## 📋 Optional: Data Collection

The `data-collection/` folder contains one-time scripts used to populate the database. **You don't need to run these** — the data is already in MongoDB Atlas.

If you want to re-collect or add new movies:

```bash
cd data-collection
cp .env.example .env
# Edit .env with your API keys
pip install -r requirements.txt
```

---

## 📝 License

Academic Project — Marwadi University, 2026
