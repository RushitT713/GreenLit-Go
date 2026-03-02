# GreenLit Go - Movie Success Prediction Web App

Data-Driven Movie Success Prediction platform for Hollywood and Indian Cinema.

## 🎬 Features

- **Revenue Prediction**: Predict box office gross
- **Success Classification**: Categorize movies as Blockbuster, Hit, Average, or Flop
- **Rating Prediction**: Predict IMDb scores
- **Trend Analysis**: Genre and seasonal performance insights
- **What-If Simulation**: Test different scenarios
- **Explainable AI**: Understand why predictions are made

## 📁 Project Structure

```
ml-web-app-project/
├── client/          # React Frontend (Vite)
├── server/          # Node.js/Express Backend
├── ml-service/      # Python Flask ML API
└── data-collection/ # Data collection scripts
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- MongoDB (local or cloud)

### 1. Setup Environment Variables

```bash
# server/.env
cp server/.env.example server/.env
# Add your API keys: TMDB_API_KEY, OMDB_API_KEY, YOUTUBE_API_KEY
```

### 2. Install Dependencies

```bash
# Backend
cd server && npm install

# Frontend
cd client && npm install

# ML Service
cd ml-service && pip install -r requirements.txt

# Data Collection
cd data-collection && pip install -r requirements.txt
```

### 3. Start MongoDB

Make sure MongoDB is running on `localhost:27017`

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
cd server && npm run dev
```

**Terminal 2 - ML Service:**
```bash
cd ml-service && python -m app
```

**Terminal 3 - Frontend:**
```bash
cd client && npm run dev
```

Visit: http://localhost:5173

## 🎯 API Endpoints

### Movies API (`/api/movies`)
- `GET /` - List movies with filters
- `GET /:id` - Get movie details
- `GET /search` - Search movies
- `GET /trending` - Trending movies

### Predictions API (`/api/predictions`)
- `POST /predict` - Get movie prediction
- `POST /explain` - Get SHAP explanation
- `POST /simulate` - What-if simulation

### Trends API (`/api/trends`)
- `GET /yearly` - Yearly trends
- `GET /genres` - Genre analysis
- `GET /seasonal` - Monthly performance

## 📊 Data Sources

- **TMDB API** - Movie metadata, cast, crew
- **OMDB API** - IMDb ratings, Rotten Tomatoes
- **YouTube API** - Trailer views, engagement
- **Web Scraping** - Indian cinema box office data

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Vite, Framer Motion |
| Backend | Node.js, Express, MongoDB |
| ML Service | Python, Flask, Scikit-Learn, XGBoost |
| Explainability | SHAP |

## 📝 License

Academic Project - 2026
