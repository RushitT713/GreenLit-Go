# GreenLit GO — Complete Project Deep-Dive Report

> Everything you need to know about the entire project — What we built, How we built it, Why each decision was made, and How every piece works together.

> **Updated: March 4, 2026**

---

## Table of Contents

### Part A: Project Foundation
1. [What is GreenLit GO?](#1-what-is-greenlit-go)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Data Collection Pipeline](#4-data-collection-pipeline)

### Part B: Machine Learning (The Brain)
5. [ML Problem Definition](#5-ml-problem-definition)
6. [Feature Engineering: The Secret Sauce](#6-feature-engineering-the-secret-sauce)
7. [Model Selection & Training](#7-model-selection--training)
8. [Hyperparameter Tuning with Optuna](#8-hyperparameter-tuning-with-optuna)
9. [Ensemble Learning](#9-ensemble-learning)
10. [Cross-Validation & Evaluation](#10-cross-validation--evaluation)
11. [Model Performance & Results](#11-model-performance--results)
12. [SHAP Explainability](#12-shap-explainability)
13. [Prediction Flow: Input to Output](#13-prediction-flow-input-to-output)
14. [Model Serialization](#14-model-serialization)

### Part C: Backend (The Backbone)
15. [Node.js Server](#15-nodejs-server)
16. [ML Service (Flask)](#16-ml-service-flask)

### Part D: Frontend (The Face)
17. [Design System & Aesthetics](#17-design-system--aesthetics)
18. [Home Page](#18-home-page)
19. [Released Movies Library](#19-released-movies-library)
20. [Movie Detail Page](#20-movie-detail-page)
21. [Insights Page (Data Analytics)](#21-insights-page-data-analytics)
22. [Upcoming Dashboard (Prediction Engine)](#22-upcoming-dashboard-prediction-engine)
23. [About Us Page](#23-about-us-page)
24. [Reusable Components](#24-reusable-components)

### Part E: Wrap-Up
25. [Key Learnings & Insights](#25-key-learnings--insights)
26. [How to Run the Project](#26-how-to-run-the-project)

---

# PART A: PROJECT FOUNDATION

---

## 1. What is GreenLit GO?

### The Problem
The film industry invests **billions** annually in movie production with extreme uncertainty about returns. Studios, producers, and filmmakers need data-driven insights to:
- Predict potential box office revenue **before** production
- Classify movies into success categories (Blockbuster → Flop)
- Understand **which factors** drive movie success
- Choose the **optimal release window**
- Analyze **competition** in their release window

### The Solution
**GreenLit GO** is a full-stack web application that uses **Machine Learning** to solve this. It provides:

| Feature | What It Does |
|---------|-------------|
| **Revenue Prediction** | Predicts worldwide box office gross using ensemble ML models |
| **Success Classification** | Categorizes movies into 5 tiers: Blockbuster, Super Hit, Hit, Average, Flop |
| **Explainable AI (SHAP)** | Shows exactly WHY a prediction was made |
| **Released Movies Library** | Browse 1,600+ movies with filters, search, and analytics |
| **Insights Dashboard** | 8+ interactive charts analyzing industry trends |
| **Optimal Release Timing** | Recommends the best release month |
| **Competition Analysis** | TMDB-powered analysis of competing movies |
| **What-If Simulator** | Tweak inputs and see predictions change in real-time |

### Project Scope

| Aspect | Coverage |
|--------|----------|
| **Movies Analyzed** | 1,600+ films (2015-2025) |
| **Industries** | Hollywood, Bollywood, Tollywood, Kollywood, Mollywood |
| **Prediction Types** | Revenue, Success Category, ROI, Confidence |
| **Revenue Model Accuracy** | R² = 0.9340 (93.4% variance explained) |
| **Classification Accuracy** | 57.4% across 5 classes (2.85× better than random) |

### Team

| Member | Role |
|--------|------|
| **Rushit Trambadia** | ML & Backend Developer — Built the ML pipeline, trained models, engineered 40+ features |
| **Vandit Doshi** | Frontend Developer — Built the React UI, interactive dashboards, premium visual design |
| **Prof. Priyanka Mangi** | Project Guide |

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (React + Vite)                           │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│   │   Home   │ │ Released │ │ Insights │ │ Upcoming │ │  About   │   │
│   │   Page   │ │  Movies  │ │   Page   │ │Dashboard │ │   Us     │   │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│              Axios HTTP Client  ←→  React Router                       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ HTTP/REST (port 5173 → 5000)
┌──────────────────────────────▼─────────────────────────────────────────┐
│                    NODE.JS SERVER (Express, port 5000)                  │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │
│   │  /movies   │ │  /talents  │ │  /trends   │ │ /predictions       │ │
│   │    API     │ │    API     │ │   API      │ │ (proxy → ML)       │ │
│   └────────────┘ └────────────┘ └────────────┘ └────────────────────┘ │
└──────────────┬─────────────────────────────────────────┬──────────────┘
               │                                         │
               ▼                                         ▼
┌──────────────────────┐              ┌─────────────────────────────────┐
│   MONGODB DATABASE   │              │    ML SERVICE (Flask, port 5001) │
│  ┌────────────────┐  │              │  ┌───────────┐ ┌─────────────┐  │
│  │    Movies      │  │              │  │ Predictor │ │   Trainer   │  │
│  │  Collection    │  │              │  │  Module   │ │   Module    │  │
│  │ (1,600+ docs)  │  │              │  └───────────┘ └─────────────┘  │
│  └────────────────┘  │              │  ┌───────────┐ ┌─────────────┐  │
│                      │              │  │ Revenue   │ │ Classifier  │  │
│                      │              │  │ Ensemble  │ │  Ensemble   │  │
│                      │              │  │ (RF+XGB   │ │ (GB+XGB    │  │
│                      │              │  │  +LGBM)   │ │  +LGBM)    │  │
│                      │              │  └───────────┘ └─────────────┘  │
└──────────────────────┘              └─────────────────────────────────┘
```

### Directory Structure

```
ml-web-app-project/
├── client/                          # React Frontend (Vite)
│   ├── src/
│   │   ├── assets/                  # Images (team photos, react.svg)
│   │   ├── components/
│   │   │   ├── common/              # Shared UI components
│   │   │   │   ├── Navbar.jsx/css   # Top navigation bar
│   │   │   │   ├── Footer.jsx/css   # Page footer
│   │   │   │   ├── FilterFab.jsx/css # Floating filter button
│   │   │   │   └── TalentSearch.jsx/css # Director/Cast search
│   │   │   └── movies/
│   │   │       └── MovieCard.jsx/css # Movie card component
│   │   ├── pages/                   # Main page components
│   │   │   ├── Home.jsx/css         # Landing page (marquee capabilities)
│   │   │   ├── ReleasedMovies.jsx/css # Movie library
│   │   │   ├── MovieDetail.jsx/css  # Individual movie page
│   │   │   ├── Insights.jsx/css     # Analytics dashboard
│   │   │   ├── UpcomingDashboard.jsx/css # Prediction engine (86KB!)
│   │   │   └── About.jsx/css        # About Us page
│   │   ├── services/
│   │   │   └── api.js               # Axios API service layer
│   │   ├── App.jsx                  # Root component + routing
│   │   └── index.css                # Global styles + fonts
│   └── index.html
│
├── server/                          # Node.js Backend (Express)
│   ├── src/
│   │   ├── routes/
│   │   │   ├── movies.js            # CRUD for movies (8KB)
│   │   │   ├── talents.js           # Director/Cast search (9KB)
│   │   │   ├── predictions.js       # ML proxy endpoints (10KB)
│   │   │   └── trends.js            # Analytics data endpoints (11KB)
│   │   ├── models/
│   │   │   └── Movie.js             # Mongoose schema
│   │   ├── config/
│   │   │   └── db.js                # MongoDB connection
│   │   └── app.js                   # Express server entry
│   └── .env                         # MONGODB_URI, ML_SERVICE_URL
│
├── ml-service/                      # Python ML Service (Flask)
│   ├── app/
│   │   ├── __init__.py              # Flask app factory
│   │   ├── routes.py                # API endpoints (/predict, /explain, etc.)
│   │   ├── predictor.py             # Loads models & makes predictions (31KB)
│   │   ├── trainer.py               # Trains models from MongoDB data (35KB)
│   │   └── models/                  # Saved model files (.pkl, .json)
│   │       ├── revenue_model.pkl    # Trained revenue ensemble (~24MB)
│   │       ├── classifier_model.pkl # Trained classification ensemble (~8MB)
│   │       ├── scaler.pkl           # StandardScaler
│   │       ├── label_encoder.pkl    # Category ↔ number mapping
│   │       ├── feature_columns.pkl  # 34 feature column names
│   │       ├── best_params.json     # Optuna-tuned hyperparameters
│   │       └── training_results.json # Model metrics
│   └── requirements.txt
│
├── data-collection/                 # Data Pipeline
│   ├── collectors/
│   │   ├── tmdb_collector.py        # The Movie Database API
│   │   ├── omdb_collector.py        # Open Movie Database API
│   │   ├── youtube_collector.py     # YouTube Data API v3
│   │   └── bom_crawler.py          # Box Office Mojo scraper
│   └── scripts/                     # Data collection scripts
│
├── ML_DETAILED_REPORT.md            # THIS file
├── PROJECT_REPORT.md                # Overview report
└── README.md                        # Quick start guide
```

### Three Separate Processes

| Process | Port | Technology | What It Does |
|---------|------|------------|-------------|
| **React Client** | 5173 | Vite + React | Serves the user interface |
| **Node.js Server** | 5000 | Express | REST API, connects to MongoDB, proxies to ML |
| **ML Service** | 5001 | Flask | Loads trained models, serves predictions |

---

## 3. Technology Stack

### 3.1 Frontend

| Technology | Purpose | Why We Chose It |
|------------|---------|----------------|
| **React 18** | UI Framework | Component-based, massive ecosystem |
| **Vite** | Build Tool | 10× faster than Create React App |
| **React Router 6** | Navigation | Client-side routing between pages |
| **Axios** | HTTP Client | Promise-based, clean API calls |
| **ApexCharts** | Data Visualization | Interactive charts for Insights page |
| **CSS (vanilla)** | Styling | Full control, no framework lock-in |

### 3.2 Backend

| Technology | Purpose | Why We Chose It |
|------------|---------|----------------|
| **Node.js 18** | Server Runtime | JavaScript everywhere, async I/O |
| **Express 4** | Web Framework | Minimal, flexible, well-documented |
| **MongoDB 7** | Database | Schema-flexible for varied movie data |
| **Mongoose 8** | ODM | Schema validation + query builder |
| **CORS** | Cross-Origin | Allow React (5173) → Express (5000) |
| **Axios** | HTTP Client | Proxy requests to ML service |

### 3.3 Machine Learning

| Technology | Purpose | Why We Chose It |
|------------|---------|----------------|
| **Python 3.11** | ML Runtime | Industry standard for ML |
| **Flask 3** | API Server | Lightweight, easy to integrate |
| **Scikit-learn** | ML Models | RandomForest, GradientBoosting, StandardScaler |
| **XGBoost** | Gradient Boosting | #1 in ML competitions |
| **LightGBM** | Gradient Boosting | Fastest implementation, leaf-wise growth |
| **Optuna** | Hyperparameter Tuning | Automated, intelligent search |
| **SHAP** | Explainability | Game-theory based feature importance |
| **Pandas** | Data Processing | DataFrame operations |
| **NumPy** | Numerical Computing | Array math, log transforms |

### 3.4 Data Collection

| Technology | Purpose | Data Collected |
|------------|---------|---------------|
| **TMDB API** | Movie metadata | Title, budget, revenue, cast, crew, genres |
| **OMDB API** | External ratings | IMDb rating, Metascore, Rotten Tomatoes |
| **YouTube API v3** | Pre-release buzz | Trailer views, likes, comments |
| **BeautifulSoup** | Web scraping | Box Office Mojo opening weekend data |

### 3.5 Design & Fonts

| Element | Value |
|---------|-------|
| **Primary Font** | TT Firs Neue |
| **Secondary Font** | Syne |
| **Primary Color** | `#ff8300` (Orange) |
| **Background** | `#000000` (Black) |
| **Surface Color** | `#111111` / `#1a1a1a` |
| **Text Color** | `#ffffff` / `#999999` |

---

## 4. Data Collection Pipeline

### 4.1 Data Sources & Coverage

| Source | Data Collected | Movies Enriched |
|--------|----------------|----------------|
| **TMDB API** | Title, budget, revenue, cast, crew, genres, release date, poster, backdrop | 1,600+ |
| **OMDB API** | IMDb rating, Metascore, Rotten Tomatoes scores, awards | ~1,000 |
| **YouTube API** | Official trailer views, likes, comments | ~285 |
| **Box Office Mojo** | Opening weekend gross, distributor | ~1,078 |
| **Indian Cinema** | Bollywood, Tollywood, Kollywood, Mollywood data | ~460 |

### 4.2 Data Collection Architecture

```
                        ┌─────────────────┐
                        │   TMDB API      │ ─── Primary source
                        │  (tmdbId, title,│     1,600+ movies
                        │   budget, etc.) │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┬────────────┐
                    ▼            ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
             │ OMDB API │ │ YouTube  │ │ Box Off. │ │  Indian  │
             │ (IMDb,   │ │ API v3   │ │  Mojo    │ │  Cinema  │
             │Metascore)│ │(trailer) │ │(opening) │ │  Data    │
             └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
                  │            │            │            │
                  └────────────┴────────┬───┴────────────┘
                                        ▼
                              ┌──────────────────┐
                              │   MongoDB Atlas   │
                              │  movies collection│
                              │   (1,600+ docs)   │
                              └──────────────────┘
```

### 4.3 MongoDB Document Schema

```javascript
{
  _id: ObjectId,
  tmdbId: Number,              // TMDB unique identifier
  title: String,               // "Avengers: Endgame"
  originalTitle: String,
  year: Number,                // 2019
  releaseDate: Date,           // 2019-04-24

  // Financial Data
  budget: Number,              // 356000000
  revenue: Number,             // 2797501328

  // Ratings & Metrics
  voteAverage: Number,         // 8.3 (TMDB 0-10)
  voteCount: Number,           // 28543
  popularity: Number,          // 165.3

  // Content
  genres: [String],            // ["Action", "Adventure", "Drama"]
  overview: String,            // Plot synopsis
  runtime: Number,             // 181 minutes
  posterPath: String,          // TMDB poster URL
  backdropPath: String,        // TMDB backdrop URL
  isSequel: Boolean,           // true

  // Credits
  director: {
    name: String,              // "Anthony Russo"
    popularity: Number,        // 45.2
    knownFor: [String]
  },
  cast: [{
    name: String,              // "Robert Downey Jr."
    character: String,         // "Tony Stark / Iron Man"
    popularity: Number,        // 92.1
    profilePath: String        // Photo URL
  }],

  // Industry
  industry: String,            // "hollywood" | "bollywood" | "tollywood" | etc.

  // Enriched: OMDB
  imdbId: String,
  imdbRating: Number,          // 8.4
  metascore: Number,           // 78
  rottenTomatoesScore: Number, // 94
  rottenTomatoesAudienceScore: Number,

  // Enriched: YouTube
  socialMetrics: {
    trailerViews: Number,      // 200000000
    trailerLikes: Number,      // 5000000
    trailerComments: Number    // 150000
  },

  // Enriched: Box Office Mojo
  releaseStrategy: {
    openingWeekendRevenue: Number,  // 357115007
    distributor: String             // "Walt Disney"
  },

  // Production
  productionCompanies: [{ name: String }],

  // ML Predictions (stored after batch-predict)
  predictions: {
    successCategory: String,   // "Blockbuster"
    predictedRevenue: Number,  // 2500000000
    predictedROI: Number,      // 602
    confidence: Number         // 78
  },

  createdAt: Date,
  updatedAt: Date
}
```

### 4.4 How Each Collector Works

**TMDB Collector** (`tmdb_collector.py`):
```python
# Fetches movies by industry, year range, and page
# Uses TMDB's /discover/movie endpoint with filters
# Enriches with /movie/{id}/credits for cast & crew
# Stores in MongoDB with industry tag
```

**OMDB Collector** (`omdb_collector.py`):
```python
# Takes IMDb IDs from TMDB data
# Calls OMDB API for each movie
# Extracts: IMDb rating, Metascore, Rotten Tomatoes
# Updates existing MongoDB documents
```

**YouTube Collector** (`youtube_collector.py`):
```python
# Searches YouTube for "{movie_title} official trailer"
# Uses YouTube Data API v3 to get video statistics
# Extracts: viewCount, likeCount, commentCount
# Updates MongoDB with socialMetrics field
```

**Box Office Mojo Crawler** (`bom_crawler.py`):
```python
# Scrapes Box Office Mojo pages using BeautifulSoup
# Extracts opening weekend gross and distributor
# Updates MongoDB with releaseStrategy field
```

---

# PART B: MACHINE LEARNING (THE BRAIN)

---

## 5. ML Problem Definition

### Two ML Tasks

**Task 1: Revenue Prediction (Regression)**
- **Input:** 34 movie features (budget, genre, cast, director, etc.)
- **Output:** Predicted worldwide box office revenue ($)
- **Type:** Supervised Regression
- **Question:** *"How much money will this movie make?"*

**Task 2: Success Classification (Multi-class)**
- **Input:** Same 34 features
- **Output:** One of 5 categories
- **Type:** Supervised Multi-class Classification
- **Question:** *"Will this movie be a hit or a flop?"*

### Success Categories (Based on ROI)

ROI = Return on Investment = `(Revenue - Budget) / Budget × 100`

| Category | ROI Threshold | Real-World Example |
|----------|---------------|-------------------|
| **Blockbuster** | ROI ≥ 400% | Avatar ($237M → $2.9B) |
| **Super Hit** | ROI ≥ 200% | John Wick ($20M → $86M) |
| **Hit** | ROI ≥ 100% | Movie doubles its budget |
| **Average** | ROI ≥ 0% | Breaks even or small profit |
| **Flop** | ROI < 0% | Lost money |

```python
SUCCESS_CATEGORIES = {
    'Blockbuster': 400,  # ROI >= 400%
    'Super Hit': 200,    # ROI >= 200%
    'Hit': 100,          # ROI >= 100%
    'Average': 0,        # ROI >= 0%
    'Flop': -100         # ROI < 0%
}
```

### Why Machine Learning (Not Simple Rules)?

| Challenge | How ML Solves It |
|-----------|-----------------|
| Too many factors (34 features) | ML considers all simultaneously |
| Complex interactions | Tree-based models capture non-linear patterns |
| Industry differences | One-hot encoded industry features learn per-industry patterns |
| Pre-release signals | YouTube trailer data provides early indicators |
| Changing trends | Model learns from historical patterns |

---

## 6. Feature Engineering: The Secret Sauce

**Feature engineering** = transforming raw movie data into numerical values ML models can understand. Models only understand numbers.

### Our 34 Features (Grouped)

#### Group 1: Basic Numeric (6 features)

| Feature | What It Is | Why It Matters |
|---------|-----------|----------------|
| `budget` | Production budget (USD) | #1 predictor of revenue |
| `runtime` | Movie length (minutes) | Optimal length affects performance |
| `vote_average` | TMDB rating (0-10) | Quality signal |
| `vote_count` | Number of ratings | Proxy for popularity |
| `popularity` | TMDB popularity score | Composite interest signal |
| `year` | Release year | Inflation, streaming trends |

```python
features['budget'] = movies_df['budget'].fillna(0)
features['runtime'] = movies_df['runtime'].fillna(120)    # Default: 2 hours
features['vote_average'] = movies_df['voteAverage'].fillna(6.0)  # Default: average
```

#### Group 2: Temporal/Seasonal (3 features)

| Feature | Calculation | Why |
|---------|------------|-----|
| `release_month` | From release date (1-12) | Summer blockbusters vs Oscar dramas |
| `is_summer_release` | 1 if month in [5,6,7,8] | Summer = action/family season |
| `is_holiday_release` | 1 if month in [11,12] | Holiday = family + awards push |

```python
features['is_summer_release'] = features['release_month'].apply(
    lambda x: 1 if x in [5, 6, 7, 8] else 0
)
```

#### Group 3: Genre (One-Hot Encoded, 9 features)

| Feature | Value |
|---------|-------|
| `genre_action` | 1 if Action, else 0 |
| `genre_comedy` | 1 if Comedy, else 0 |
| `genre_drama` | 1 if Drama, else 0 |
| `genre_horror`, `genre_thriller`, `genre_science_fiction`, `genre_animation`, `genre_romance`, `genre_adventure` | Same pattern |

> **One-Hot Encoding:** We can't give a model the word "Action." Instead, we create a separate binary column for each genre. A movie can have multiple 1s (e.g., Action-Adventure).

```python
popular_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller',
                  'Science Fiction', 'Animation', 'Romance', 'Adventure']

for genre in popular_genres:
    features[f'genre_{genre.lower()}'] = movies_df['genres'].apply(
        lambda x: 1 if isinstance(x, list) and genre in x else 0
    )
```

#### Group 4: Industry (One-Hot Encoded, ~4 features)

| Feature | Meaning |
|---------|---------|
| `industry_hollywood` | 1 if Hollywood |
| `industry_bollywood` | 1 if Bollywood |
| `industry_tollywood` | 1 if Tollywood |
| `industry_kollywood` | 1 if Kollywood |

```python
industry_dummies = pd.get_dummies(features['industry'], prefix='industry')
features = pd.concat([features, industry_dummies], axis=1)
```

#### Group 5: Talent (3 features)

| Feature | How Calculated | Why |
|---------|---------------|-----|
| `director_popularity` | TMDB popularity score | Famous directors draw audiences |
| `cast_popularity` | Average top cast TMDB popularity | Star power sells tickets |
| `is_sequel` | 1 if sequel/franchise | Built-in audience |

```python
def get_cast_popularity(cast):
    if isinstance(cast, list) and len(cast) > 0:
        pops = [c.get('popularity', 0) for c in cast if isinstance(c, dict)]
        return np.mean(pops) if pops else 0
    return 0

features['cast_popularity'] = movies_df['cast'].apply(get_cast_popularity)
```

#### Group 6: Pre-Release Buzz — YouTube (5 features)

| Feature | What | Why |
|---------|------|-----|
| `trailer_views` | Raw view count | Direct anticipation measure |
| `trailer_likes` | Like count | Positive sentiment |
| `trailer_comments` | Comment count | Engagement level |
| `trailer_views_log` | `log(views + 1)` | Compressed scale |
| `trailer_engagement_ratio` | `(likes + comments) / views × 100` | Quality of engagement |

> **Why engagement ratio?** 10M views + 1M likes (10% engagement) > 100M views + 1M likes (1% engagement)

#### Group 7: External Ratings (3 features)

| Feature | Source | Why |
|---------|--------|-----|
| `imdb_rating` | OMDB API | Professional + audience consensus |
| `metascore` | OMDB API | Critical review score |
| `opening_gross` | Box Office Mojo | First-weekend predicts total |

### Why Log-Transform Revenue?

Revenue distribution is extremely **skewed** — most movies earn $1M-$100M, but outliers like Avatar earn $2.8B. Without log transform, the model focuses on outliers.

```
$1M        → log(1,000,000)     = 13.8
$100M      → log(100,000,000)   = 18.4
$1B        → log(1,000,000,000) = 20.7
$2.8B      → log(2,800,000,000) = 21.8
```

Now the range is 13.8–21.8 instead of $1M–$2.8B — much easier for the model!

```python
# Training: compress
log_revenue = np.log1p(revenue)  # log(x + 1) to handle $0

# Prediction: decompress
revenue_pred = np.expm1(log_revenue_pred)  # e^x - 1
```

---

## 7. Model Selection & Training

### Algorithm 1: RandomForest

**What:** Ensemble of hundreds of decision trees, each trained on random data/feature subsets. Final = average of all trees.

**Why:** Resistant to overfitting, handles non-linear relationships, provides feature importance.

```
Tree 1: IF budget > $100M AND is_sequel=1 → predict $500M
Tree 2: IF cast_popularity > 50 AND genre_action=1 → predict $400M
Tree 3: IF trailer_views > 10M AND is_summer=1 → predict $600M
Final = average(500, 400, 600) = $500M
```

**Our tuned hyperparameters (found by Optuna):**
```python
RandomForestRegressor(
    n_estimators=200,      # 200 trees
    max_depth=14,          # Max 14 levels deep
    min_samples_split=2,   # Min 2 samples to split
    min_samples_leaf=1,    # Min 1 sample per leaf
    max_features=None,     # Consider ALL features
    random_state=42
)
```

### Algorithm 2: XGBoost (eXtreme Gradient Boosting)

**What:** Builds trees sequentially — each new tree fixes errors of previous trees. "Boosting" = learning from mistakes.

**Why:** Often #1 in ML competitions, built-in regularization, handles missing values.

```
Tree 1: predict $300M (error: actual was $500M, error = -$200M)
Tree 2: try to predict -$200M error → -$150M
Tree 3: predict remaining -$50M → -$40M
Final: $300M + $150M + $40M = $490M (much closer!)
```

**Our tuned hyperparameters:**
```python
XGBRegressor(
    n_estimators=450,        # 450 sequential trees
    max_depth=7,             # Shallower (prevent overfitting)
    learning_rate=0.0339,    # Small step = conservative learning
    subsample=0.912,         # Use 91% data per tree
    colsample_bytree=0.749,  # Use 75% features per tree
    reg_alpha=0.000111,      # L1 regularization (lasso)
    reg_lambda=1.359         # L2 regularization (ridge)
)
```

### Algorithm 3: LightGBM

**What:** Like XGBoost but faster. Uses "leaf-wise" growth (grows most impactful leaf first).

**Why:** Fastest gradient boosting, often outperforms XGBoost, **achieved our best R² = 0.9340**.

```python
LGBMRegressor(
    n_estimators=300,        # 300 trees
    max_depth=7,
    learning_rate=0.0185,    # Very conservative
    num_leaves=30,           # Max leaves per tree (LightGBM-specific)
    subsample=0.728,
    colsample_bytree=0.962,
    reg_alpha=1.17e-08,
    reg_lambda=1.76e-05
)
```

> **`num_leaves`** is unique to LightGBM. Controls how many leaf nodes (predictions) each tree can make.

### Classification Models

| Model | Type | Role |
|-------|------|------|
| GradientBoostingClassifier | Sklearn | Baseline |
| XGBClassifier | XGBoost | Tuned with Optuna |
| LGBMClassifier | LightGBM | Tuned with Optuna |

---

## 8. Hyperparameter Tuning with Optuna

### What Are Hyperparameters?
- **Parameters:** Learned during training (decision boundaries)
- **Hyperparameters:** Set BEFORE training (how many trees, how deep)

### What is Optuna?
Optuna is an **automated hyperparameter optimization framework** using TPE (Tree-structured Parzen Estimator) — it learns from previous trials which regions are most promising.

```
Trial 1:  n_estimators=150, max_depth=5  → R² = 0.89
Trial 2:  n_estimators=300, max_depth=10 → R² = 0.91
Trial 3:  n_estimators=250, max_depth=8  → R² = 0.92  ← promising!
...
Trial 50: n_estimators=200, max_depth=14 → R² = 0.934 ← BEST!
```

**50 trials per model × 5 models = 250 total experiments**

```python
def tune_rf_regressor(self, X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params
```

---

## 9. Ensemble Learning

### Why Combine Models?

| Model | Strength | Weakness |
|-------|----------|----------|
| RandomForest | Stable, resistant to outliers | Can underfit complex patterns |
| XGBoost | Great at capturing patterns | Can overfit on small data |
| LightGBM | Fast, handles diverse features | Sensitive to noise |

### Revenue: Weighted VotingRegressor

```
Input: Movie features
    ┌────────┬────────────┐
    ▼        ▼            ▼
RandomForest  XGBoost    LightGBM
  $450M       $480M      $520M
  × 0.332   × 0.332     × 0.336  ← weights from R² scores
         Weighted Average = $483.5M
```

```python
total_r2 = rf_r2 + xgb_r2 + lgbm_r2
weights = [rf_r2/total_r2, xgb_r2/total_r2, lgbm_r2/total_r2]

ensemble = VotingRegressor(
    estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    weights=weights
)
```

### Classification: Soft VotingClassifier

Uses **probability averaging** instead of majority vote:

```python
ensemble_cls = VotingClassifier(
    estimators=[('gb', gb_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft'  # Average probabilities, not hard votes
)
```

---

## 10. Cross-Validation & Evaluation

### 5-Fold Cross-Validation

```
Fold 1: [TEST] [Train] [Train] [Train] [Train]  → Score: 0.92
Fold 2: [Train] [TEST] [Train] [Train] [Train]  → Score: 0.91
Fold 3: [Train] [Train] [TEST] [Train] [Train]  → Score: 0.93
Fold 4: [Train] [Train] [Train] [TEST] [Train]  → Score: 0.90
Fold 5: [Train] [Train] [Train] [Train] [TEST]  → Score: 0.92
Final = 0.916 ± 0.031
```

### Stratified K-Fold (for Classification)

Ensures each fold has the **same class proportions** — critical for imbalanced datasets.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_cls, X_train, y_train, cv=skf, scoring='accuracy')
```

---

## 11. Model Performance & Results

### Revenue Prediction

| Model | R² Score | MAE (log) | Improvement |
|-------|----------|-----------|-------------|
| RandomForest (old baseline) | 0.9083 | 0.42 | — |
| **RandomForest** (Optuna) | **0.9210** | 0.2984 | +1.4% |
| **XGBoost** (Optuna) | **0.9229** | 0.2995 | +1.6% |
| **LightGBM** (Optuna) | **0.9340** 🏆 | 0.2828 | **+2.8%** |
| **Ensemble** (weighted) | **0.9294** | 0.2852 | +2.3% |

> **R² = 0.934** means the model explains **93.4%** of revenue variance. MAE = 0.28 in log space ≈ ±32% in dollars.

### Classification

| Model | Accuracy | Notes |
|-------|----------|-------|
| GradientBoosting | 55.9% | Original baseline |
| XGBoost (tuned) | **57.4%** | Best individual |
| LightGBM (tuned) | **57.4%** | Tied for best |
| Ensemble (soft) | 56.5% | Combined |
| 5-Fold CV | **54.7% ± 6.4%** | Honest evaluation |

> **Why 57% is good for 5 classes:** Random guessing = 20%. Our model = 2.85× better. Adjacent categories (Hit vs Super Hit) overlap — the model rarely confuses Blockbusters with Flops.

---

## 12. SHAP Explainability

**SHAP (SHapley Additive exPlanations)** explains individual predictions — which features pushed prediction up or down, and by how much.

```
Prediction: "Hit" ($350M)

Why?
  budget = $150M         → UP by $120M    ████████████
  cast_popularity = 65   → UP by $50M     █████
  genre_action = 1       → UP by $30M     ███
  is_summer_release = 1  → UP by $20M     ██
  trailer_views = 5M     → DOWN by -$10M  █ (low)
  is_sequel = 0          → DOWN by -$15M  ██ (not sequel)
```

### Top 10 Most Important Features

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | `budget` | Highest |
| 2 | `trailer_views_log` | High |
| 3 | `popularity` | Medium-High |
| 4 | `vote_count` | Medium |
| 5 | `cast_popularity` | Medium |
| 6 | `director_popularity` | Medium |
| 7 | `vote_average` | Medium |
| 8 | `runtime` | Low-Medium |
| 9 | `is_summer_release` | Low |
| 10 | `genre_action` | Low |

---

## 13. Prediction Flow: Input to Output

### What Happens When a User Clicks "Predict"

```
1. USER fills form:
   Title: "Avengers 5", Budget: $250M, Genre: Action+Adventure
   Industry: Hollywood, Director: 85, Cast: 90, Month: July, Sequel: Yes

2. REACT → POST http://localhost:5000/api/predictions/predict
   Body: { budget: 250000000, genres: ["Action","Adventure"], ... }

3. NODE.JS proxies → POST http://localhost:5001/api/predict

4. PREDICTOR.PY:
   a) _prepare_features(data):
      Creates 34-feature dict, fills values, handles missing data
   b) scaler.transform(features):
      budget 250M → z-score ~2.1 (2.1 std devs above mean)
   c) revenue_model.predict():
      RF: 21.2, XGB: 21.5, LGBM: 21.3 → Weighted avg: 21.33
      Convert: expm1(21.33) = $1,837,000,000
   d) classifier_model.predict():
      → "Blockbuster" (78% confidence)
   e) Calculate ROI:
      ($1.837B - $250M) / $250M × 100 = 635%

5. RESPONSE:
   { successCategory: "Blockbuster", predictedRevenue: 1837000000,
     predictedROI: 635, confidence: 78, featureImportance: [...] }

6. REACT renders prediction card with animated badge
```

---

## 14. Model Serialization

### What We Save (Pickle Files)

| File | Contents | Size |
|------|----------|------|
| `revenue_model.pkl` | VotingRegressor (RF+XGB+LGBM) | ~24 MB |
| `classifier_model.pkl` | VotingClassifier (GB+XGB+LGBM) | ~8 MB |
| `scaler.pkl` | StandardScaler (mean/std per feature) | ~2 KB |
| `label_encoder.pkl` | LabelEncoder (category ↔ number) | ~300 B |
| `feature_columns.pkl` | List of 34 feature names | ~600 B |
| `best_params.json` | Optuna's winning hyperparameters | ~1.2 KB |
| `training_results.json` | All metrics + CV scores | ~640 B |

```python
# Save
with open('revenue_model.pkl', 'wb') as f:
    pickle.dump(self.revenue_model, f)

# Load (happens once at server startup)
with open('revenue_model.pkl', 'rb') as f:
    self.revenue_model = pickle.load(f)
```

> **Why not retrain every time?** Training takes ~14 minutes. Loading a pickle takes <1 second.

---

# PART C: BACKEND (THE BACKBONE)

---

## 15. Node.js Server

### Server Structure

The Express server acts as the **API gateway** — it serves data from MongoDB and proxies ML requests to the Flask service.

### API Endpoints

#### Movies API (`/api/movies`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies` | List movies with filters, search, pagination |
| GET | `/api/movies/:id` | Get single movie with full details |
| GET | `/api/movies/stats` | Aggregated statistics (counts, averages) |
| GET | `/api/movies/filters` | Available filter options |

**Query Parameters:**
- `industry` — Filter by Hollywood, Bollywood, etc.
- `genre` — Filter by genre
- `category` — Filter by success category
- `search` — Text search on title
- `sort` — Sort by popularity, revenue, rating, release date
- `page`, `limit` — Pagination

#### Talents API (`/api/talents`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/talents/search?q=&type=` | Search directors/actors by name |

Uses MongoDB text search + regex for fuzzy matching. Returns TMDB popularity, known movies, and profile photos.

#### Predictions API (`/api/predictions`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predictions/predict` | Proxy to ML service `/api/predict` |
| POST | `/api/predictions/explain` | Proxy to ML service `/api/explain` |
| POST | `/api/predictions/optimal-release` | Optimal release month |
| POST | `/api/predictions/competition` | Competition analysis |
| POST | `/api/predictions/whatif` | What-if simulation |

#### Trends API (`/api/trends`) — Powers the Insights Page

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/trends/yearly` | Industry trends over time |
| GET | `/api/trends/genres` | Genre distribution analysis |
| GET | `/api/trends/seasonal` | Monthly performance patterns |
| GET | `/api/trends/talent/:type` | Top directors/actors by revenue |
| GET | `/api/trends/regional` | Indian cinema comparison |
| GET | `/api/trends/budget-revenue` | Budget vs Revenue scatter data |
| GET | `/api/trends/youtube-hype` | Trailer views vs box office |
| GET | `/api/trends/production-houses` | Production company rankings |
| GET | `/api/trends/opening-weekend` | Opening weekend strength |
| GET | `/api/trends/critic-audience` | Critic vs Audience score gap |

---

## 16. ML Service (Flask)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict` | Revenue + category prediction |
| POST | `/api/explain` | SHAP feature importance |
| POST | `/api/optimal-release` | Best release month analysis |
| POST | `/api/batch-predict` | Predict multiple movies |

### Key Modules

**`predictor.py` (31KB):**
- Loads all `.pkl` model files at startup
- `predict(data)` — Full prediction pipeline
- `explain(data)` — SHAP-based explanation
- `get_optimal_release(data)` — Monthly revenue analysis
- `_prepare_features(data)` — Creates 34-feature vector from raw input

**`trainer.py` (35KB):**
- Connects to MongoDB, fetches all movies
- Engineers 34 features from raw data
- Tunes hyperparameters with Optuna (50 trials × 5 models)
- Trains individual + ensemble models
- Evaluates with 5-fold CV
- Saves everything to `.pkl` and `.json`

---

# PART D: FRONTEND (THE FACE)

---

## 17. Design System & Aesthetics

### Design Philosophy
GreenLit GO uses a **cinematic, premium dark theme** inspired by movie industry aesthetics — dark backgrounds, orange accents, and modern typography.

### Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| `--primary` | `#ff8300` | Orange accent, CTAs, highlights |
| `--bg-primary` | `#000000` | Page backgrounds |
| `--bg-card` | `#111111` | Card backgrounds |
| `--bg-elevated` | `#1a1a1a` | Elevated surfaces, borders |
| `--text-primary` | `#ffffff` | Headings, primary text |
| `--text-secondary` | `#999999` | Body text, descriptions |
| `--text-muted` | `#666666` | Labels, hints |

### Typography
- **TT Firs Neue** — Primary font for headings and body
- **Syne** — Secondary font for special headings
- Font imported via `@font-face` in `index.css`

### Animation Patterns
- **Scroll-reveal:** Elements fade up as user scrolls (IntersectionObserver)
- **Marquee:** Infinite horizontal scrolling (CSS keyframes)
- **Hover effects:** Cards lift, glow, scale on hover
- **Count-up:** Numbers animate from 0 to target value

---

## 18. Home Page

**Files:** `Home.jsx` (10KB), `Home.css` (12KB)

### Sections

1. **Hero Section**
   - Animated title: "DATA-DRIVEN BLOCKBUSTERS"
   - Subtitle with typing effect
   - CTA buttons: "Predict Now" → Upcoming Dashboard, "Explore Library" → Released Movies

2. **Stats Bar**
   - 4 animated counters: 1,600+ Movies | 93% Accuracy | 2015-25 Range | 5+ Industries
   - Uses custom `useCountUp` hook for smooth number animation

3. **Capabilities — Marquee Grid**
   - **10 capability cards** in 2 infinite-scrolling rows (CSS `@keyframes`):
     - Row 1: Revenue Prediction, Success Classification, Explainable AI, What-If Simulator, Competition Analyzer
     - Row 2: Multi-Industry Support, Optimal Release Timing, Trend Analytics, Smart Dashboard, AI-Powered Insights
   - Rows scroll in **opposite directions** for visual effect
   - Cards pause on hover and show lift + orange glow animation
   - Pure CSS — no JS animation library needed

4. **CTA Section**
   - "Ready to Predict Your Next Blockbuster?" with button linking to Upcoming Dashboard

---

## 19. Released Movies Library

**Files:** `ReleasedMovies.jsx` (11KB), `ReleasedMovies.css` (10KB)

### Features

- **Netflix/IMDB-style movie grid** with poster cards
- **FAB (Floating Action Button) Filter** — expandable filter panel:
  - Industry: All, Hollywood, Bollywood, Tollywood, Kollywood, Mollywood
  - Genre: Dropdown with all genres
  - Success Category: Blockbuster, Super Hit, Hit, Average, Flop
  - Sort: Popularity, Revenue, Rating, Release Date
- **Text search** across movie titles
- **Infinite scroll / pagination**
- **Movie cards** show: poster, title, year, rating badge, success category badge
- Click → navigates to Movie Detail page

---

## 20. Movie Detail Page

**Files:** `MovieDetail.jsx` (13KB), `MovieDetail.css` (16KB)

### Sections

1. **Hero Banner** — Full-width backdrop image with gradient overlay
2. **Key Metrics** — Budget, Revenue, ROI in large stat cards
3. **Success Badge** — Animated category badge (Blockbuster/Hit/etc.)
4. **Cast Section** — Profile photos, character names, popularity scores
5. **Movie Info** — Runtime, release date, genres as tags, overview text
6. **Ratings** — TMDB, IMDb, Metascore, Rotten Tomatoes (when available)

---

## 21. Insights Page (Data Analytics)

**Files:** `Insights.jsx` (52KB), `Insights.css` (11KB)

This is the **analytics dashboard** powered by ApexCharts — 8+ interactive visualization panels:

### Charts

| Chart | Type | What It Shows |
|-------|------|--------------|
| **Revenue Trends** | Line chart | Revenue over years, filterable by industry |
| **Genre Distribution** | Bar/Treemap | Most common genres and their avg revenue |
| **Seasonal Trends** | Area chart | Monthly revenue patterns (summer vs winter) |
| **Budget vs Revenue** | Scatter plot | Correlation between investment and return |
| **YouTube Hype vs Box Office** | Scatter plot | Trailer views → actual revenue |
| **Production Houses** | Horizontal bar | Top studios ranked by total revenue |
| **Opening Weekend** | Scatter | Opening weekend % of total gross |
| **Critic vs Audience** | Diverging bar | Gap between RT critics and audience scores |
| **Top Directors** | Table/bar | Highest-grossing directors |
| **Regional Comparison** | Bar | Indian cinema: Bollywood vs Tollywood vs Kollywood |

### Features
- **Industry filter** at the top (All, Hollywood, Bollywood, etc.)
- **Interactive tooltips** on every data point
- **Responsive** — charts resize for all screen sizes
- **Dark theme** matching the app's aesthetic

---

## 22. Upcoming Dashboard (Prediction Engine)

**Files:** `UpcomingDashboard.jsx` (86KB!), `UpcomingDashboard.css` (30KB)

This is the **largest and most complex page** — the core ML prediction interface.

### Sections

1. **Prediction Form**
   - Movie title input
   - Industry selection (5 industries with flags)
   - Genre selection (multi-select chip UI)
   - Budget input with currency formatting
   - Runtime input
   - Release month selector
   - Sequel toggle

2. **Talent Search**
   - **Director search** with TMDB auto-complete (TalentSearch component)
   - **Cast search** with multi-select, shows popularity score for each selected actor
   - Average cast popularity auto-calculated

3. **Prediction Results Panel** (appears after predict)
   - **Success category badge** with animation
   - **Predicted revenue** with formatting
   - **Predicted ROI** percentage
   - **Confidence score** with progress bar
   - **SHAP Feature Importance** bar chart

4. **Optimal Release Timing**
   - Monthly revenue heatmap
   - Best month recommendation
   - Seasonal analysis

5. **Competition Analysis**
   - Movies releasing in same window
   - Threat assessment scores
   - Competition density

6. **What-If Simulator**
   - Sliders for budget, runtime, release month
   - Real-time prediction updates as user adjusts
   - Side-by-side comparison of scenarios

---

## 23. About Us Page

**Files:** `About.jsx` (17KB), `About.css` (14KB)

Premium dark-themed page with 6 sections:

1. **Hero** — "About GreenLit GO" with orange glow effect and "INTRODUCTION" label
2. **Our Mission** — Mission text + 4 stat cards (1,600+ Movies, 5 Industries, 40+ ML Features, 93% Model Accuracy) with SVG icons
3. **Built With Precision** — 8 tech stack cards with **real brand logos** from devicon CDN:
   - React, Node.js, Python, MongoDB, Flask, Scikit-Learn, XGBoost, TMDB API
4. **What We Offer** — Vertical timeline with orange dots, 6 capability descriptions
5. **Meet the Builders** — 3 team cards matching reference design:
   - **Rushit Trambadia** — ML & Backend Developer (with photo)
   - **Vandit Doshi** — Frontend Developer (with photo)
   - **Prof. Priyanka Mangi** — Project Guide (placeholder initials)
6. **CTA Footer** — Orange gradient banner: "Ready to Greenlight Your Next Hit?" with buttons to Predict and Library pages

### Design Features
- Scroll-reveal animations (IntersectionObserver)
- Section dividers between each section
- Fully responsive for all screen sizes

---

## 24. Reusable Components

### Navbar (`Navbar.jsx`)
- Fixed top navigation bar
- Logo "GreenLit **GO**" with orange accent
- Links: Home, Released Movies, Insights, Upcoming Movies, About Us
- Active page highlighting
- Responsive hamburger menu on mobile

### Footer (`Footer.jsx`)
- Multi-column footer with links to all pages
- Social media icons (placeholders)
- Copyright notice
- GreenLit GO branding

### FilterFab (`FilterFab.jsx`)
- **Floating Action Button** that expands into a filter panel
- Used on Released Movies page
- Animated expand/collapse
- Industry, Genre, Category, Sort filters
- Apply/Reset buttons

### TalentSearch (`TalentSearch.jsx`)
- TMDB-powered autocomplete search
- Works for both directors and actors
- Shows profile photos, popularity scores, known movies
- Used in Upcoming Dashboard for director/cast selection

### MovieCard (`MovieCard.jsx`)
- Reusable movie poster card
- Shows: poster image, title, year, rating, success badge
- Hover effects (lift, shadow)
- Click navigates to Movie Detail page

---

# PART E: WRAP-UP

---

## 25. Key Learnings & Insights

### What We Learned About Movie Success
1. **Money predicts money** — Budget is the #1 feature
2. **Trailers matter A LOT** — YouTube views are the #2 feature
3. **Stars still sell tickets** — Director/cast popularity are significant
4. **Timing is modest** — Summer/holiday releases have small advantage
5. **Sequels have an edge** — Built-in audiences reduce risk

### What We Learned About ML
1. **Log transformation is essential** for skewed targets (R² 0.72 → 0.93)
2. **Hyperparameter tuning matters** — Optuna improved R² by +2.8%
3. **Ensemble > Individual** — No single model always wins
4. **Honest evaluation is crucial** — Stratified K-Fold gives real performance
5. **Feature engineering > more data** — 34 engineered features from 1,600 movies outperforms raw data from 10,000

### What We Learned About Web Development
1. **Component architecture** pays off — reusable Navbar, Footer, FilterFab save time
2. **Vanilla CSS > frameworks** — Full control over dark theme, animations, responsive design
3. **Scroll-reveal animations** make pages feel premium with minimal code
4. **ApexCharts** is excellent for React data visualization
5. **Proxy pattern** (Node → Flask) cleanly separates JS and Python worlds

### Technical Decisions & Rationale

| Decision | Why |
|----------|-----|
| RF + XGBoost + LightGBM ensemble | Three complementary tree algorithms |
| 50 Optuna trials per model | Balances quality vs training time (~14 min) |
| Weighted voting | Better models get proportionally more influence |
| Soft voting for classification | Probability averaging > hard majority |
| React + Vite | Fastest build tool, modern React |
| Vanilla CSS | Maximum flexibility for custom dark theme |
| MongoDB | Schema-flexible for varied movie data structures |
| Flask for ML | Python ecosystem + lightweight serving |
| devicon CDN for logos | Real brand logos without bundling image assets |

---

## 26. How to Run the Project

### Prerequisites
- Node.js 18+
- Python 3.11+
- MongoDB (local or Atlas)

### Step 1: Start MongoDB
```bash
mongod
# or use MongoDB Atlas (cloud) with connection string in .env
```

### Step 2: Start ML Service (Port 5001)
```bash
cd ml-service
pip install -r requirements.txt
python -m app
```

### Step 3: Start Node.js Server (Port 5000)
```bash
cd server
npm install
npm run dev
```

### Step 4: Start React Client (Port 5173)
```bash
cd client
npm install
npm run dev
```

### Step 5: Open Browser
Navigate to **http://localhost:5173**

### Environment Variables

**`server/.env`:**
```
MONGODB_URI=mongodb://localhost:27017/greenlit-go
ML_SERVICE_URL=http://localhost:5001
PORT=5000
```

**`data-collection/.env`:**
```
TMDB_API_KEY=your_tmdb_api_key
OMDB_API_KEY=your_omdb_api_key
YOUTUBE_API_KEY=your_youtube_api_key
MONGODB_URI=mongodb://localhost:27017/greenlit-go
```

---

*Report Updated: March 4, 2026*
*Project: GreenLit GO — Movie Success Prediction Platform*
*Team: Rushit Trambadia (ML & Backend) + Vandit Doshi (Frontend)*
*Guide: Prof. Priyanka Mangi*
*Stack: React + Node.js + Python (Scikit-learn + XGBoost + LightGBM + Optuna + SHAP) + MongoDB*
