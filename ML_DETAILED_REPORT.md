# GreenLit GO — Machine Learning: Complete Deep-Dive Report

> Everything you need to know about the ML pipeline — What, Why, When, Where, Who, How (W5H)

---

## Table of Contents

1. [What is the ML Problem?](#1-what-is-the-ml-problem)
2. [Why Machine Learning?](#2-why-machine-learning)
3. [Where Does ML Fit in the System?](#3-where-does-ml-fit-in-the-system)
4. [How the ML Pipeline Works (Step-by-Step)](#4-how-the-ml-pipeline-works-step-by-step)
5. [Data: The Foundation](#5-data-the-foundation)
6. [Feature Engineering: The Secret Sauce](#6-feature-engineering-the-secret-sauce)
7. [Model Selection: Which Algorithms and Why](#7-model-selection-which-algorithms-and-why)
8. [Hyperparameter Tuning with Optuna](#8-hyperparameter-tuning-with-optuna)
9. [Ensemble Learning: Combining Models](#9-ensemble-learning-combining-models)
10. [Cross-Validation: Honest Evaluation](#10-cross-validation-honest-evaluation)
11. [Model Performance & Results](#11-model-performance--results)
12. [SHAP Explainability](#12-shap-explainability)
13. [Prediction Flow: From Input to Output](#13-prediction-flow-from-input-to-output)
14. [Model Serialization: Saving & Loading](#14-model-serialization-saving--loading)
15. [Key Learnings & Insights](#15-key-learnings--insights)

---

## 1. What is the ML Problem?

### Problem Statement
We have two machine learning tasks:

**Task 1: Revenue Prediction (Regression)**
- **Input:** Movie attributes (budget, genre, cast, director, etc.)
- **Output:** Predicted worldwide box office revenue (a number in dollars)
- **Type:** Supervised Regression
- **Question it answers:** *"How much money will this movie make?"*

**Task 2: Success Classification (Multi-class Classification)**
- **Input:** Same movie attributes
- **Output:** One of 5 categories: Blockbuster, Super Hit, Hit, Average, Flop
- **Type:** Supervised Multi-class Classification
- **Question it answers:** *"Will this movie be a hit or a flop?"*

### Success Categories (Based on ROI)

ROI = Return on Investment = `(Revenue - Budget) / Budget × 100`

| Category | ROI Threshold | Real-World Example |
|----------|---------------|-------------------|
| **Blockbuster** | ROI ≥ 400% | Avatar, Avengers ($200M budget → $2B+ revenue) |
| **Super Hit** | ROI ≥ 200% | John Wick ($20M budget → $86M revenue) |
| **Hit** | ROI ≥ 100% | A movie that doubles its budget |
| **Average** | ROI ≥ 0% | Breaks even or small profit |
| **Flop** | ROI < 0% | Lost money (budget > revenue) |

**Code that defines this** (`trainer.py`):
```python
SUCCESS_CATEGORIES = {
    'Blockbuster': 400,  # ROI >= 400%
    'Super Hit': 200,    # ROI >= 200%
    'Hit': 100,          # ROI >= 100%
    'Average': 0,        # ROI >= 0%
    'Flop': -100         # ROI < 0%
}
```

---

## 2. Why Machine Learning?

### Why Not Simple Rules?

You might think: *"Just check if budget is high and cast is popular, then it's a hit."*

**Problems with simple rules:**
1. Budget alone doesn't guarantee success (many expensive movies flopped)
2. There are 34 features interacting in complex ways
3. Patterns change across industries (Bollywood vs Hollywood)
4. Non-linear relationships exist (e.g., very high budgets sometimes indicate risky projects)

### Why ML Works Here

| Challenge | How ML Solves It |
|-----------|-----------------|
| Too many factors | ML considers 34 features simultaneously |
| Complex interactions | Tree-based models capture non-linear patterns |
| Industry differences | One-hot encoded industry features let the model learn per-industry patterns |
| Pre-release signals | YouTube trailer data gives early indicators |

---

## 3. Where Does ML Fit in the System?

### Architecture Position

```
User fills prediction form in React
         │
         ▼
React → Node.js API (port 5000) → proxies to → ML Service (port 5001)
                                                      │
                                                      ▼
                                              ┌──────────────┐
                                              │ predictor.py │
                                              │   loads:     │
                                              │  - models    │
                                              │  - scaler    │
                                              │  - encoder   │
                                              └──────┬───────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │  .pkl files  │
                                              │ (serialized  │
                                              │   models)    │
                                              └──────────────┘
```

### File Structure

```
ml-service/
├── app/
│   ├── __init__.py        ← Flask app factory (creates the web server)
│   ├── routes.py          ← API endpoints (/predict, /explain, etc.)
│   ├── predictor.py       ← Loads models & makes predictions (INFERENCE)
│   ├── trainer.py         ← Trains models from data (TRAINING)
│   └── models/            ← Saved model files
│       ├── revenue_model.pkl       ← Trained revenue ensemble model
│       ├── classifier_model.pkl    ← Trained classification ensemble model
│       ├── scaler.pkl              ← StandardScaler (fitted on training data)
│       ├── label_encoder.pkl       ← Maps category names ↔ numbers
│       ├── feature_columns.pkl     ← List of 34 feature column names
│       ├── best_params.json        ← Optuna-tuned hyperparameters
│       └── training_results.json   ← Model performance metrics
└── requirements.txt
```

### Two Separate Processes

| Process | When | What |
|---------|------|------|
| **Training** (`trainer.py`) | Run manually, occasionally | Reads MongoDB → trains models → saves .pkl files |
| **Inference** (`predictor.py`) | Always running (Flask server) | Loads .pkl files → receives API requests → returns predictions |

---

## 4. How the ML Pipeline Works (Step-by-Step)

### Training Pipeline (What happens when you run `python -c "from app.trainer import MovieModelTrainer; t = MovieModelTrainer(); t.train()"`)

```
Step 1: LOAD DATA
    ↓  Connect to MongoDB, fetch all 1,652 movies
    ↓  Convert to Pandas DataFrame
    ↓
Step 2: FILTER VALID DATA
    ↓  Keep only movies with budget > 0 OR revenue > 0
    ↓  Result: ~1,652 valid movies for training
    ↓
Step 3: FEATURE ENGINEERING
    ↓  Extract 34 numerical features from raw movie data
    ↓  (Explained in detail in Section 6)
    ↓
Step 4: CALCULATE TARGETS
    ↓  Revenue target: log(revenue + 1)  ← log transformation!
    ↓  Category target: Calculate ROI → assign category
    ↓
Step 5: SPLIT DATA
    ↓  80% Training, 20% Testing
    ↓  Classification uses STRATIFIED split
    ↓
Step 6: SCALE FEATURES
    ↓  StandardScaler: transforms each feature to mean=0, std=1
    ↓
Step 7: TUNE HYPERPARAMETERS (Optuna)
    ↓  50 trials per model × 5 models = 250 total experiments
    ↓
Step 8: TRAIN INDIVIDUAL MODELS
    ↓  RandomForest, XGBoost, LightGBM for each task
    ↓
Step 9: CREATE ENSEMBLE
    ↓  Combine all 3 models using weighted voting
    ↓
Step 10: EVALUATE
    ↓  R² score, MAE for regression
    ↓  Accuracy, F1 for classification
    ↓  5-Fold Cross-validation for both
    ↓
Step 11: SAVE
    ↓  Pickle all models, scaler, encoder, feature list
    ↓  Save params and results as JSON
    Done!
```

---

## 5. Data: The Foundation

### What Data Do We Have?

Each movie in MongoDB has these fields that feed the ML:

| Data Point | Source | Example |
|------------|--------|---------|
| Budget | TMDB API | $200,000,000 |
| Revenue | TMDB API | $2,799,439,100 |
| Runtime | TMDB API | 162 minutes |
| Vote Average | TMDB API | 7.8 |
| Vote Count | TMDB API | 28,543 |
| Popularity | TMDB API | 165.3 |
| Release Date | TMDB API | 2019-04-24 |
| Genres | TMDB API | ["Action", "Adventure", "Sci-Fi"] |
| Director | TMDB API | {name: "Russo Brothers", popularity: 45.2} |
| Cast | TMDB API | [{name: "RDJ", popularity: 92.1}, ...] |
| Is Sequel | TMDB API | true |
| Industry | Custom | "hollywood" |
| IMDb Rating | OMDB API | 8.4 |
| Metascore | OMDB API | 78 |
| Opening Gross | Box Office Mojo | $357,115,007 |
| Trailer Views | YouTube API | 200,000,000 |
| Trailer Likes | YouTube API | 5,000,000 |
| Trailer Comments | YouTube API | 150,000 |

### Why Log-Transform Revenue?

Revenue has an extremely **skewed distribution**:

```
Most movies:    $1M - $100M     (many movies here)
Some movies:    $100M - $500M   (fewer)
Few movies:     $500M - $2.8B   (very few outliers like Avatar)
```

Without log transform, the model focuses on predicting Avatar-level outliers and ignores small movies.

**Log transformation** (`log(revenue + 1)`) compresses the scale:

```
$1M        → log(1,000,000)     = 13.8
$100M      → log(100,000,000)   = 18.4
$1B        → log(1,000,000,000) = 20.7
$2.8B      → log(2,800,000,000) = 21.8
```

Now the range is 13.8 to 21.8 instead of $1M to $2.8B — much easier for the model!

**Code:**
```python
# During training
log_revenue = np.log1p(revenue)  # log(x + 1) to handle 0 values

# During prediction (reversing the transform)
revenue_pred = np.expm1(log_revenue_pred)  # e^x - 1
```

---

## 6. Feature Engineering: The Secret Sauce

### What is Feature Engineering?

**Feature engineering** is the process of transforming raw data into numerical values that ML models can understand. Models only understand numbers — they can't read "Action" or "Christopher Nolan".

### Our 34 Features (Grouped)

#### Group 1: Basic Numeric Features (6 features)
These are directly taken from the data:

| Feature | What It Is | Why It Matters |
|---------|-----------|----------------|
| `budget` | Production budget in USD | #1 predictor — big-budget films tend to earn more |
| `runtime` | Movie length in minutes | Very long or short movies may perform differently |
| `vote_average` | TMDB user rating (0-10) | Higher-rated movies tend to earn more |
| `vote_count` | How many users rated | Proxy for popularity/awareness |
| `popularity` | TMDB's popularity score | Composite signal of public interest |
| `year` | Release year | Trends change over time (inflation, streaming) |

**Code:**
```python
features['budget'] = movies_df['budget'].fillna(0)       # If missing, assume 0
features['runtime'] = movies_df['runtime'].fillna(120)    # If missing, assume 2 hours
features['vote_average'] = movies_df['voteAverage'].fillna(6.0)  # Default to average
```

#### Group 2: Temporal/Seasonal Features (3 features)
When a movie releases matters:

| Feature | How It's Calculated | Why |
|---------|-------------------|-----|
| `release_month` | Extracted from release date (1-12) | Summer blockbusters vs Oscar-season dramas |
| `is_summer_release` | 1 if month in [5,6,7,8], else 0 | Summer = action/family movie season |
| `is_holiday_release` | 1 if month in [11,12], else 0 | Holidays = family viewing + awards push |

**Code:**
```python
features['is_summer_release'] = features['release_month'].apply(
    lambda x: 1 if x in [5, 6, 7, 8] else 0
)
```

> **Concept: Binary Encoding** — Instead of month as a number (1-12), we create yes/no features. The model learns "summer release = +X revenue" more easily than "month 7 is like month 6 but different from month 12."

#### Group 3: Genre Features — One-Hot Encoding (9 features)

| Feature | Value |
|---------|-------|
| `genre_action` | 1 if genres contain "Action", else 0 |
| `genre_comedy` | 1 if genres contain "Comedy", else 0 |
| `genre_drama` | 1 if genres contain "Drama", else 0 |
| `genre_horror` | 1 if genres contain "Horror", else 0 |
| `genre_thriller` | 1 if genres contain "Thriller", else 0 |
| `genre_science_fiction` | 1 if genres contain "Sci-Fi", else 0 |
| `genre_animation` | 1 if genres contain "Animation", else 0 |
| `genre_romance` | 1 if genres contain "Romance", else 0 |
| `genre_adventure` | 1 if genres contain "Adventure", else 0 |

> **Concept: One-Hot Encoding** — We can't give a model the word "Action". Instead, we create a separate column for each genre and put 1 or 0. A movie can have multiple 1s (e.g., Action-Adventure).

**Code:**
```python
popular_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller',
                  'Science Fiction', 'Animation', 'Romance', 'Adventure']

for genre in popular_genres:
    features[f'genre_{genre.lower()}'] = movies_df['genres'].apply(
        lambda x: 1 if isinstance(x, list) and genre in x else 0
    )
```

#### Group 4: Industry Features — One-Hot Encoding (~4 features)
| Feature | Value |
|---------|-------|
| `industry_hollywood` | 1 if Hollywood, else 0 |
| `industry_bollywood` | 1 if Bollywood, else 0 |
| `industry_tollywood` | 1 if Tollywood, else 0 |
| `industry_kollywood` | 1 if Kollywood, else 0 |

**Code:**
```python
industry_dummies = pd.get_dummies(features['industry'], prefix='industry')
features = pd.concat([features, industry_dummies], axis=1)
```

> `pd.get_dummies()` is Pandas' built-in one-hot encoder. It automatically creates columns like `industry_hollywood`, `industry_bollywood`, etc.

#### Group 5: Talent Features (3 features)
| Feature | How Calculated | Why |
|---------|---------------|-----|
| `director_popularity` | TMDB popularity score of director | Well-known directors draw audiences |
| `cast_popularity` | Average TMDB popularity of top cast | Star power sells tickets |
| `is_sequel` | 1 if sequel/franchise, else 0 | Sequels have built-in audience |

**Code:**
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
| `trailer_views` | Raw trailer view count | Direct measure of anticipation |
| `trailer_likes` | Number of likes | Positive sentiment |
| `trailer_comments` | Number of comments | Engagement level |
| `trailer_views_log` | `log(views + 1)` | Compressed scale (same log trick as revenue) |
| `trailer_engagement_ratio` | `(likes + comments) / views × 100` | Quality of engagement, not just volume |

> **Why engagement ratio?** A trailer with 10M views and 1M likes (10% engagement) signals more excitement than one with 100M views and 1M likes (1% engagement).

#### Group 7: External Ratings (3 features)
| Feature | Source | Why |
|---------|--------|-----|
| `imdb_rating` | OMDB API | Professional + audience consensus |
| `metascore` | OMDB API | Critical review score |
| `opening_gross` | Box Office Mojo | First-weekend sales predict total |

### Handling Missing Values

Not every movie has all 34 features. Our approach:

```python
features['budget'] = movies_df['budget'].fillna(0)        # No budget info → 0
features['runtime'] = movies_df['runtime'].fillna(120)     # Unknown → assume 2hrs
features['vote_average'] = movies_df['voteAverage'].fillna(6.0)  # Unknown → average
features['trailer_views'] = 0  # If no YouTube data, assume 0 views
```

> **Why these defaults?** We use the *mean/median/zero* strategy. Using 0 is safe because tree-based models (RandomForest, XGBoost) can learn that "0 means missing" and handle it accordingly.

---

## 7. Model Selection: Which Algorithms and Why

### Algorithm 1: RandomForestRegressor

**What it is:** An ensemble of hundreds of decision trees, each trained on a random subset of data and features. Final prediction = average of all trees.

**Why we use it:**
- Resistant to overfitting
- Handles non-linear relationships
- Provides feature importance
- Works well "out of the box"

**How it works (simplified):**
```
Tree 1: IF budget > $100M AND is_sequel=1 → predict $500M
Tree 2: IF cast_popularity > 50 AND genre_action=1 → predict $400M
Tree 3: IF trailer_views > 10M AND is_summer=1 → predict $600M
...
Final prediction = average(Tree 1, Tree 2, Tree 3, ...) = $500M
```

**Our tuned hyperparameters (found by Optuna):**
```python
RandomForestRegressor(
    n_estimators=200,      # 200 trees in the forest
    max_depth=14,          # Each tree can be up to 14 levels deep
    min_samples_split=2,   # Minimum 2 samples to split a node
    min_samples_leaf=1,    # Minimum 1 sample in a leaf
    max_features=None,     # Consider ALL features for each split
    random_state=42        # Reproducibility seed
)
```

### Algorithm 2: XGBoostRegressor (eXtreme Gradient Boosting)

**What it is:** Builds trees one at a time, where each new tree tries to fix the errors of the previous trees. This is "boosting" — models learn from mistakes.

**Why we use it:**
- Often #1 in ML competitions
- Built-in regularization prevents overfitting
- Fast training
- Handles missing values natively

**How it works (simplified):**
```
Tree 1: predict $300M (error: actual was $500M, so error = -$200M)
Tree 2: try to predict the -$200M error → predicts -$150M
Tree 3: try to predict remaining -$50M error → predicts -$40M
Final: $300M + $150M + $40M = $490M (much closer to $500M!)
```

**Our tuned hyperparameters:**
```python
XGBRegressor(
    n_estimators=450,            # 450 sequential trees
    max_depth=7,                 # Shallower trees (prevent overfitting)
    learning_rate=0.0339,        # Small steps (0.03 = conservative, careful learning)
    subsample=0.912,             # Use 91% of data for each tree (randomness)
    colsample_bytree=0.749,      # Use 75% of features per tree
    reg_alpha=0.000111,          # L1 regularization (lasso)
    reg_lambda=1.359             # L2 regularization (ridge)
)
```

> **Why learning_rate=0.034?** A lower learning rate = slower learning = less overfitting. Combined with more trees (n_estimators=450), this gives better generalization.

### Algorithm 3: LightGBMRegressor (Light Gradient Boosting Machine)

**What it is:** Similar to XGBoost but faster. Uses "leaf-wise" tree growth instead of "level-wise," which means it grows the most impactful leaf first.

**Why we use it:**
- Fastest gradient boosting implementation
- Handles large datasets efficiently
- Often outperforms XGBoost on medium-sized data
- Achieved our **best R² score (0.9340)**

**Our tuned hyperparameters:**
```python
LGBMRegressor(
    n_estimators=300,            # 300 trees
    max_depth=7,                 # Max depth per tree
    learning_rate=0.0185,        # Very conservative learning
    num_leaves=30,               # Max leaves per tree (LightGBM-specific!)
    subsample=0.728,             # 73% of data per tree
    colsample_bytree=0.962,      # 96% of features per tree
    reg_alpha=1.17e-08,          # Almost no L1 regularization
    reg_lambda=1.76e-05          # Very minimal L2 regularization
)
```

> **`num_leaves`** is unique to LightGBM. Instead of controlling tree depth, you directly control how many leaf nodes the tree can have. `num_leaves=30` means each tree can make up to 30 different predictions.

### Classification Models

For the 5-class success prediction, we use the same three algorithms plus the original:

| Model | Type | Role |
|-------|------|------|
| GradientBoostingClassifier | Sklearn built-in | Original baseline |
| XGBClassifier | XGBoost | Tuned with Optuna |
| LGBMClassifier | LightGBM | Tuned with Optuna |

---

## 8. Hyperparameter Tuning with Optuna

### What Are Hyperparameters?

**Parameters** are learned during training (e.g., decision boundaries in trees).
**Hyperparameters** are set BEFORE training (e.g., how many trees, how deep).

Choosing good hyperparameters is crucial — bad choices lead to underfitting (too simple) or overfitting (memorizing training data).

### What is Optuna?

Optuna is an **automated hyperparameter optimization framework**. Instead of manually trying different values, Optuna intelligently searches for the best combination.

### How Optuna Works

```
Trial 1:  n_estimators=150, max_depth=5  → R² = 0.89
Trial 2:  n_estimators=300, max_depth=10 → R² = 0.91
Trial 3:  n_estimators=250, max_depth=8  → R² = 0.92  ← looks promising!
Trial 4:  n_estimators=280, max_depth=9  → R² = 0.915 ← try nearby values
...
Trial 50: n_estimators=200, max_depth=14 → R² = 0.934 ← BEST!
```

Optuna uses **Tree-structured Parzen Estimator (TPE)** — it learns from previous trials which regions of the hyperparameter space are most promising and focuses exploration there.

### Code Example (from our trainer):

```python
def tune_rf_regressor(self, X_train, y_train):
    """Tune RandomForest hyperparameters"""

    def objective(trial):
        # Optuna suggests values to try
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }

        # Train with these params and evaluate
        model = RandomForestRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        return scores.mean()  # Optuna maximizes this value

    # Run 50 trials
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    return study.best_params  # The winning combination!
```

### Optuna Search Spaces We Used

| Parameter | Search Range | What It Controls |
|-----------|-------------|-----------------|
| `n_estimators` | 100-500 | Number of trees |
| `max_depth` | 3-25 | Tree depth (complexity) |
| `learning_rate` | 0.01-0.3 (log scale) | How fast model learns |
| `subsample` | 0.6-1.0 | Fraction of data per tree |
| `colsample_bytree` | 0.6-1.0 | Fraction of features per tree |
| `reg_alpha` | 1e-8 to 10 | L1 regularization strength |
| `reg_lambda` | 1e-8 to 10 | L2 regularization strength |
| `num_leaves` | 20-100 | Max leaves (LightGBM only) |

---

## 9. Ensemble Learning: Combining Models

### Why Ensemble?

Individual models have different strengths:

| Model | Strength | Weakness |
|-------|----------|----------|
| RandomForest | Stable, resistant to outliers | Can underfit complex patterns |
| XGBoost | Great at capturing patterns | Can overfit on small data |
| LightGBM | Fast, handles diverse features | Sensitive to noise |

By **combining** them, we get the best of all worlds. Errors from one model are compensated by the others.

### How Our Ensemble Works

#### Revenue: Weighted VotingRegressor

```
Input: Movie features
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
RandomForest  XGBoost    LightGBM
  predicts    predicts   predicts
  $450M       $480M      $520M
    │         │            │
    ▼         ▼            ▼
 × 0.332   × 0.332     × 0.336     (weights based on R² scores)
    │         │            │
    └────┬────┘            │
         └────────┬────────┘
                  ▼
    Weighted Average = $483.5M
```

**Code:**
```python
# Weights proportional to each model's R² score
total_r2 = rf_r2 + xgb_r2 + lgbm_r2
weights = [rf_r2 / total_r2, xgb_r2 / total_r2, lgbm_r2 / total_r2]
# Result: [0.332, 0.332, 0.336] — LightGBM gets slightly more weight (best R²)

ensemble_model = VotingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(**rf_best_params)),
        ('xgb', XGBRegressor(**xgb_best_params)),
        ('lgbm', LGBMRegressor(**lgbm_best_params))
    ],
    weights=weights
)
```

#### Classification: Soft VotingClassifier

Instead of averaging predictions (which are numbers for regression), classification uses **probability voting**:

```
Input: Movie features
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
GradBoost   XGBoost    LightGBM
 predicts   predicts    predicts
 probabilities:
 Block: 0.1   0.2         0.15
 Super: 0.2   0.15        0.25
 Hit:   0.4   0.35        0.30     ← Average probabilities
 Avg:   0.2   0.2         0.20
 Flop:  0.1   0.1         0.10
                  ▼
    Average → Hit (highest average probability)
```

**Code:**
```python
ensemble_cls = VotingClassifier(
    estimators=[
        ('gb', GradientBoostingClassifier(...)),
        ('xgb', XGBClassifier(...)),
        ('lgbm', LGBMClassifier(...))
    ],
    voting='soft'  # Use probability averaging, not majority vote
)
```

> **`voting='soft'` vs `voting='hard'`:**
> - `hard`: Each model votes for one class, majority wins (like elections)
> - `soft`: Each model gives probabilities, average probabilities win (more nuanced, usually better)

---

## 10. Cross-Validation: Honest Evaluation

### The Problem with Single Train/Test Split

If you just split data once (80/20), your results depend on **which** 20% you happened to pick. You could get lucky or unlucky.

### What is K-Fold Cross-Validation?

Split data into K parts ("folds"), train on K-1 parts, test on 1 part. Repeat K times:

```
5-Fold Cross-Validation:

Fold 1: [TEST] [Train] [Train] [Train] [Train]  → Score: 0.92
Fold 2: [Train] [TEST] [Train] [Train] [Train]  → Score: 0.91
Fold 3: [Train] [Train] [TEST] [Train] [Train]  → Score: 0.93
Fold 4: [Train] [Train] [Train] [TEST] [Train]  → Score: 0.90
Fold 5: [Train] [Train] [Train] [Train] [TEST]  → Score: 0.92

Final CV Score = Mean ± Std = 0.916 ± 0.031
```

Every data point gets to be in the test set exactly once!

### Stratified K-Fold (for Classification)

Regular K-Fold randomly splits data. But if your classes are **imbalanced** (e.g., many Hits, few Blockbusters), a random split might put all Blockbusters in one fold.

**Stratified K-Fold** ensures each fold has the **same proportion** of each class:

```
Dataset: 40% Hit, 30% Flop, 15% Average, 10% Super Hit, 5% Blockbuster

Each fold will have exactly:
  40% Hit, 30% Flop, 15% Average, 10% Super Hit, 5% Blockbuster
```

**Code:**
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_cls, X_train, y_train, cv=skf, scoring='accuracy')
# Result: [0.54, 0.56, 0.58, 0.52, 0.54] → Mean: 0.547 ± 0.064
```

### Our CV Results

| Model | CV Mean | ± Std | Interpretation |
|-------|---------|-------|---------------|
| Revenue Ensemble | **0.916** | ± 0.031 | Very stable, high performance |
| Classification Ensemble | **0.547** | ± 0.064 | Moderate — 5-class problem is hard |

---

## 11. Model Performance & Results

### Revenue Prediction Results

| Model | R² Score | MAE (log) | Meaning |
|-------|----------|-----------|---------|
| RandomForest (old) | 0.9083 | 0.42 | Baseline before improvements |
| **RandomForest** (Optuna-tuned) | **0.9210** | 0.2984 | +1.4% improvement |
| **XGBoost** (Optuna-tuned) | **0.9229** | 0.2995 | +1.6% improvement |
| **LightGBM** (Optuna-tuned) | **0.9340** 🏆 | 0.2828 | **+2.8% improvement (BEST)** |
| **Ensemble** (weighted voting) | **0.9294** | 0.2852 | All 3 combined |

> **R² = 0.934 means:** The model explains **93.4%** of the variance in movie revenue. Only 6.6% of revenue variation is unexplained (due to luck, marketing, etc.).

> **MAE = 0.28 (in log space)** means: On average, our prediction is off by `e^0.28 - 1 ≈ 32%` of the actual revenue. For a $100M movie, we'd predict somewhere between $68M-$132M.

### Classification Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| GradientBoosting | 55.9% | Original sklearn model |
| XGBoost (tuned) | **57.4%** | Best individual model |
| LightGBM (tuned) | **57.4%** | Tied for best |
| Ensemble (soft voting) | 56.5% | Combined |
| Stratified 5-Fold CV | **54.7% ± 6.4%** | Honest evaluation |

> **Why is 57% "good" for 5 classes?**
> - Random guessing = 20% (1 out of 5)
> - Our model = 57% = **2.85× better than random**
> - Adjacent classes (Hit vs Super Hit) are very similar, making exact classification hard
> - The model rarely confuses Blockbusters with Flops — it's the middle categories that overlap

### Ensemble Weights

| Model | Weight (Revenue) | Meaning |
|-------|-----------------|---------|
| RandomForest | 0.332 (33.2%) | Equal contribution |
| XGBoost | 0.332 (33.2%) | Equal contribution |
| LightGBM | **0.336 (33.6%)** | Slightly more weight (best R²) |

---

## 12. SHAP Explainability

### What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions. It tells you exactly which features pushed the prediction up or down, and by how much.

Based on **Shapley values** from game theory — a fair way to distribute credit among players in a cooperative game.

### How We Use It

For each prediction, we can explain:

```
Prediction: This movie will be a "Hit" (predicted revenue: $350M)

Why?
  budget = $150M         → pushed prediction UP by $120M    ████████████
  cast_popularity = 65   → pushed prediction UP by $50M     █████
  genre_action = 1       → pushed prediction UP by $30M     ███
  is_summer_release = 1  → pushed prediction UP by $20M     ██
  trailer_views = 5M     → pushed prediction DOWN by -$10M  █ (low views)
  is_sequel = 0          → pushed prediction DOWN by -$15M  ██ (not a sequel)
```

### Feature Importance (Averaged Across All Predictions)

From our ensemble model, the top 10 most important features:

| Rank | Feature | Importance | What This Means |
|------|---------|------------|-----------------|
| 1 | `budget` | High | Budget is the #1 indicator of revenue potential |
| 2 | `trailer_views_log` | High | Pre-release buzz strongly predicts success |
| 3 | `popularity` | Medium-High | TMDB popularity captures public awareness |
| 4 | `vote_count` | Medium | More votes = more viewers |
| 5 | `cast_popularity` | Medium | Star power still matters |
| 6 | `director_popularity` | Medium | Known directors draw audiences |
| 7 | `vote_average` | Medium | Quality matters, but budget matters more |
| 8 | `runtime` | Low-Medium | Optimal length exists (~120-150 min) |
| 9 | `is_summer_release` | Low | Seasonal advantage is modest |
| 10 | `genre_action` | Low | Action has slight edge but isn't decisive |

---

## 13. Prediction Flow: From Input to Output

### What Happens When a User Clicks "Predict"

```
1. USER fills form:
   Title: "Avengers 5"
   Budget: $250,000,000
   Genre: Action, Adventure
   Industry: Hollywood
   Director Popularity: 85
   Cast Popularity: 90
   Release Month: July
   Is Sequel: Yes

2. REACT sends POST request to Node.js:
   POST http://localhost:5000/api/ml/predict
   Body: { budget: 250000000, genres: ["Action", "Adventure"], ... }

3. NODE.JS proxies to ML Service:
   POST http://localhost:5001/api/predict

4. PREDICTOR.PY processes:

   a) _prepare_features(data):
      Creates dict with 34 features, fills in values:
      {
        budget: 250000000,
        runtime: 120,
        genre_action: 1,
        genre_adventure: 1,
        genre_comedy: 0,
        ...
        industry_hollywood: 1,
        director_popularity: 85,
        cast_popularity: 90,
        is_sequel: 1,
        is_summer_release: 1,
        ...
      }

   b) scaler.transform(features):
      Converts each feature to z-score:
      budget 250M → standardized to ~2.1 (2.1 std devs above mean)
      cast_popularity 90 → standardized to ~3.5

   c) revenue_model.predict(scaled_features):
      Ensemble makes prediction:
        RF predicts: 21.2 (in log-revenue)
        XGB predicts: 21.5
        LGBM predicts: 21.3
        Weighted avg: 21.33
      
      Convert back: expm1(21.33) = $1,837,000,000

   d) classifier_model.predict(scaled_features):
      → "Blockbuster" (with 78% confidence)

   e) Calculate ROI:
      ($1.837B - $250M) / $250M × 100 = 635%

5. RESPONSE sent back:
   {
     predictions: {
       successCategory: "Blockbuster",
       predictedRevenue: 1837000000,
       predictedROI: 635,
       confidence: 78,
       featureImportance: [
         { feature: "Budget", impact: 0.35 },
         { feature: "Cast Popularity", impact: 0.15 },
         ...
       ]
     }
   }

6. REACT renders the prediction card with animated badge
```

---

## 14. Model Serialization: Saving & Loading

### What is Pickle?

**Pickle** is Python's built-in serialization format. It converts Python objects (like trained models) into binary files that can be saved to disk and loaded later.

### What We Save

| File | What's Inside | Size | Why |
|------|--------------|------|-----|
| `revenue_model.pkl` | VotingRegressor (3 models inside) | ~24 MB | Contains RF, XGB, LGBM with learned parameters |
| `classifier_model.pkl` | VotingClassifier (3 models inside) | ~7.7 MB | Contains GB, XGB, LGBM classifiers |
| `scaler.pkl` | StandardScaler | ~2 KB | Mean and std of each feature (from training data) |
| `label_encoder.pkl` | LabelEncoder | ~300 B | Mapping: 0=Average, 1=Blockbuster, 2=Flop, 3=Hit, 4=Super Hit |
| `feature_columns.pkl` | List of 34 strings | ~600 B | Ensuring prediction features are in correct order |
| `best_params.json` | Dict of dicts | ~1.2 KB | Optuna's best hyperparameters |
| `training_results.json` | Metrics dict | ~640 B | R² scores, accuracies, CV results |

### Save Code:
```python
# Save model
with open('revenue_model.pkl', 'wb') as f:    # 'wb' = write binary
    pickle.dump(self.revenue_model, f)

# Load model
with open('revenue_model.pkl', 'rb') as f:    # 'rb' = read binary
    self.revenue_model = pickle.load(f)
```

> **Why not retrain every time?** Training takes 14 minutes and requires MongoDB connection. Loading a pickle file takes <1 second. The server starts instantly with pre-trained models.

---

## 15. Key Learnings & Insights

### What We Learned About Movie Success

1. **Money predicts money** — Budget is the #1 feature. Studios that invest more typically earn more (but not always proportionally).

2. **Trailers matter A LOT** — YouTube trailer views are the 2nd most important feature. Pre-release buzz is a strong signal.

3. **Stars still sell tickets** — Director and cast popularity are significant features, confirming that star power hasn't died in the streaming era.

4. **Timing is modest** — Summer and holiday releases have a small advantage, but aren't decisive. A great movie can succeed any time.

5. **Sequels have an edge** — Built-in audiences make sequels more predictable (lower risk, but not necessarily higher ROI).

### What We Learned About ML

1. **Log transformation is essential** for skewed targets — Without it, R² was 0.72. With it: 0.93.

2. **Hyperparameter tuning matters** — Optuna improved R² from 0.9083 to 0.9340 (+2.8%).

3. **Ensemble > Individual** — No single model is always best. Combining three models gives stable, high performance.

4. **Honest evaluation is crucial** — The old "85% accuracy" was misleading. Stratified K-Fold gives the real picture (54.7%).

5. **Feature engineering > more data** — Our 34 carefully engineered features from 1,652 movies outperform models trained on raw data from 10,000 movies.

6. **StandardScaler is necessary** — Without scaling, models that use distance/gradient (XGBoost, LightGBM) would be dominated by large-scale features like budget ($200M) over vote_average (7.5).

### Technical Decisions & Rationale

| Decision | Why |
|----------|-----|
| RandomForest + XGBoost + LightGBM | Three different tree algorithms that complement each other |
| 50 Optuna trials | Balances search quality vs training time (~14 min total) |
| Weighted voting (not simple average) | Better models get proportionally more influence |
| Soft voting for classification | Probability-based voting outperforms hard majority voting |
| 80/20 train-test split | Standard split; with 1,652 samples, 20% test ≈ 330 movies |
| 5-fold CV | Industry standard; gives 5 independent estimates |
| Log1p instead of log | Handles zero-revenue movies safely (`log(0)` = error, `log(0+1)` = 0) |

---

*Report Generated: February 10, 2026*
*For: GreenLit GO — Movie Success Prediction Project*
*ML Stack: scikit-learn + XGBoost + LightGBM + Optuna + SHAP*
