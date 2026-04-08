"""
GreenLit Go - Enhanced ML Training Pipeline
With: Optuna Hyperparameter Tuning, Ensemble Models, Stratified K-Fold CV
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingClassifier,
    StackingRegressor, StackingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    classification_report, f1_score
)

# Advanced models
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ─── Success Categories (ROI-based) ─────────────────────────
SUCCESS_CATEGORIES = {
    'Blockbuster': 400,
    'Super Hit': 200,
    'Hit': 100,
    'Average': 0,
    'Flop': -100
}


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

class MovieFeatureEngineer:
    """Feature engineering for movie data"""

    def __init__(self):
        self.genre_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.all_genres = set()

    def extract_features(self, movies_df):
        """Extract and engineer features from movie data"""
        print("   Extracting features...")

        features = pd.DataFrame()

        # Basic features
        features['budget'] = movies_df['budget'].fillna(0)
        features['runtime'] = movies_df['runtime'].fillna(120)
        features['vote_average'] = movies_df['voteAverage'].fillna(6.0)
        features['vote_count'] = movies_df['voteCount'].fillna(100)
        features['popularity'] = movies_df['popularity'].fillna(10)

        # Year and release timing
        features['year'] = movies_df['year'].fillna(2020)

        # Release month (seasonality)
        def get_release_month(row):
            try:
                if pd.isna(row['releaseDate']):
                    return 6
                if isinstance(row['releaseDate'], str):
                    return int(row['releaseDate'].split('-')[1])
                return row['releaseDate'].month
            except:
                return 6

        features['release_month'] = movies_df.apply(get_release_month, axis=1)

        # Summer release (May-Aug)
        features['is_summer_release'] = features['release_month'].apply(
            lambda x: 1 if x in [5, 6, 7, 8] else 0
        )

        # Holiday release (Nov-Dec)
        features['is_holiday_release'] = features['release_month'].apply(
            lambda x: 1 if x in [11, 12] else 0
        )

        # Genre features (one-hot encoding)
        def get_primary_genre(genres):
            if isinstance(genres, list) and len(genres) > 0:
                return genres[0]
            return 'Drama'

        features['primary_genre'] = movies_df['genres'].apply(get_primary_genre)

        # Popular genres as binary features
        popular_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Thriller',
                          'Science Fiction', 'Animation', 'Romance', 'Adventure']

        for genre in popular_genres:
            features[f'genre_{genre.lower().replace(" ", "_")}'] = movies_df['genres'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )

        # Industry encoding
        features['industry'] = movies_df['industry'].fillna('hollywood')
        industry_dummies = pd.get_dummies(features['industry'], prefix='industry')
        features = pd.concat([features, industry_dummies], axis=1)

        # Director popularity
        def get_director_popularity(director):
            if isinstance(director, dict):
                return director.get('popularity', 0)
            return 0

        features['director_popularity'] = movies_df['director'].apply(get_director_popularity)

        # Cast popularity (average of top cast)
        def get_cast_popularity(cast):
            if isinstance(cast, list) and len(cast) > 0:
                pops = [c.get('popularity', 0) for c in cast if isinstance(c, dict)]
                return np.mean(pops) if pops else 0
            return 0

        features['cast_popularity'] = movies_df['cast'].apply(get_cast_popularity)

        # Sequel indicator
        features['is_sequel'] = movies_df['isSequel'].fillna(False).astype(int)

        # OMDB data
        def get_omdb_rating(omdb):
            if isinstance(omdb, dict):
                return omdb.get('imdbRating', 0) or 0
            return 0

        def get_metascore(omdb):
            if isinstance(omdb, dict):
                score = omdb.get('metascore', 0)
                return score if score else 0
            return 0

        features['imdb_rating'] = movies_df['omdb'].apply(get_omdb_rating) if 'omdb' in movies_df else 0
        features['metascore'] = movies_df['omdb'].apply(get_metascore) if 'omdb' in movies_df else 0

        # BOM data
        def get_opening_gross(bom):
            if isinstance(bom, dict):
                return bom.get('openingGross', 0) or 0
            return 0

        features['opening_gross'] = movies_df['bom'].apply(get_opening_gross) if 'bom' in movies_df else 0

        # YouTube trailer data
        def get_youtube_views(youtube):
            if isinstance(youtube, dict):
                return youtube.get('viewCount', 0) or 0
            return 0

        def get_youtube_likes(youtube):
            if isinstance(youtube, dict):
                return youtube.get('likeCount', 0) or 0
            return 0

        def get_youtube_comments(youtube):
            if isinstance(youtube, dict):
                return youtube.get('commentCount', 0) or 0
            return 0

        def get_youtube_engagement_ratio(youtube):
            if isinstance(youtube, dict):
                views = youtube.get('viewCount', 0) or 0
                likes = youtube.get('likeCount', 0) or 0
                comments = youtube.get('commentCount', 0) or 0
                if views > 0:
                    return (likes + comments) / views * 100
            return 0

        if 'youtube' in movies_df.columns:
            features['trailer_views'] = movies_df['youtube'].apply(get_youtube_views)
            features['trailer_likes'] = movies_df['youtube'].apply(get_youtube_likes)
            features['trailer_comments'] = movies_df['youtube'].apply(get_youtube_comments)
            features['trailer_engagement_ratio'] = movies_df['youtube'].apply(get_youtube_engagement_ratio)
            features['trailer_views_log'] = np.log1p(features['trailer_views'])
            print(f"   ✓ Added YouTube features for {(features['trailer_views'] > 0).sum()} movies")
        else:
            features['trailer_views'] = 0
            features['trailer_likes'] = 0
            features['trailer_comments'] = 0
            features['trailer_engagement_ratio'] = 0
            features['trailer_views_log'] = 0

        # ─── DERIVED / INTERACTION FEATURES ─────────────────────
        # Inflation-adjusted budget (to 2024 dollars, 3% annual)
        features['budget_adjusted'] = features['budget'] * (1.03 ** (2024 - features['year']))

        # Cyclic month encoding (captures seasonal circularity)
        features['month_sin'] = np.sin(2 * np.pi * features['release_month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['release_month'] / 12)

        # Cast max popularity (strongest single star)
        def get_cast_max_popularity(cast):
            if isinstance(cast, list) and len(cast) > 0:
                pops = [c.get('popularity', 0) for c in cast if isinstance(c, dict)]
                return max(pops) if pops else 0
            return 0
        features['cast_max_popularity'] = movies_df['cast'].apply(get_cast_max_popularity)

        # Genre count
        features['genre_count'] = movies_df['genres'].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )

        # Log budget (better distribution)
        features['log_budget'] = np.log1p(features['budget'])

        # Release quarter
        features['release_quarter'] = (features['release_month'] - 1) // 3 + 1

        # Budget per minute
        features['budget_per_minute'] = features['budget'] / features['runtime'].clip(lower=60)

        # Budget × cast popularity interaction
        features['budget_x_cast_pop'] = features['budget'] * features['cast_popularity']

        # Budget per talent (budget efficiency relative to star power)
        total_talent = (features['director_popularity'] + features['cast_popularity']).clip(lower=0.1)
        features['budget_per_talent'] = features['budget'] / total_talent
        features['log_budget_per_talent'] = np.log1p(features['budget_per_talent'])

        # Talent synergy (director × cast interaction)
        features['talent_synergy'] = features['director_popularity'] * features['cast_popularity']

        # Drop non-numeric columns
        features = features.drop(columns=['primary_genre', 'industry'], errors='ignore')

        return features

    def calculate_target(self, movies_df):
        """Calculate target variables"""
        print("   Calculating targets...")

        revenue = movies_df['revenue'].fillna(0)
        log_revenue = np.log1p(revenue)

        def get_success_category(row):
            budget = row.get('budget', 0) or 0
            rev = row.get('revenue', 0) or 0

            if budget <= 0:
                vote = row.get('voteAverage', 5)
                if vote >= 8:
                    return 'Blockbuster'
                elif vote >= 7:
                    return 'Hit'
                elif vote >= 6:
                    return 'Average'
                else:
                    return 'Flop'

            roi = ((rev - budget) / budget) * 100

            if roi >= 400:
                return 'Blockbuster'
            elif roi >= 200:
                return 'Super Hit'
            elif roi >= 100:
                return 'Hit'
            elif roi >= 0:
                return 'Average'
            else:
                return 'Flop'

        success_category = movies_df.apply(get_success_category, axis=1)

        return log_revenue, success_category


# ═══════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════

class OptunaHyperparamTuner:
    """Automated hyperparameter tuning with Optuna"""

    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.best_params = {}

    def tune_rf_regressor(self, X_train, y_train):
        """Tune RandomForest hyperparameters"""
        print(f"\n🔍 Tuning RandomForest ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['rf_regressor'] = study.best_params
        print(f"   ✓ Best R² (CV): {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params

    def tune_xgb_regressor(self, X_train, y_train):
        """Tune XGBoost regressor hyperparameters"""
        print(f"\n🔍 Tuning XGBoost Regressor ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1
            }
            model = XGBRegressor(**params, verbosity=0)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['xgb_regressor'] = study.best_params
        print(f"   ✓ Best R² (CV): {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params

    def tune_lgbm_regressor(self, X_train, y_train):
        """Tune LightGBM regressor hyperparameters"""
        print(f"\n🔍 Tuning LightGBM Regressor ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            model = LGBMRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['lgbm_regressor'] = study.best_params
        print(f"   ✓ Best R² (CV): {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params

    def tune_xgb_classifier(self, X_train, y_train):
        """Tune XGBoost classifier hyperparameters"""
        print(f"\n🔍 Tuning XGBoost Classifier ({self.n_trials} trials)...")
        n_classes = len(np.unique(y_train))

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }
            model = XGBClassifier(**params, verbosity=0)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['xgb_classifier'] = study.best_params
        print(f"   ✓ Best Accuracy (CV): {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params

    def tune_lgbm_classifier(self, X_train, y_train):
        """Tune LightGBM classifier hyperparameters"""
        print(f"\n🔍 Tuning LightGBM Classifier ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            model = LGBMClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params['lgbm_classifier'] = study.best_params
        print(f"   ✓ Best Accuracy (CV): {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params

    def tune_knn_regressor(self, X, y):
        """Optuna tuning for KNN Regressor"""
        print("\n   Tuning KNN Regressor...")

        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
            model = KNeighborsRegressor(**params, n_jobs=-1)
            return cross_val_score(model, X, y, cv=3, scoring='r2').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params['knn_regressor'] = study.best_params
        print(f"   ✓ Best R² (CV): {study.best_value:.4f}")
        return study.best_params

    def save_best_params(self, filepath):
        """Save best parameters to JSON"""
        # Convert numpy types to Python types for JSON serialization
        clean_params = {}
        for key, params in self.best_params.items():
            clean_params[key] = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in params.items()
            }
        with open(filepath, 'w') as f:
            json.dump(clean_params, f, indent=2)
        print(f"   ✓ Best params saved to {filepath}")


# ═══════════════════════════════════════════════════════════════
#  ENHANCED MODEL TRAINER
# ═══════════════════════════════════════════════════════════════

class MovieModelTrainer:
    """Train and evaluate movie prediction models with ensembling"""

    def __init__(self, n_tuning_trials=50):
        self.feature_engineer = MovieFeatureEngineer()
        self.tuner = OptunaHyperparamTuner(n_trials=n_tuning_trials)
        self.revenue_model = None
        self.classifier_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_results = {}

    # ─── Data Loading ─────────────────────────────────────

    def load_data_from_mongodb(self):
        """Load movie data from MongoDB"""
        print("\n📊 Loading data from MongoDB...")

        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
        client = MongoClient(mongo_uri)
        db = client.get_database()

        movies = list(db['movies'].find())
        client.close()

        df = pd.DataFrame(movies)
        print(f"   Loaded {len(df)} movies")

        return df

    def prepare_data(self, df):
        """Prepare data for training"""
        print("\n🔧 Preparing data...")

        df_valid = df[(df['budget'] > 0) | (df['revenue'] > 0)].copy()
        print(f"   Movies with financial data: {len(df_valid)}")

        X = self.feature_engineer.extract_features(df_valid)
        y_revenue, y_category = self.feature_engineer.calculate_target(df_valid)

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"   Features shape: {X.shape}")
        print(f"   Feature columns: {list(X.columns)[:10]}...")

        # Show category distribution
        print(f"\n   📊 Category distribution:")
        for cat, count in y_category.value_counts().items():
            pct = count / len(y_category) * 100
            print(f"      {cat:15} {count:4} ({pct:.1f}%)")

        return X, y_revenue, y_category, df_valid

    # ─── Revenue Model Training ──────────────────────────

    def train_revenue_model(self, X, y):
        """Train ensemble revenue prediction model with Optuna tuning"""
        print("\n" + "═" * 60)
        print("💰 REVENUE PREDICTION MODEL")
        print("═" * 60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── Step 1: Tune hyperparameters ──
        print("\n📐 STEP 1: Hyperparameter Tuning")
        print("─" * 40)

        rf_params = self.tuner.tune_rf_regressor(X_train_scaled, y_train)
        xgb_params = self.tuner.tune_xgb_regressor(X_train_scaled, y_train)
        lgbm_params = self.tuner.tune_lgbm_regressor(X_train_scaled, y_train)
        knn_params = self.tuner.tune_knn_regressor(X_train_scaled, y_train)

        # ── Step 2: Train individual models with best params ──
        print("\n🏋️ STEP 2: Training Individual Models")
        print("─" * 40)

        # RandomForest
        rf_model = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        print(f"   RandomForest   → R²: {rf_r2:.4f}, MAE: {rf_mae:.4f}")

        # XGBoost
        xgb_model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        print(f"   XGBoost        → R²: {xgb_r2:.4f}, MAE: {xgb_mae:.4f}")

        # LightGBM
        lgbm_model = LGBMRegressor(**lgbm_params, random_state=42, n_jobs=-1, verbose=-1)
        lgbm_model.fit(X_train_scaled, y_train)
        lgbm_pred = lgbm_model.predict(X_test_scaled)
        lgbm_r2 = r2_score(y_test, lgbm_pred)
        lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
        print(f"   LightGBM       → R²: {lgbm_r2:.4f}, MAE: {lgbm_mae:.4f}")

        # KNN
        knn_model = KNeighborsRegressor(**knn_params, n_jobs=-1)
        knn_model.fit(X_train_scaled, y_train)
        knn_pred = knn_model.predict(X_test_scaled)
        knn_r2 = r2_score(y_test, knn_pred)
        knn_mae = mean_absolute_error(y_test, knn_pred)
        print(f"   KNN            → R²: {knn_r2:.4f}, MAE: {knn_mae:.4f}")

        # ── Step 3: Create Ensemble (StackingRegressor) ──
        print("\n🤝 STEP 3: Ensemble Model (Stacking with Ridge Meta-learner)")
        print("─" * 40)

        # Stacking Ensemble
        ensemble_model = StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0)),
                ('lgbm', LGBMRegressor(**lgbm_params, random_state=42, n_jobs=-1, verbose=-1)),
                ('knn', KNeighborsRegressor(**knn_params, n_jobs=-1))
            ],
            final_estimator=Ridge()
        )

        # ── Step 4: Cross-Validation on best approach ──
        print("\n📊 STEP 4: 5-Fold Cross-Validation")
        print("─" * 40)

        cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   CV Mean:   {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

        # ── Final: Fit ensemble on full training data ──
        ensemble_model.fit(X_train_scaled, y_train)
        final_pred = ensemble_model.predict(X_test_scaled)
        final_r2 = r2_score(y_test, final_pred)
        final_mae = mean_absolute_error(y_test, final_pred)

        self.revenue_model = ensemble_model

        # Store results for comparison
        self.training_results['revenue'] = {
            'rf_r2': rf_r2, 'xgb_r2': xgb_r2, 'lgbm_r2': lgbm_r2, 'knn_r2': knn_r2,
            'ensemble_r2': final_r2, 'ensemble_mae': final_mae,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }

        print(f"\n   ✅ Final Ensemble R²: {final_r2:.4f}")
        return final_mae, final_r2

    # ─── Classification Model Training ───────────────────

    def train_classification_model(self, X, y):
        """Train ensemble classification model with Optuna + Stratified K-Fold"""
        print("\n" + "═" * 60)
        print("🎯 SUCCESS CLASSIFICATION MODEL")
        print("═" * 60)

        # Encode categories
        y_encoded = self.label_encoder.fit_transform(y)

        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── Step 1: Tune hyperparameters ──
        print("\n📐 STEP 1: Hyperparameter Tuning")
        print("─" * 40)

        xgb_cls_params = self.tuner.tune_xgb_classifier(X_train_scaled, y_train)
        lgbm_cls_params = self.tuner.tune_lgbm_classifier(X_train_scaled, y_train)

        # ── Step 2: Train individual models ──
        print("\n🏋️ STEP 2: Training Individual Models")
        print("─" * 40)

        # GradientBoosting (original)
        gb_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)
        print(f"   GradientBoost  → Accuracy: {gb_acc:.4f}")

        # XGBoost
        xgb_cls = XGBClassifier(
            **xgb_cls_params, random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss', verbosity=0
        )
        xgb_cls.fit(X_train_scaled, y_train)
        xgb_pred = xgb_cls.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        print(f"   XGBoost        → Accuracy: {xgb_acc:.4f}")

        # LightGBM
        lgbm_cls = LGBMClassifier(
            **lgbm_cls_params, random_state=42, n_jobs=-1, verbose=-1
        )
        lgbm_cls.fit(X_train_scaled, y_train)
        lgbm_pred = lgbm_cls.predict(X_test_scaled)
        lgbm_acc = accuracy_score(y_test, lgbm_pred)
        print(f"   LightGBM       → Accuracy: {lgbm_acc:.4f}")

        # ── Step 3: Ensemble Stacking Classifier ──
        print("\n🤝 STEP 3: Ensemble Stacking Classifier")
        print("─" * 40)

        ensemble_cls = StackingClassifier(
            estimators=[
                ('gb', GradientBoostingClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
                )),
                ('xgb', XGBClassifier(
                    **xgb_cls_params, random_state=42, n_jobs=-1,
                    use_label_encoder=False, eval_metric='mlogloss', verbosity=0
                )),
                ('lgbm', LGBMClassifier(
                    **lgbm_cls_params, random_state=42, n_jobs=-1, verbose=-1
                ))
            ],
            final_estimator=LogisticRegression(max_iter=1000)
        )

        # ── Step 4: Stratified K-Fold Cross-Validation ──
        print("\n📊 STEP 4: Stratified 5-Fold Cross-Validation")
        print("─" * 40)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(ensemble_cls, X_train_scaled, y_train, cv=skf, scoring='accuracy')

        print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   CV Mean:   {cv_scores.mean():.4f} (± {cv_scores.std() * 2:.4f})")

        # ── Final: Fit ensemble on full training data ──
        ensemble_cls.fit(X_train_scaled, y_train)
        final_pred = ensemble_cls.predict(X_test_scaled)
        final_acc = accuracy_score(y_test, final_pred)

        self.classifier_model = ensemble_cls

        # Classification report
        print(f"\n   ✅ Final Ensemble Accuracy: {final_acc:.4f}")
        print("\n   Classification Report:")
        report = classification_report(
            y_test, final_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        print(report)

        # Store results
        self.training_results['classification'] = {
            'gb_acc': gb_acc, 'xgb_acc': xgb_acc, 'lgbm_acc': lgbm_acc,
            'ensemble_acc': final_acc,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }

        return final_acc

    # ─── Feature Importance ──────────────────────────────

    def get_feature_importance(self, X):
        """Get feature importance from ensemble revenue model"""
        print("\n📈 Feature Importance (Top 15):")

        # Average importance across ensemble members
        importances = np.zeros(len(X.columns))
        n_models = 0

        for name, model in self.revenue_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
                n_models += 1

        if n_models > 0:
            importances /= n_models

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        for _, row in importance.head(15).iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature'][:25]:25} {bar} {row['importance']:.4f}")

        return importance

    # ─── Save / Load Models ──────────────────────────────

    def save_models(self, output_dir):
        """Save trained models and hyperparameters"""
        print(f"\n💾 Saving models to {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        # Save models
        with open(os.path.join(output_dir, 'revenue_model.pkl'), 'wb') as f:
            pickle.dump(self.revenue_model, f)

        with open(os.path.join(output_dir, 'classifier_model.pkl'), 'wb') as f:
            pickle.dump(self.classifier_model, f)

        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Save feature columns
        with open(os.path.join(output_dir, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(list(self.feature_columns), f)

        # Save tuned hyperparameters
        self.tuner.save_best_params(os.path.join(output_dir, 'best_params.json'))

        # Save training results
        with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)

        print("   ✓ Models saved successfully!")

    # ─── Main Training Pipeline ──────────────────────────

    def train(self):
        """Run complete enhanced training pipeline"""
        print("=" * 60)
        print("🎬 GreenLit Go - Enhanced ML Model Training")
        print("   With: Optuna + Ensemble + Stratified K-Fold")
        print("=" * 60)

        start_time = datetime.now()

        # Load data
        df = self.load_data_from_mongodb()

        # Prepare data
        X, y_revenue, y_category, df_valid = self.prepare_data(df)
        self.feature_columns = X.columns

        # Train models
        mae, r2 = self.train_revenue_model(X, y_revenue)
        accuracy = self.train_classification_model(X, y_category)

        # Feature importance
        importance = self.get_feature_importance(X)

        # Save models
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.save_models(model_dir)

        # Training time
        elapsed = datetime.now() - start_time

        # ── Summary ──
        print("\n" + "=" * 60)
        print("✅ ENHANCED TRAINING COMPLETE!")
        print("=" * 60)

        rev_results = self.training_results.get('revenue', {})
        cls_results = self.training_results.get('classification', {})

        print(f"\n   ⏱️  Training time: {elapsed}")
        print(f"   📊 Movies trained on: {len(X)}")
        print(f"   🔢 Features used: {len(X.columns)}")

        print(f"\n   💰 Revenue Model (Ensemble):")
        print(f"      RF:       R² = {rev_results.get('rf_r2', 0):.4f}")
        print(f"      XGBoost:  R² = {rev_results.get('xgb_r2', 0):.4f}")
        print(f"      LightGBM: R² = {rev_results.get('lgbm_r2', 0):.4f}")
        print(f"      Ensemble: R² = {rev_results.get('ensemble_r2', 0):.4f}")
        print(f"      CV Score: {rev_results.get('cv_mean', 0):.4f} (± {rev_results.get('cv_std', 0) * 2:.4f})")

        print(f"\n   🎯 Classification Model (Ensemble):")
        print(f"      GB:       Acc = {cls_results.get('gb_acc', 0):.4f}")
        print(f"      XGBoost:  Acc = {cls_results.get('xgb_acc', 0):.4f}")
        print(f"      LightGBM: Acc = {cls_results.get('lgbm_acc', 0):.4f}")
        print(f"      Ensemble: Acc = {cls_results.get('ensemble_acc', 0):.4f}")
        print(f"      CV Score: {cls_results.get('cv_mean', 0):.4f} (± {cls_results.get('cv_std', 0) * 2:.4f})")

        print(f"\n   📁 Models saved to: {model_dir}")

        return {
            'movies_trained': len(X),
            'features': len(X.columns),
            'revenue_r2': r2,
            'classification_accuracy': accuracy,
            'training_time': str(elapsed)
        }


if __name__ == '__main__':
    trainer = MovieModelTrainer(n_tuning_trials=50)
    results = trainer.train()
