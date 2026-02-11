"""
Batch Prediction Script
Runs predictions for all movies without success category and updates the database.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go'))
db = client.get_database()
movies_collection = db.movies

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'app', 'models')

try:
    with open(os.path.join(MODEL_DIR, 'revenue_model.pkl'), 'rb') as f:
        revenue_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'classifier_model.pkl'), 'rb') as f:
        classifier_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    print(f"✓ Models loaded. Features: {len(feature_columns)}")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    sys.exit(1)


def classify_movie(budget, revenue, vote_average):
    """Classify movie success based on budget, revenue, and rating."""
    if budget and budget > 0 and revenue:
        roi = (revenue - budget) / budget
        if roi >= 2.0 and vote_average >= 7.5:
            return 'Blockbuster'
        elif roi >= 1.0:
            return 'Hit'
        elif roi >= 0:
            return 'Average'
        else:
            return 'Flop'
    elif vote_average:
        if vote_average >= 8.0:
            return 'Blockbuster'
        elif vote_average >= 7.0:
            return 'Hit'
        elif vote_average >= 5.5:
            return 'Average'
        else:
            return 'Flop'
    return 'Average'


def prepare_features(movie):
    """Prepare feature vector for a movie."""
    features = {}
    
    # Basic features
    features['budget'] = movie.get('budget', 0) or 0
    features['runtime'] = movie.get('runtime', 120) or 120
    features['vote_average'] = movie.get('voteAverage', 6.0) or 6.0
    features['vote_count'] = movie.get('voteCount', 100) or 100
    features['popularity'] = movie.get('popularity', 10) or 10
    
    # Temporal features
    release_date = movie.get('releaseDate')
    if release_date:
        try:
            from datetime import datetime
            if isinstance(release_date, str):
                dt = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
            else:
                dt = release_date
            features['release_month'] = dt.month
            features['release_quarter'] = (dt.month - 1) // 3 + 1
            features['release_year'] = dt.year
        except:
            features['release_month'] = 6
            features['release_quarter'] = 2
            features['release_year'] = 2023
    else:
        features['release_month'] = 6
        features['release_quarter'] = 2
        features['release_year'] = 2023
    
    # Genre encoding (one-hot)
    all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
                  'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
    movie_genres = movie.get('genres', [])
    for genre in all_genres:
        features[f'genre_{genre.lower().replace(" ", "_")}'] = 1 if genre in movie_genres else 0
    
    # Industry encoding
    industry = movie.get('industry', 'hollywood')
    for ind in ['hollywood', 'bollywood', 'tollywood', 'kollywood', 'mollywood']:
        features[f'industry_{ind}'] = 1 if industry == ind else 0
    
    # Derived features
    features['budget_log'] = np.log1p(features['budget'])
    features['vote_count_log'] = np.log1p(features['vote_count'])
    features['popularity_log'] = np.log1p(features['popularity'])
    
    # Fill missing with defaults
    for col in feature_columns:
        if col not in features:
            features[col] = 0
    
    return features


def run_batch_predictions():
    """Run predictions for all movies without predictions."""
    # Find movies without success category
    query = {
        '$or': [
            {'predictions.successCategory': {'$exists': False}},
            {'predictions.successCategory': None},
            {'predictions.successCategory': ''}
        ]
    }
    
    movies_to_update = list(movies_collection.find(query))
    total = len(movies_to_update)
    
    print(f"\n📊 Found {total} movies without predictions")
    print("=" * 50)
    
    updated = 0
    errors = 0
    
    for i, movie in enumerate(movies_to_update, 1):
        try:
            # Rule-based classification
            success_category = classify_movie(
                movie.get('budget'),
                movie.get('revenue'),
                movie.get('voteAverage')
            )
            
            # Prepare predictions object
            predictions = movie.get('predictions', {}) or {}
            predictions['successCategory'] = success_category
            
            # Try ML prediction if possible
            try:
                features = prepare_features(movie)
                feature_df = pd.DataFrame([features])[feature_columns]
                
                # Revenue prediction
                pred_revenue = revenue_model.predict(feature_df)[0]
                predictions['predictedRevenue'] = float(max(0, pred_revenue))
                
                # Classification
                if hasattr(classifier_model, 'predict_proba'):
                    proba = classifier_model.predict_proba(feature_df)[0]
                    predictions['confidence'] = float(max(proba) * 100)
            except Exception as ml_error:
                # If ML fails, use rule-based only
                pass
            
            # Update in database
            movies_collection.update_one(
                {'_id': movie['_id']},
                {'$set': {'predictions': predictions}}
            )
            
            updated += 1
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] Processed... ({updated} updated, {errors} errors)")
                
        except Exception as e:
            errors += 1
            print(f"Error on {movie.get('title', 'Unknown')}: {e}")
    
    print("\n" + "=" * 50)
    print(f"✅ Batch prediction complete!")
    print(f"   Updated: {updated}")
    print(f"   Errors: {errors}")
    
    # Show stats by industry
    pipeline = [
        {'$group': {
            '_id': '$industry',
            'total': {'$sum': 1},
            'withPredictions': {
                '$sum': {'$cond': [{'$ne': ['$predictions.successCategory', None]}, 1, 0]}
            }
        }}
    ]
    stats = list(movies_collection.aggregate(pipeline))
    print("\n📈 Current Status by Industry:")
    for s in stats:
        pct = (s['withPredictions'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"   {s['_id']}: {s['withPredictions']}/{s['total']} ({pct:.0f}%)")


if __name__ == '__main__':
    run_batch_predictions()
