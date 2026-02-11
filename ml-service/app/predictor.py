"""
GreenLit Go - Movie Prediction Service
Loads trained models and provides prediction APIs
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Success categories
SUCCESS_CATEGORIES = ['Average', 'Blockbuster', 'Flop', 'Hit', 'Super Hit']

class MoviePredictor:
    """Load trained models and make predictions"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            # Default to app/models directory
            self.models_dir = Path(__file__).parent / 'models'
        else:
            self.models_dir = Path(models_dir)
        
        self.revenue_model = None
        self.classifier_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.demo_mode = False
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from disk"""
        try:
            # Check if models exist
            revenue_path = self.models_dir / 'revenue_model.pkl'
            classifier_path = self.models_dir / 'classifier_model.pkl'
            scaler_path = self.models_dir / 'scaler.pkl'
            encoder_path = self.models_dir / 'label_encoder.pkl'
            features_path = self.models_dir / 'feature_columns.pkl'
            
            if not revenue_path.exists():
                print("⚠️ Models not found. Running in demo mode.")
                self.demo_mode = True
                return
            
            # Load models
            with open(revenue_path, 'rb') as f:
                self.revenue_model = pickle.load(f)
            
            with open(classifier_path, 'rb') as f:
                self.classifier_model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"✓ Models loaded from {self.models_dir}")
            print(f"  Revenue model: {type(self.revenue_model).__name__}")
            print(f"  Classifier: {type(self.classifier_model).__name__}")
            print(f"  Features: {len(self.feature_columns)} columns")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.demo_mode = True
    
    # ----------------------------------------------------------------
    #  Safe type conversion helpers
    # ----------------------------------------------------------------
    def _safe_float(self, val, default=0):
        """Safely convert any value to float"""
        if val is None or val == '':
            return float(default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)
    
    def _safe_int(self, val, default=0):
        """Safely convert any value to int"""
        if val is None or val == '':
            return int(default)
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return int(default)
    
    # ----------------------------------------------------------------
    #  Smart Feature Imputation for Upcoming Movies
    # ----------------------------------------------------------------
    def _impute_upcoming_features(self, data, features):
        """
        Smartly estimate unknown features for upcoming movies.
        
        Problem: The model was trained on RELEASED movies that have actual
        vote_count, popularity, opening_gross, imdb_rating, metascore.
        For UPCOMING movies, these are unknown and default to 0, making
        the model think every movie is obscure → terrible predictions.
        
        Solution: Estimate these features based on budget, cast quality,
        franchise status, trailer buzz, and genre patterns.
        
        Training data stats (for calibration):
          budget: mean=$54.5M  |  vote_average: mean=6.58
          vote_count: mean=4042  |  popularity: mean=5.09
          opening_gross: mean=$65M  |  imdb_rating: mean=6.45
          metascore: mean=39.1  |  trailer_views: mean=5.8M
        """
        budget = features.get('budget', 0)
        cast_pop = features.get('cast_popularity', 0)
        director_pop = features.get('director_popularity', 0)
        is_sequel = features.get('is_sequel', 0)
        trailer_views = features.get('trailer_views', 0)
        
        # Combined talent score (0-100 scale)
        talent_score = min((cast_pop + director_pop) / 2, 100) if (cast_pop + director_pop) > 0 else 15
        
        # Multipliers based on signals
        talent_mult = 1.0 + (talent_score / 100) * 0.5  # up to 1.5x
        sequel_mult = 1.3 if is_sequel else 1.0
        
        trailer_mult = 1.0
        if trailer_views > 100_000_000:
            trailer_mult = 1.8
        elif trailer_views > 50_000_000:
            trailer_mult = 1.5
        elif trailer_views > 10_000_000:
            trailer_mult = 1.3
        elif trailer_views > 1_000_000:
            trailer_mult = 1.1
        
        # ------ VOTE AVERAGE (IMDb-like rating, mean=6.58) ------
        if budget > 200_000_000:
            base_rating = 6.8
        elif budget > 100_000_000:
            base_rating = 6.6
        elif budget > 50_000_000:
            base_rating = 6.4
        elif budget > 20_000_000:
            base_rating = 6.2
        else:
            base_rating = 6.0
        
        # Talent boosts rating slightly
        talent_boost = (talent_score / 100) * 0.8  # up to +0.8
        vote_avg = min(base_rating + talent_boost, 8.0)
        
        if 'vote_average' in features and features['vote_average'] <= 0.1:
            features['vote_average'] = round(vote_avg, 1)
        
        # ------ VOTE COUNT (mean=4042, scale=5121) ------
        # Budget tiers calibrated to training data distribution
        if budget > 200_000_000:
            base_votes = 8000   # Well above mean for big films
        elif budget > 100_000_000:
            base_votes = 5000   # Above mean
        elif budget > 50_000_000:
            base_votes = 3000   # Near mean
        elif budget > 20_000_000:
            base_votes = 1500   # Below mean
        elif budget > 5_000_000:
            base_votes = 500    # Small film
        else:
            base_votes = 200    # Micro budget
        
        estimated_votes = base_votes * talent_mult * sequel_mult * trailer_mult
        
        if 'vote_count' in features and features['vote_count'] <= 0.1:
            features['vote_count'] = int(estimated_votes)
        
        # ------ POPULARITY (mean=5.09, scale=5.30) ------
        # TMDB popularity is surprisingly low in training data!
        if budget > 200_000_000:
            base_pop = 12
        elif budget > 100_000_000:
            base_pop = 8
        elif budget > 50_000_000:
            base_pop = 5
        elif budget > 20_000_000:
            base_pop = 3
        else:
            base_pop = 1.5
        
        estimated_pop = base_pop * talent_mult * sequel_mult * trailer_mult
        
        if 'popularity' in features and features['popularity'] <= 0.1:
            features['popularity'] = round(estimated_pop, 1)
        
        # ------ OPENING GROSS (mean=$65M, scale=$98.8M) ------
        # Heavily correlated with budget — calibrated to training data
        if budget > 200_000_000:
            base_opening = budget * 0.35  # ~$70M-$105M
        elif budget > 100_000_000:
            base_opening = budget * 0.28  # ~$28M-$56M
        elif budget > 50_000_000:
            base_opening = budget * 0.22  # ~$11M-$22M
        elif budget > 20_000_000:
            base_opening = budget * 0.18  # ~$3.6M-$9M
        else:
            base_opening = budget * 0.12  # ~$0.6M-$2.4M
        
        opening_mult = sequel_mult * (1.0 + (talent_score / 100) * 0.3)
        if trailer_views > 100_000_000:
            opening_mult *= 1.4
        elif trailer_views > 50_000_000:
            opening_mult *= 1.2
        elif trailer_views > 10_000_000:
            opening_mult *= 1.1
        
        estimated_opening = base_opening * opening_mult
        
        if 'opening_gross' in features and features['opening_gross'] <= 0.1:
            features['opening_gross'] = int(estimated_opening)
        
        # ------ IMDB RATING (mean=6.45, scale=1.28) ------
        if 'imdb_rating' in features and features['imdb_rating'] <= 0.1:
            features['imdb_rating'] = vote_avg  # Same as vote_average estimate
        
        # ------ METASCORE (mean=39.1, scale=29.1) ------
        # Many movies have 0 metascore in training data (missing data)
        # Keep conservative: base of 35 + talent bonus
        metascore = 35 + (talent_score / 100) * 20  # 35-55 range
        if budget > 150_000_000:
            metascore += 5
        
        if 'metascore' in features and features['metascore'] <= 0.1:
            features['metascore'] = round(min(metascore, 70), 0)
        
        return features
    
    # ----------------------------------------------------------------
    #  Feature Preparation
    # ----------------------------------------------------------------
    def _prepare_features(self, data):
        """Prepare feature vector from input data"""
        # Create a feature dict with all columns initialized to 0
        features = {col: 0 for col in self.feature_columns}
        
        # Basic features - cast all to proper numeric types
        features['budget'] = self._safe_float(data.get('budget', 0))
        features['runtime'] = self._safe_float(data.get('runtime', 120), 120)
        features['vote_average'] = self._safe_float(data.get('voteAverage', 0), 0)
        features['vote_count'] = self._safe_float(data.get('voteCount', 0), 0)
        features['popularity'] = self._safe_float(data.get('popularity', 0), 0)
        features['year'] = self._safe_int(data.get('year', 2026), 2026)
        
        # Release month
        release_month = data.get('releaseMonth', 6)
        if isinstance(release_month, str) and release_month:
            months = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                      'May': 5, 'June': 6, 'July': 7, 'August': 8,
                      'September': 9, 'October': 10, 'November': 11, 'December': 12}
            release_month = months.get(release_month, 6)
        else:
            release_month = self._safe_int(release_month, 6)
        features['release_month'] = release_month
        
        # Seasonal indicators
        features['is_summer_release'] = 1 if release_month in [5, 6, 7, 8] else 0
        features['is_holiday_release'] = 1 if release_month in [11, 12] else 0
        
        # Genre features
        genres = data.get('genres', [])
        if isinstance(genres, str):
            genres = [genres]
        
        genre_mapping = {
            'Action': 'genre_action',
            'Comedy': 'genre_comedy',
            'Drama': 'genre_drama',
            'Horror': 'genre_horror',
            'Thriller': 'genre_thriller',
            'Science Fiction': 'genre_science_fiction',
            'Sci-Fi': 'genre_science_fiction',
            'Animation': 'genre_animation',
            'Romance': 'genre_romance',
            'Adventure': 'genre_adventure'
        }
        
        for genre in genres:
            col_name = genre_mapping.get(genre)
            if col_name and col_name in features:
                features[col_name] = 1
        
        # Industry features
        industry = data.get('industry', 'hollywood').lower()
        industry_col = f'industry_{industry}'
        if industry_col in features:
            features[industry_col] = 1
        
        # Director popularity - use TMDB popularity from talent search
        director_pop = self._safe_float(data.get('directorPopularity', 0))
        # Also check if director object was sent with popularity
        director_data = data.get('director', {})
        if isinstance(director_data, dict) and director_data.get('popularity'):
            director_pop = self._safe_float(director_data.get('popularity', 0))
        features['director_popularity'] = director_pop
        
        # Cast popularity - use TMDB popularity from talent search
        cast_pop = self._safe_float(data.get('castPopularity', 0))
        # Also check if cast members were sent with individual popularity
        cast_members = data.get('castMembers', [])
        if isinstance(cast_members, list) and len(cast_members) > 0:
            pop_scores = [self._safe_float(m.get('popularity', 0)) for m in cast_members if isinstance(m, dict)]
            if pop_scores:
                cast_pop = max(cast_pop, sum(pop_scores) / len(pop_scores))
        # Also accept combined cast score (0-100 from TalentSearch)
        cast_score = self._safe_float(data.get('castScore', 0))
        if cast_score > 0 and cast_pop == 0:
            cast_pop = cast_score  # Use cast score as popularity proxy
        features['cast_popularity'] = cast_pop
        
        # Sequel
        is_sequel = data.get('isSequel', False)
        features['is_sequel'] = 1 if (is_sequel is True or is_sequel == 'true' or is_sequel == 1 or is_sequel == '1') else 0
        
        # IMDB rating and metascore (will be imputed if 0)
        features['imdb_rating'] = self._safe_float(data.get('imdbRating', 0))
        features['metascore'] = self._safe_float(data.get('metascore', 0))
        features['opening_gross'] = self._safe_float(data.get('openingGross', 0))
        
        # Trailer metrics (if available)
        features['trailer_views'] = self._safe_float(data.get('trailerViews', 0))
        features['trailer_likes'] = self._safe_float(data.get('trailerLikes', 0))
        features['trailer_comments'] = self._safe_float(data.get('trailerComments', 0))
        features['trailer_views_log'] = np.log1p(features['trailer_views'])
        
        if features['trailer_views'] > 0:
            features['trailer_engagement_ratio'] = (
                (features['trailer_likes'] + features['trailer_comments']) / 
                features['trailer_views'] * 100
            )
        
        # ======= SMART IMPUTATION FOR UPCOMING MOVIES =======
        # This is the key fix: estimate unknown features from known signals
        features = self._impute_upcoming_features(data, features)
        
        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([features])[self.feature_columns]
        
        return feature_df
    
    # ----------------------------------------------------------------
    #  Prediction
    # ----------------------------------------------------------------
    def predict(self, data):
        """Make prediction for a movie"""
        if self.demo_mode:
            return self._demo_prediction(data)
        
        try:
            # Prepare features (includes smart imputation)
            X = self._prepare_features(data)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict revenue (log-transformed)
            log_revenue_pred = self.revenue_model.predict(X_scaled)[0]
            revenue_pred = np.expm1(log_revenue_pred)  # Inverse log transform
            
            # Ensure revenue is non-negative
            revenue_pred = max(revenue_pred, 0)
            
            # ======= POST-PREDICTION SANITY CHECKS =======
            budget = self._safe_float(data.get('budget', 0))
            is_sequel = data.get('isSequel', False)
            is_sequel = (is_sequel is True or is_sequel == 'true' or is_sequel == 1 or is_sequel == '1')
            industry = data.get('industry', 'hollywood').lower()
            trailer_views = self._safe_float(data.get('trailerViews', 0))
            
            # ======= INDUSTRY-SPECIFIC REVENUE SCALING =======
            # Model is trained on 72% Hollywood data (avg revenue $265M)
            # Indian cinema has fundamentally different revenue scales
            # Scale predictions to match realistic industry ranges
            industry_scale = {
                'hollywood':  1.0,    # Base - model trained primarily on this
                'bollywood':  0.25,   # Avg revenue $42M vs $265M Hollywood
                'tollywood':  0.18,   # Avg revenue $22M
                'kollywood':  0.15,   # Avg revenue $16M
                'mollywood':  0.08,   # Avg revenue $5M
            }
            scale_factor = industry_scale.get(industry, 1.0)
            
            if industry != 'hollywood':
                revenue_pred = revenue_pred * scale_factor
            
            # Revenue ceiling based on budget 
            if budget > 0:
                max_revenue_mult = 12  # Max 12x budget
                if is_sequel and trailer_views > 100_000_000:
                    max_revenue_mult = 15
                if industry in ['bollywood', 'tollywood', 'kollywood', 'mollywood']:
                    max_revenue_mult = 10  # Indian cinema has lower ceilings
                
                revenue_pred = min(revenue_pred, budget * max_revenue_mult)
            
            # Revenue floor
            if budget > 0:
                revenue_pred = max(revenue_pred, budget * 0.1)
            
            # Predict success category
            category_encoded = self.classifier_model.predict(X_scaled)[0]
            category = self.label_encoder.inverse_transform([category_encoded])[0]
            
            # Get prediction probabilities
            category_probs = self.classifier_model.predict_proba(X_scaled)[0]
            confidence = float(max(category_probs) * 100)
            
            # Calculate ROI
            roi = ((revenue_pred - budget) / budget * 100) if budget > 0 else 0
            
            # ======= INDUSTRY-AWARE CATEGORY SANITY CHECK =======
            # Different industries have different revenue thresholds for categories
            blockbuster_threshold = {
                'hollywood': 500_000_000.0, 'bollywood': 75_000_000.0,
                'tollywood': 40_000_000.0, 'kollywood': 30_000_000.0,
                'mollywood': 10_000_000.0
            }
            superhit_threshold = {
                'hollywood': 200_000_000.0, 'bollywood': 30_000_000.0,
                'tollywood': 15_000_000.0, 'kollywood': 10_000_000.0,
                'mollywood': 5_000_000.0
            }
            hit_threshold = {
                'hollywood': 100_000_000.0, 'bollywood': 15_000_000.0,
                'tollywood': 8_000_000.0, 'kollywood': 5_000_000.0,
                'mollywood': 2_000_000.0
            }
            
            bb_thresh = blockbuster_threshold.get(industry, 500_000_000.0)
            sh_thresh = superhit_threshold.get(industry, 200_000_000.0)
            ht_thresh = hit_threshold.get(industry, 100_000_000.0)
            
            # Override category based on revenue + ROI
            if revenue_pred >= bb_thresh and roi > 150:
                category = 'Blockbuster'
                confidence = max(confidence, 75)
            elif revenue_pred >= sh_thresh and roi > 100:
                if category in ('Flop', 'Average'):
                    category = 'Super Hit'
                    confidence = max(confidence, 60)
            elif revenue_pred >= ht_thresh and roi > 50:
                if category in ('Flop', 'Average'):
                    category = 'Hit'
                    confidence = max(confidence, 50)
            elif roi > 200:
                # High ROI regardless of absolute revenue
                if category in ('Flop', 'Average'):
                    category = 'Super Hit'
                    confidence = max(confidence, 55)
            elif roi > 50:
                if category == 'Flop':
                    category = 'Average'
            elif roi < -30:
                category = 'Flop'
                confidence = max(confidence, 55)
            
            # Downgrade inflated categories based on industry thresholds
            if revenue_pred < sh_thresh and category == 'Blockbuster':
                category = 'Super Hit' if roi > 150 else 'Hit'
            if revenue_pred < ht_thresh and category in ('Blockbuster', 'Super Hit'):
                category = 'Hit' if roi > 50 else 'Average'
            
            # Feature importance (per-prediction, not global)
            feature_importance = self._get_feature_importance(data, X_scaled)
            
            return {
                'predictions': {
                    'successCategory': category,
                    'predictedRevenue': float(revenue_pred),
                    'predictedROI': round(float(roi), 1),
                    'confidence': round(confidence, 1),
                    'featureImportance': feature_importance
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._demo_prediction(data)
    
    # ----------------------------------------------------------------
    #  Feature Importance (Normalized)
    # ----------------------------------------------------------------
    def _get_feature_importance(self, data, X_scaled=None):
        """Get PER-PREDICTION feature importance for XAI explanation.
        
        Instead of showing the same global importances for every movie,
        we compute contribution = |scaled_feature_value| x global_importance.
        This way, a big-budget movie shows Budget as highly important,
        while a sequel shows Sequel/Franchise as important.
        """
        importance_list = []
        
        # Human-readable feature names
        display_names = {
            'budget': 'Budget',
            'runtime': 'Runtime',
            'vote_average': 'Audience Rating',
            'vote_count': 'Audience Engagement',
            'popularity': 'Popularity Score',
            'year': 'Release Year',
            'release_month': 'Release Month',
            'is_summer_release': 'Summer Release',
            'is_holiday_release': 'Holiday Release',
            'genre_action': 'Action Genre',
            'genre_comedy': 'Comedy Genre',
            'genre_drama': 'Drama Genre',
            'genre_horror': 'Horror Genre',
            'genre_thriller': 'Thriller Genre',
            'genre_science_fiction': 'Sci-Fi Genre',
            'genre_animation': 'Animation Genre',
            'genre_romance': 'Romance Genre',
            'genre_adventure': 'Adventure Genre',
            'director_popularity': 'Director Power',
            'cast_popularity': 'Cast Star Power',
            'is_sequel': 'Sequel/Franchise',
            'imdb_rating': 'IMDb Rating',
            'metascore': 'Metascore',
            'opening_gross': 'Opening Weekend',
            'trailer_views': 'Trailer Views',
            'trailer_likes': 'Trailer Likes',
            'trailer_comments': 'Trailer Comments',
            'trailer_engagement_ratio': 'Trailer Engagement',
            'trailer_views_log': 'Trailer Buzz',
            'industry_hollywood': 'Hollywood Market',
            'industry_bollywood': 'Bollywood Market',
            'industry_tollywood': 'Tollywood Market',
            'industry_kollywood': 'Kollywood Market',
            'industry_mollywood': 'Mollywood Market'
        }
        
        # Get global feature importances from the model
        global_importances = None
        
        if hasattr(self.revenue_model, 'feature_importances_'):
            global_importances = self.revenue_model.feature_importances_
        elif hasattr(self.revenue_model, 'named_estimators_'):
            all_imp = np.zeros(len(self.feature_columns))
            n = 0
            for name, model in self.revenue_model.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    all_imp += model.feature_importances_
                    n += 1
            if n > 0:
                global_importances = all_imp / n
        elif hasattr(self.revenue_model, 'estimators_'):
            all_imp = np.zeros(len(self.feature_columns))
            n = 0
            for model in self.revenue_model.estimators_:
                if hasattr(model, 'feature_importances_'):
                    all_imp += model.feature_importances_
                    n += 1
            if n > 0:
                global_importances = all_imp / n
        
        if global_importances is not None and X_scaled is not None:
            # ====== PER-PREDICTION IMPORTANCE ======
            # Contribution = |scaled_feature_value| x global_importance
            # This makes the XAI unique for each movie input
            try:
                feature_values = np.abs(X_scaled.ravel())
                contributions = global_importances * feature_values
                
                # Normalize to sum to 1.0
                total_contribution = contributions.sum()
                if total_contribution > 0:
                    normalized = contributions / total_contribution
                else:
                    total = global_importances.sum()
                    normalized = global_importances / total if total > 0 else global_importances
            except Exception:
                # Fallback to global importances
                total = global_importances.sum()
                normalized = global_importances / total if total > 0 else global_importances
        elif global_importances is not None:
            # No scaled features — fallback to global importances
            total = global_importances.sum()
            normalized = global_importances / total if total > 0 else global_importances
        else:
            # No model importances at all
            return [
                {'feature': 'Budget', 'impact': 0.25},
                {'feature': 'Director Track Record', 'impact': 0.18},
                {'feature': 'Cast Popularity', 'impact': 0.15},
                {'feature': 'Genre', 'impact': 0.12},
                {'feature': 'Release Timing', 'impact': 0.10},
                {'feature': 'Sequel Factor', 'impact': 0.08},
                {'feature': 'Pre-release Buzz', 'impact': 0.07}
            ]
        
        # Get top 7 contributing features
        sorted_idx = np.argsort(normalized)[::-1][:7]
        for idx in sorted_idx:
            feature_name = self.feature_columns[idx]
            importance_list.append({
                'feature': display_names.get(feature_name, feature_name.replace('_', ' ').title()),
                'impact': round(float(normalized[idx]), 4)
            })
        
        return importance_list
    
    # ----------------------------------------------------------------
    #  Demo Prediction (fallback)
    # ----------------------------------------------------------------
    def _demo_prediction(self, data):
        """Demo prediction when models not loaded"""
        budget = self._safe_float(data.get('budget', 100000000), 100000000)
        
        # Simple heuristic-based prediction
        predicted_revenue = budget * 2.5
        
        genres = data.get('genres', [])
        if 'Action' in genres or 'Adventure' in genres:
            predicted_revenue *= 1.3
        if data.get('isSequel'):
            predicted_revenue *= 1.2
        
        roi = ((predicted_revenue - budget) / budget * 100) if budget > 0 else 0
        
        if roi > 300:
            category = 'Blockbuster'
        elif roi > 150:
            category = 'Hit'
        elif roi > 50:
            category = 'Average'
        else:
            category = 'Flop'
        
        return {
            'predictions': {
                'successCategory': category,
                'predictedRevenue': predicted_revenue,
                'predictedROI': round(roi, 1),
                'confidence': 75,
                'featureImportance': [
                    {'feature': 'Budget', 'impact': 0.25},
                    {'feature': 'Genre', 'impact': 0.18},
                    {'feature': 'Cast Popularity', 'impact': 0.15},
                    {'feature': 'Director', 'impact': 0.12},
                    {'feature': 'Release Month', 'impact': 0.10}
                ]
            },
            'message': 'Demo prediction. Train ML models for improved accuracy.'
        }
    
    # ----------------------------------------------------------------
    #  SHAP Explanation
    # ----------------------------------------------------------------
    def explain(self, data):
        """Get SHAP explanation for prediction"""
        prediction = self.predict(data)
        
        return {
            'explanation': prediction['predictions'].get('featureImportance', []),
            'prediction': prediction['predictions']
        }
    
    # ----------------------------------------------------------------
    #  Optimal Release
    # ----------------------------------------------------------------
    def get_optimal_release(self, data):
        """Recommend optimal release timing"""
        genre = data.get('genre', 'Action')
        industry = data.get('industry', 'hollywood')
        
        recommendations = {
            'Action': {'months': [5, 6, 7, 12], 'reason': 'Summer blockbuster season and holidays'},
            'Horror': {'months': [10], 'reason': 'Halloween season drives horror viewership'},
            'Comedy': {'months': [6, 7, 11, 12], 'reason': 'Summer and holiday family viewing'},
            'Drama': {'months': [10, 11, 12], 'reason': 'Awards season consideration'},
            'Animation': {'months': [6, 7, 11, 12], 'reason': 'School breaks and holidays'},
            'Romance': {'months': [2, 12], 'reason': "Valentine's Day and Christmas season"}
        }
        
        rec = recommendations.get(genre, {'months': [6, 7, 12], 'reason': 'Peak entertainment periods'})
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        optimal_months = [month_names[m-1] for m in rec['months']]
        
        return {
            'optimalMonths': optimal_months,
            'reason': rec['reason'],
            'genre': genre,
            'industry': industry,
            'recommendation': f"For {genre} movies in {industry.title()}, consider releasing in {', '.join(optimal_months)}."
        }
