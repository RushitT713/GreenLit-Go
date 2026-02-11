from flask import Blueprint, request, jsonify
import numpy as np
import os
from .predictor import MoviePredictor

api_bp = Blueprint('api', __name__)

# Initialize predictor (will use demo mode if models not found)
predictor = MoviePredictor()

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict movie success, revenue, and rating.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get prediction
        result = predictor.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/explain', methods=['POST'])
def explain():
    """
    Get SHAP explanation for a prediction.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get explanation
        result = predictor.explain(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/optimal-release', methods=['POST'])
def optimal_release():
    """
    Recommend optimal release date based on genre and industry.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get optimal release recommendations
        result = predictor.get_optimal_release(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict for multiple movies at once.
    """
    try:
        data = request.get_json()
        
        if not data or 'movies' not in data:
            return jsonify({'error': 'No movies data provided'}), 400
        
        results = []
        for movie in data['movies']:
            result = predictor.predict(movie)
            results.append(result)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
