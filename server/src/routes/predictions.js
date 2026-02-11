const express = require('express');
const router = express.Router();
const axios = require('axios');
const Movie = require('../models/Movie');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001';

// Get prediction for a new/upcoming movie
router.post('/predict', async (req, res) => {
  try {
    const movieData = req.body;
    
    // Validate required fields
    if (!movieData.title || !movieData.genres || !movieData.budget) {
      return res.status(400).json({ 
        error: 'Missing required fields: title, genres, budget' 
      });
    }

    // Call ML service for prediction
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/api/predict`, movieData);
    
    // ML service returns {predictions: {...}}, pass it through directly
    res.json({
      movie: movieData,
      ...mlResponse.data
    });
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Get explanation for a prediction
router.post('/explain', async (req, res) => {
  try {
    const movieData = req.body;
    
    // Call ML service for SHAP explanation
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/api/explain`, movieData);
    
    res.json(mlResponse.data);
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// What-if simulation
router.post('/simulate', async (req, res) => {
  try {
    const { baseMovie, modifications } = req.body;
    
    if (!baseMovie) {
      return res.status(400).json({ error: 'Base movie data is required' });
    }

    // Get base prediction
    const baseResponse = await axios.post(`${ML_SERVICE_URL}/api/predict`, baseMovie);
    
    // Get modified predictions for each scenario
    const scenarios = [];
    for (const mod of modifications || []) {
      const modifiedMovie = { ...baseMovie, ...mod.changes };
      const modResponse = await axios.post(`${ML_SERVICE_URL}/api/predict`, modifiedMovie);
      scenarios.push({
        name: mod.name,
        changes: mod.changes,
        predictions: modResponse.data
      });
    }
    
    res.json({
      basePrediction: baseResponse.data,
      scenarios
    });
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Optimal release date recommendation
router.post('/optimal-release', async (req, res) => {
  try {
    const movieData = req.body;
    
    // Call ML service for release date optimization
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/api/optimal-release`, movieData);
    
    res.json(mlResponse.data);
  } catch (error) {
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// Competitive window analysis
router.post('/competitive-analysis', async (req, res) => {
  try {
    const { releaseDate, genres, industry } = req.body;
    
    if (!releaseDate) {
      return res.status(400).json({ error: 'Release date is required' });
    }

    const targetDate = new Date(releaseDate);
    const startWindow = new Date(targetDate);
    startWindow.setDate(startWindow.getDate() - 14);
    const endWindow = new Date(targetDate);
    endWindow.setDate(endWindow.getDate() + 14);

    // Find competing movies in the window
    const query = {
      releaseDate: { $gte: startWindow, $lte: endWindow }
    };
    if (industry) query.industry = industry;

    const competitors = await Movie.find(query)
      .select('title releaseDate genres budget predictions posterPath')
      .sort({ releaseDate: 1 });

    // Analyze competition level
    let competitionScore = 0;
    const directCompetitors = [];
    
    for (const movie of competitors) {
      const sharedGenres = movie.genres.filter(g => genres?.includes(g));
      if (sharedGenres.length > 0) {
        directCompetitors.push({
          ...movie.toObject(),
          sharedGenres,
          threatLevel: sharedGenres.length >= 2 ? 'High' : 'Medium'
        });
        competitionScore += sharedGenres.length * 2;
      } else {
        competitionScore += 0.5;
      }
    }

    res.json({
      releaseWindow: {
        start: startWindow,
        end: endWindow,
        targetDate
      },
      totalCompetitors: competitors.length,
      directCompetitors,
      competitionScore,
      recommendation: competitionScore > 10 ? 'High Competition - Consider Alternative Date' :
                      competitionScore > 5 ? 'Moderate Competition' : 'Favorable Window'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
