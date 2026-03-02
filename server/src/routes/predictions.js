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

// Competitive window analysis — fetches upcoming movies from TMDB
router.post('/competitive-analysis', async (req, res) => {
  try {
    const { releaseDate, genres, industry } = req.body;
    
    if (!releaseDate) {
      return res.status(400).json({ error: 'Release date is required' });
    }

    const TMDB_API_KEY = process.env.TMDB_API_KEY;
    const targetDate = new Date(releaseDate);
    const startWindow = new Date(targetDate);
    startWindow.setDate(startWindow.getDate() - 14);
    const endWindow = new Date(targetDate);
    endWindow.setDate(endWindow.getDate() + 14);

    const formatDate = (d) => d.toISOString().split('T')[0];

    let competingMovies = [];

    if (TMDB_API_KEY) {
      // Fetch upcoming movies from TMDB discover API
      try {
        // Map industry to TMDB region/language
        const regionMap = {
          'hollywood': { region: 'US', language: 'en' },
          'bollywood': { region: 'IN', language: 'hi' },
          'tollywood': { region: 'IN', language: 'te' },
          'kollywood': { region: 'IN', language: 'ta' },
          'mollywood': { region: 'IN', language: 'ml' }
        };
        const regionConfig = regionMap[industry] || {};

        // Fetch up to 2 pages to get more movies
        for (let page = 1; page <= 2; page++) {
          const params = {
            api_key: TMDB_API_KEY,
            'primary_release_date.gte': formatDate(startWindow),
            'primary_release_date.lte': formatDate(endWindow),
            sort_by: 'popularity.desc',
            page,
            include_adult: false
          };
          if (regionConfig.region) params.region = regionConfig.region;
          if (regionConfig.language) params.with_original_language = regionConfig.language;

          const response = await axios.get('https://api.themoviedb.org/3/discover/movie', { params });
          
          if (response.data?.results) {
            competingMovies.push(...response.data.results);
          }
          // Stop if we got all results
          if (response.data?.total_pages <= page) break;
        }

        // Map TMDB genre IDs to names
        const genreMap = {
          28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
          80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
          14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
          9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi', 10770: 'TV Movie',
          53: 'Thriller', 10752: 'War', 37: 'Western'
        };

        // Transform TMDB movies to our format
        competingMovies = competingMovies.map(movie => ({
          title: movie.title,
          releaseDate: movie.release_date,
          genres: (movie.genre_ids || []).map(id => genreMap[id] || 'Other'),
          popularity: movie.popularity,
          voteAverage: movie.vote_average,
          posterPath: movie.poster_path 
            ? `https://image.tmdb.org/t/p/w185${movie.poster_path}` 
            : null,
          overview: movie.overview?.substring(0, 120),
          tmdbId: movie.id
        }));
      } catch (tmdbError) {
        console.error('TMDB fetch error:', tmdbError.message);
      }
    }

    // Fallback: also check local DB
    if (competingMovies.length === 0) {
      const query = {
        releaseDate: { $gte: startWindow, $lte: endWindow }
      };
      if (industry) query.industry = industry;

      const dbMovies = await Movie.find(query)
        .select('title releaseDate genres budget predictions posterPath')
        .sort({ releaseDate: 1 });

      competingMovies = dbMovies.map(m => ({
        title: m.title,
        releaseDate: m.releaseDate,
        genres: m.genres || [],
        budget: m.budget,
        posterPath: m.posterUrl || null
      }));
    }

    // Analyze competition level
    let directCompetitors = [];
    let totalPopularity = 0;
    const hasGenreFilter = genres && genres.length > 0;
    
    for (const movie of competingMovies) {
      const movieGenres = movie.genres || [];
      totalPopularity += movie.popularity || 0;

      if (!hasGenreFilter) {
        // No genre filter — every movie is a potential competitor
        directCompetitors.push({
          ...movie,
          sharedGenres: movieGenres,
          threatLevel: 'Medium'
        });
      } else {
        const sharedGenres = movieGenres.filter(g => genres.includes(g));
        if (sharedGenres.length > 0) {
          directCompetitors.push({
            ...movie,
            sharedGenres,
            threatLevel: sharedGenres.length >= 2 ? 'High' : 'Medium'
          });
        }
      }
    }

    // Competition score (0-10) based on three factors:
    const movieCount = competingMovies.length;
    const directCount = directCompetitors.length;

    // 1. Volume: how many movies in this window (0-4 pts)
    //    0 movies → 0, 5 → ~2.5, 15 → ~3.5, 30+ → ~4
    const volumeScore = movieCount === 0 ? 0 : Math.min(4, Math.log10(movieCount + 1) * 2.6);

    // 2. Genre/density score (0-3 pts)
    let genreScore = 0;
    if (hasGenreFilter && movieCount > 0) {
      // With genre filter: what fraction of releases share the user's genre
      genreScore = Math.min(3, (directCount / movieCount) * 5 + (directCount >= 3 ? 0.5 : 0));
    } else if (movieCount > 0) {
      // Without genre filter: use total popularity density of the window
      // This differentiates "20 low-profile movies" from "20 blockbusters"
      const avgPopularity = totalPopularity / movieCount;
      genreScore = Math.min(3, (avgPopularity / 40) * 3);
    }

    // 3. Popularity pressure from top competitors (0-3 pts)
    const competitorPool = hasGenreFilter ? directCompetitors : competingMovies;
    const sortedByPop = [...competitorPool].sort((a, b) => (b.popularity || 0) - (a.popularity || 0));
    const topN = sortedByPop.slice(0, Math.min(5, sortedByPop.length));
    const avgTopPopularity = topN.length > 0
      ? topN.reduce((sum, m) => sum + (m.popularity || 0), 0) / topN.length
      : 0;
    // Baseline: avg top-5 popularity of 50 = max pressure (3 pts)
    const popularityScore = Math.min(3, (avgTopPopularity / 50) * 3);

    const normalizedScore = Math.min(10, Math.round((volumeScore + genreScore + popularityScore) * 10) / 10);

    res.json({
      releaseWindow: {
        start: startWindow,
        end: endWindow,
        targetDate
      },
      totalCompetitors: competingMovies.length,
      competingMovies: competingMovies.slice(0, 20),
      directCompetitors: directCompetitors.slice(0, 10),
      competitionScore: normalizedScore,
      recommendation: normalizedScore > 7 ? 'High Competition - Consider Alternative Date' :
                      normalizedScore > 4 ? 'Moderate Competition' : 'Favorable Window'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
