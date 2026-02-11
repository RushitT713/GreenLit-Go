const express = require('express');
const router = express.Router();
const axios = require('axios');
require('dotenv').config();

const TMDB_API_KEY = process.env.TMDB_API_KEY;
const TMDB_BASE_URL = 'https://api.themoviedb.org/3';

// Debug log on module load
console.log('Talents route loaded. TMDB API Key:', TMDB_API_KEY ? 'SET (' + TMDB_API_KEY.slice(0, 4) + '...)' : 'NOT SET');

// Search for people (actors/directors)
router.get('/search', async (req, res) => {
  try {
    const { q, limit = 10 } = req.query;
    
    if (!q || q.length < 2) {
      return res.json([]);
    }

    if (!TMDB_API_KEY) {
      console.error('TMDB_API_KEY is not set!');
      return res.status(500).json({ error: 'TMDB API key not configured' });
    }

    console.log('Searching TMDB for:', q);
    
    const response = await axios.get(`${TMDB_BASE_URL}/search/person`, {
      params: {
        api_key: TMDB_API_KEY,
        query: q,
        page: 1
      }
    });

    console.log('TMDB response:', response.data.results?.length, 'results');

    const results = response.data.results.slice(0, parseInt(limit)).map(person => ({
      id: person.id,
      name: person.name,
      profilePath: person.profile_path 
        ? `https://image.tmdb.org/t/p/w185${person.profile_path}`
        : null,
      popularity: person.popularity,
      knownFor: person.known_for_department,
      knownForMovies: (person.known_for || []).slice(0, 3).map(m => ({
        title: m.title || m.name,
        year: (m.release_date || m.first_air_date || '').slice(0, 4)
      }))
    }));

    res.json(results);
  } catch (error) {
    console.error('TMDB search error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to search people' });
  }
});

// Get director details with calculated metrics
router.get('/director/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Get person details
    const personRes = await axios.get(`${TMDB_BASE_URL}/person/${id}`, {
      params: { api_key: TMDB_API_KEY }
    });

    // Get movie credits
    const creditsRes = await axios.get(`${TMDB_BASE_URL}/person/${id}/movie_credits`, {
      params: { api_key: TMDB_API_KEY }
    });

    const person = personRes.data;
    const directorMovies = creditsRes.data.crew
      .filter(c => c.job === 'Director')
      .sort((a, b) => (b.release_date || '').localeCompare(a.release_date || ''));

    // Calculate director metrics
    const moviesWithRatings = directorMovies.filter(m => m.vote_average > 0 && m.vote_count > 50);
    const avgRating = moviesWithRatings.length > 0
      ? moviesWithRatings.reduce((sum, m) => sum + m.vote_average, 0) / moviesWithRatings.length
      : 0;

    // Calculate hit rate (movies with rating >= 7.0)
    const hits = moviesWithRatings.filter(m => m.vote_average >= 7.0).length;
    const hitRate = moviesWithRatings.length > 0 ? hits / moviesWithRatings.length : 0;

    // Calculate power score (0-100)
    const popularityScore = Math.min(person.popularity / 50 * 30, 30); // Max 30 points
    const hitRateScore = hitRate * 40; // Max 40 points
    const experienceScore = Math.min(directorMovies.length / 10 * 20, 20); // Max 20 points
    const ratingBonus = Math.max(0, (avgRating - 6) * 2.5); // Max 10 points
    const powerScore = Math.round(popularityScore + hitRateScore + experienceScore + ratingBonus);

    res.json({
      id: person.id,
      name: person.name,
      profilePath: person.profile_path 
        ? `https://image.tmdb.org/t/p/w185${person.profile_path}`
        : null,
      popularity: person.popularity,
      biography: person.biography?.slice(0, 300),
      metrics: {
        totalMovies: directorMovies.length,
        avgRating: parseFloat(avgRating.toFixed(1)),
        hitRate: parseFloat(hitRate.toFixed(2)),
        powerScore: Math.min(powerScore, 100)
      },
      recentMovies: directorMovies.slice(0, 5).map(m => ({
        id: m.id,
        title: m.title,
        year: (m.release_date || '').slice(0, 4),
        rating: m.vote_average,
        posterPath: m.poster_path 
          ? `https://image.tmdb.org/t/p/w92${m.poster_path}`
          : null
      }))
    });
  } catch (error) {
    console.error('Director fetch error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to fetch director details' });
  }
});

// Get actor details with popularity metrics
router.get('/actor/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Get person details
    const personRes = await axios.get(`${TMDB_BASE_URL}/person/${id}`, {
      params: { api_key: TMDB_API_KEY }
    });

    // Get movie credits
    const creditsRes = await axios.get(`${TMDB_BASE_URL}/person/${id}/movie_credits`, {
      params: { api_key: TMDB_API_KEY }
    });

    const person = personRes.data;
    const actorMovies = creditsRes.data.cast
      .filter(m => m.vote_count > 20)
      .sort((a, b) => (b.release_date || '').localeCompare(a.release_date || ''));

    // Calculate actor metrics
    const avgRating = actorMovies.length > 0
      ? actorMovies.slice(0, 20).reduce((sum, m) => sum + (m.vote_average || 0), 0) / Math.min(actorMovies.length, 20)
      : 0;

    // Calculate popularity score (0-100)
    const popularityScore = Math.min(person.popularity, 100);

    // Calculate star power score
    const baseScore = Math.min(person.popularity / 80 * 50, 50); // Max 50 points from popularity
    const experienceScore = Math.min(actorMovies.length / 30 * 25, 25); // Max 25 points
    const ratingBonus = Math.max(0, (avgRating - 6) * 6.25); // Max 25 points
    const starPower = Math.round(baseScore + experienceScore + ratingBonus);

    res.json({
      id: person.id,
      name: person.name,
      profilePath: person.profile_path 
        ? `https://image.tmdb.org/t/p/w185${person.profile_path}`
        : null,
      popularity: person.popularity,
      metrics: {
        totalMovies: actorMovies.length,
        avgRating: parseFloat(avgRating.toFixed(1)),
        popularityScore: Math.round(popularityScore),
        starPower: Math.min(starPower, 100)
      },
      recentMovies: actorMovies.slice(0, 5).map(m => ({
        id: m.id,
        title: m.title,
        year: (m.release_date || '').slice(0, 4),
        character: m.character,
        rating: m.vote_average
      }))
    });
  } catch (error) {
    console.error('Actor fetch error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to fetch actor details' });
  }
});

// Calculate combined cast score for multiple actors
router.post('/cast-score', async (req, res) => {
  try {
    const { actorIds } = req.body;

    if (!actorIds || !Array.isArray(actorIds) || actorIds.length === 0) {
      return res.json({ combinedScore: 0, actors: [] });
    }

    // Fetch all actor details
    const actorPromises = actorIds.slice(0, 10).map(async (id) => {
      try {
        const personRes = await axios.get(`${TMDB_BASE_URL}/person/${id}`, {
          params: { api_key: TMDB_API_KEY }
        });
        const person = personRes.data;
        
        // Normalize TMDB popularity to a 0-100 score
        // TMDB popularity is raw (e.g., Margot Robbie=34, Dwayne Johnson=100+, unknown=2)
        // Use logarithmic scaling: log(pop+1) / log(max_expected+1) * 100
        // Max expected popularity ~200 (top Hollywood stars)
        const rawPop = person.popularity || 0;
        const score = Math.min(Math.round((Math.log(rawPop + 1) / Math.log(201)) * 100), 100);
        
        return {
          id: person.id,
          name: person.name,
          profilePath: person.profile_path 
            ? `https://image.tmdb.org/t/p/w92${person.profile_path}`
            : null,
          popularity: person.popularity,
          score
        };
      } catch (err) {
        return null;
      }
    });

    const actors = (await Promise.all(actorPromises)).filter(Boolean);

    // Calculate combined score (weighted average - lead actors count more)
    let combinedScore = 0;
    if (actors.length > 0) {
      const weights = actors.map((_, i) => Math.max(1, 5 - i)); // [5, 4, 3, 2, 1, 1, ...]
      const totalWeight = weights.reduce((a, b) => a + b, 0);
      combinedScore = Math.round(
        actors.reduce((sum, actor, i) => sum + actor.score * weights[i], 0) / totalWeight
      );
    }

    res.json({
      combinedScore: Math.min(combinedScore, 100),
      actorCount: actors.length,
      actors
    });
  } catch (error) {
    console.error('Cast score error:', error.response?.data || error.message);
    res.status(500).json({ error: 'Failed to calculate cast score' });
  }
});

module.exports = router;
