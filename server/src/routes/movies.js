const express = require('express');
const router = express.Router();
const Movie = require('../models/Movie');

// Get all movies with pagination and filters
router.get('/', async (req, res) => {
  try {
    const {
      page = 1,
      limit = 20,
      industry,
      genre,
      year,
      status,
      search,
      category, // Accept 'category' from frontend
      successCategory, // Also accept 'successCategory' for backwards compatibility
      sortBy = 'popularity',
      sortOrder = 'desc'
    } = req.query;

    const query = {};
    
    // Search by title (case-insensitive regex)
    if (search) {
      query.title = { $regex: search, $options: 'i' };
    }
    
    // Industry filter
    if (industry) query.industry = industry.toLowerCase();
    
    // Genre filter (case-insensitive)
    if (genre) {
      const genreRegex = new RegExp(`^${genre}$`, 'i');
      query.genres = { $regex: genreRegex };
    }
    
    // Year filter
    if (year) query.year = parseInt(year);
    
    // Status filter
    if (status) query.status = status;
    
    // Success category filter (accept both param names, case-insensitive)
    const categoryValue = category || successCategory;
    if (categoryValue) {
      query['predictions.successCategory'] = { $regex: new RegExp(`^${categoryValue}$`, 'i') };
    }

    const sortOptions = {};
    sortOptions[sortBy] = sortOrder === 'asc' ? 1 : -1;

    const movies = await Movie.find(query)
      .sort(sortOptions)
      .limit(parseInt(limit))
      .skip((parseInt(page) - 1) * parseInt(limit))
      .select('-crew -cast.character');

    const total = await Movie.countDocuments(query);

    res.json({
      movies,
      pagination: {
        currentPage: parseInt(page),
        totalPages: Math.ceil(total / parseInt(limit)),
        pages: Math.ceil(total / parseInt(limit)), // Also include 'pages' for frontend compatibility
        total: total,
        totalMovies: total,
        hasMore: parseInt(page) * parseInt(limit) < total
      }
    });
  } catch (error) {
    console.error('Movies fetch error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Search movies
router.get('/search', async (req, res) => {
  try {
    const { q, limit = 10 } = req.query;
    
    if (!q) {
      return res.status(400).json({ error: 'Search query is required' });
    }

    const movies = await Movie.find(
      { $text: { $search: q } },
      { score: { $meta: 'textScore' } }
    )
      .sort({ score: { $meta: 'textScore' } })
      .limit(parseInt(limit))
      .select('title year posterPath industry genres voteAverage predictions.successCategory');

    res.json(movies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get trending movies
router.get('/trending', async (req, res) => {
  try {
    const { industry, limit = 10 } = req.query;
    const query = { status: 'Released' };
    if (industry) query.industry = industry;

    const movies = await Movie.find(query)
      .sort({ popularity: -1 })
      .limit(parseInt(limit))
      .select('title year posterPath backdropPath industry genres voteAverage revenue predictions');

    res.json(movies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get movies by category (Blockbuster, Hit, Flop, etc.)
router.get('/by-category/:category', async (req, res) => {
  try {
    const { category } = req.params;
    const { industry, limit = 20 } = req.query;
    
    const query = { 'predictions.successCategory': category };
    if (industry) query.industry = industry;

    const movies = await Movie.find(query)
      .sort({ revenue: -1 })
      .limit(parseInt(limit))
      .select('title year posterPath industry genres voteAverage revenue budget predictions');

    res.json(movies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get movies by genre
router.get('/by-genre/:genre', async (req, res) => {
  try {
    const { genre } = req.params;
    const { industry, limit = 20, sortBy = 'revenue' } = req.query;
    
    const query = { genres: genre };
    if (industry) query.industry = industry;

    const sortOptions = {};
    sortOptions[sortBy] = -1;

    const movies = await Movie.find(query)
      .sort(sortOptions)
      .limit(parseInt(limit))
      .select('title year posterPath industry genres voteAverage revenue predictions');

    res.json(movies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get upcoming movies
router.get('/upcoming', async (req, res) => {
  try {
    const { industry, limit = 20 } = req.query;
    const query = { 
      status: { $in: ['Upcoming', 'In Production'] },
      releaseDate: { $gte: new Date() }
    };
    if (industry) query.industry = industry;

    const movies = await Movie.find(query)
      .sort({ releaseDate: 1 })
      .limit(parseInt(limit))
      .select('title releaseDate posterPath backdropPath industry genres predictions');

    res.json(movies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get single movie by ID
router.get('/:id', async (req, res) => {
  try {
    const movie = await Movie.findById(req.params.id);
    
    if (!movie) {
      return res.status(404).json({ error: 'Movie not found' });
    }

    res.json(movie);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get movie by TMDB ID
router.get('/tmdb/:tmdbId', async (req, res) => {
  try {
    const movie = await Movie.findOne({ tmdbId: parseInt(req.params.tmdbId) });
    
    if (!movie) {
      return res.status(404).json({ error: 'Movie not found' });
    }

    res.json(movie);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get similar movies
router.get('/:id/similar', async (req, res) => {
  try {
    const movie = await Movie.findById(req.params.id);
    
    if (!movie) {
      return res.status(404).json({ error: 'Movie not found' });
    }

    const similarMovies = await Movie.find({
      _id: { $ne: movie._id },
      genres: { $in: movie.genres },
      industry: movie.industry
    })
      .sort({ voteAverage: -1 })
      .limit(10)
      .select('title year posterPath genres voteAverage predictions.successCategory');

    res.json(similarMovies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get stats overview
router.get('/stats/overview', async (req, res) => {
  try {
    const { industry } = req.query;
    const query = {};
    if (industry) query.industry = industry;

    const stats = await Movie.aggregate([
      { $match: query },
      {
        $group: {
          _id: null,
          totalMovies: { $sum: 1 },
          avgRating: { $avg: '$voteAverage' },
          avgBudget: { $avg: '$budget' },
          avgRevenue: { $avg: '$revenue' },
          totalRevenue: { $sum: '$revenue' }
        }
      }
    ]);

    const genreStats = await Movie.aggregate([
      { $match: query },
      { $unwind: '$genres' },
      {
        $group: {
          _id: '$genres',
          count: { $sum: 1 },
          avgRevenue: { $avg: '$revenue' },
          avgRating: { $avg: '$voteAverage' }
        }
      },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);

    const categoryStats = await Movie.aggregate([
      { $match: { ...query, 'predictions.successCategory': { $exists: true } } },
      {
        $group: {
          _id: '$predictions.successCategory',
          count: { $sum: 1 }
        }
      }
    ]);

    res.json({
      overview: stats[0] || {},
      genreStats,
      categoryStats
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// =============================================
// Personalized Movie Recommendation Engine
// =============================================

const MOOD_MAP = {
  'feel-good': {
    genres: ['Comedy', 'Drama', 'Animation', 'Romance', 'Family', 'Adventure', 'Music'],
    weights: { Comedy: 1.5, Animation: 1.3, Romance: 1.2, Family: 1.4, Drama: 1.0 },
    minRating: 7.0,
    sortBoost: 'voteAverage',
    description: 'Uplifting movies that leave you smiling'
  },
  'thrilling': {
    genres: ['Thriller', 'Action', 'Crime', 'Mystery'],
    weights: { Thriller: 1.5, Action: 1.3, Crime: 1.2, Mystery: 1.4 },
    minRating: 6.5,
    sortBoost: 'popularity',
    description: 'Edge-of-your-seat suspense and action'
  },
  'mind-bending': {
    genres: ['Science Fiction', 'Mystery', 'Thriller', 'Fantasy'],
    weights: { 'Science Fiction': 1.5, Mystery: 1.4, Thriller: 1.2 },
    minRating: 7.0,
    sortBoost: 'voteAverage',
    description: 'Complex narratives that make you think'
  },
  'romantic': {
    genres: ['Romance', 'Drama', 'Comedy'],
    weights: { Romance: 1.8, Drama: 1.0, Comedy: 1.2 },
    minRating: 6.5,
    sortBoost: 'voteAverage',
    description: 'Love stories that touch your heart'
  },
  'dark-intense': {
    genres: ['Crime', 'Thriller', 'Drama', 'Horror', 'War'],
    weights: { Crime: 1.5, Thriller: 1.3, Drama: 1.0, War: 1.2 },
    minRating: 7.0,
    sortBoost: 'voteAverage',
    description: 'Gritty, serious, and powerful storytelling'
  },
  'epic-adventure': {
    genres: ['Adventure', 'Action', 'Fantasy', 'Science Fiction'],
    weights: { Adventure: 1.5, Action: 1.3, Fantasy: 1.4, 'Science Fiction': 1.2 },
    minRating: 7.0,
    sortBoost: 'revenue',
    description: 'Grand-scale journeys and spectacular worlds'
  },
  'laugh-out-loud': {
    genres: ['Comedy', 'Animation'],
    weights: { Comedy: 1.8, Animation: 1.3 },
    minRating: 6.0,
    sortBoost: 'popularity',
    description: 'Pure laughs and fun entertainment'
  },
  'emotional': {
    genres: ['Drama', 'War', 'History', 'Biography'],
    weights: { Drama: 1.5, War: 1.3, History: 1.2 },
    minRating: 7.5,
    sortBoost: 'voteAverage',
    description: 'Deep stories with emotional impact'
  },
  'scary': {
    genres: ['Horror', 'Thriller', 'Mystery'],
    weights: { Horror: 1.8, Thriller: 1.2, Mystery: 1.0 },
    minRating: 6.0,
    sortBoost: 'popularity',
    description: 'Spine-chilling frights and tension'
  },
  'inspirational': {
    genres: ['Drama', 'Biography', 'History', 'Sport', 'Music'],
    weights: { Biography: 1.5, Drama: 1.3, History: 1.2, Sport: 1.4 },
    minRating: 7.0,
    sortBoost: 'voteAverage',
    description: 'True stories and motivating journeys'
  }
};

// Cosine similarity helper for genre vectors
function computeGenreSimilarity(genres1, genres2) {
  const allGenres = [...new Set([...genres1, ...genres2])];
  const vec1 = allGenres.map(g => genres1.includes(g) ? 1 : 0);
  const vec2 = allGenres.map(g => genres2.includes(g) ? 1 : 0);
  
  const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
  
  return mag1 && mag2 ? dotProduct / (mag1 * mag2) : 0;
}

// POST /api/movies/recommend
router.post('/recommend', async (req, res) => {
  try {
    const { mode, mood, genres, movieId, talentName, talentType, industry, era, limit = 20 } = req.body;

    let movies = [];
    let meta = {};

    switch (mode) {
      // ─── MOOD-BASED ───
      case 'mood': {
        const moodConfig = MOOD_MAP[mood];
        if (!moodConfig) {
          return res.status(400).json({ error: `Unknown mood: ${mood}. Available: ${Object.keys(MOOD_MAP).join(', ')}` });
        }

        const query = {
          genres: { $in: moodConfig.genres.map(g => new RegExp(g, 'i')) },
          voteAverage: { $gte: moodConfig.minRating }
        };
        if (industry && industry !== 'all') query.industry = industry.toLowerCase();

        const rawMovies = await Movie.find(query)
          .sort({ [moodConfig.sortBoost]: -1 })
          .limit(parseInt(limit) * 3)
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        // Score and rank by genre weight match
        movies = rawMovies.map(movie => {
          const movieObj = movie.toObject();
          let score = 0;
          const movieGenres = (movieObj.genres || []).map(g => g.toLowerCase());
          
          for (const [genre, weight] of Object.entries(moodConfig.weights)) {
            if (movieGenres.some(mg => mg.includes(genre.toLowerCase()))) {
              score += weight;
            }
          }
          // Boost by rating
          score += (movieObj.voteAverage || 0) / 10;
          
          movieObj.matchScore = Math.min(Math.round((score / (Object.keys(moodConfig.weights).length + 1)) * 100), 99);
          return movieObj;
        })
        .sort((a, b) => b.matchScore - a.matchScore)
        .slice(0, parseInt(limit));

        meta = { mood, description: moodConfig.description, moodGenres: moodConfig.genres };
        break;
      }

      // ─── GENRE MIX ───
      case 'genre': {
        if (!genres || genres.length === 0) {
          return res.status(400).json({ error: 'At least one genre is required' });
        }

        const query = {
          genres: { $in: genres.map(g => new RegExp(g, 'i')) },
          voteAverage: { $gte: 5.5 }
        };
        if (industry && industry !== 'all') query.industry = industry.toLowerCase();

        const rawMovies = await Movie.find(query)
          .sort({ voteAverage: -1, popularity: -1 })
          .limit(parseInt(limit) * 3)
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        // Score by how many requested genres match
        movies = rawMovies.map(movie => {
          const movieObj = movie.toObject();
          const movieGenres = (movieObj.genres || []).map(g => g.toLowerCase());
          const matchCount = genres.filter(g => movieGenres.some(mg => mg.includes(g.toLowerCase()))).length;
          const genreOverlap = matchCount / genres.length;
          const ratingBoost = (movieObj.voteAverage || 0) / 10;
          
          movieObj.matchScore = Math.min(Math.round((genreOverlap * 0.7 + ratingBoost * 0.3) * 100), 99);
          return movieObj;
        })
        .sort((a, b) => b.matchScore - a.matchScore)
        .slice(0, parseInt(limit));

        meta = { selectedGenres: genres };
        break;
      }

      // ─── SIMILAR TO MOVIE ───
      case 'similar': {
        if (!movieId) {
          return res.status(400).json({ error: 'movieId is required for similar mode' });
        }

        const sourceMovie = await Movie.findById(movieId);
        if (!sourceMovie) {
          return res.status(404).json({ error: 'Source movie not found' });
        }

        const sourceGenres = sourceMovie.genres || [];
        const sourceIndustry = sourceMovie.industry;
        const sourceBudget = sourceMovie.budget || 0;
        const sourceDirector = sourceMovie.director?.name;

        // Find candidates with overlapping genres
        const candidates = await Movie.find({
          _id: { $ne: sourceMovie._id },
          genres: { $in: sourceGenres },
          voteAverage: { $gte: 5.0 }
        })
          .limit(200)
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        movies = candidates.map(movie => {
          const movieObj = movie.toObject();
          let score = 0;

          // Genre similarity (40% weight)
          const genreSim = computeGenreSimilarity(sourceGenres, movieObj.genres || []);
          score += genreSim * 40;

          // Same industry (15% weight)
          if (movieObj.industry === sourceIndustry) score += 15;

          // Budget similarity (15% weight)
          if (sourceBudget > 0 && movieObj.budget > 0) {
            const budgetRatio = Math.min(sourceBudget, movieObj.budget) / Math.max(sourceBudget, movieObj.budget);
            score += budgetRatio * 15;
          }

          // Same director (10% weight)
          if (sourceDirector && movieObj.director?.name === sourceDirector) score += 10;

          // Rating similarity (10% weight)
          const ratingDiff = Math.abs((sourceMovie.voteAverage || 0) - (movieObj.voteAverage || 0));
          score += Math.max(0, 10 - ratingDiff * 2);

          // Popularity boost (10% weight)
          score += Math.min((movieObj.popularity || 0) / 100, 10);

          movieObj.matchScore = Math.min(Math.round(score), 99);
          return movieObj;
        })
        .sort((a, b) => b.matchScore - a.matchScore)
        .slice(0, parseInt(limit));

        meta = {
          sourceMovie: {
            _id: sourceMovie._id,
            title: sourceMovie.title,
            year: sourceMovie.year,
            posterPath: sourceMovie.posterPath,
            genres: sourceMovie.genres
          }
        };
        break;
      }

      // ─── BY TALENT ───
      case 'talent': {
        if (!talentName) {
          return res.status(400).json({ error: 'talentName is required' });
        }

        const nameRegex = new RegExp(talentName, 'i');
        let query;

        if (talentType === 'director') {
          query = { 'director.name': nameRegex };
        } else {
          query = { 'cast.name': nameRegex };
        }
        if (industry && industry !== 'all') query.industry = industry.toLowerCase();

        const rawMovies = await Movie.find(query)
          .sort({ voteAverage: -1 })
          .limit(parseInt(limit))
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        movies = rawMovies.map(movie => {
          const movieObj = movie.toObject();
          movieObj.matchScore = Math.min(Math.round(((movieObj.voteAverage || 0) / 10) * 100), 99);
          return movieObj;
        });

        meta = { talentName, talentType: talentType || 'actor' };
        break;
      }

      // ─── TOP RATED ───
      case 'top-rated': {
        const query = { voteAverage: { $gte: 7.0 }, voteCount: { $gte: 100 } };
        if (industry && industry !== 'all') query.industry = industry.toLowerCase();
        if (era) {
          const [startYear, endYear] = era.split('-').map(Number);
          if (startYear && endYear) {
            query.year = { $gte: startYear, $lte: endYear };
          }
        }

        const rawMovies = await Movie.find(query)
          .sort({ voteAverage: -1, voteCount: -1 })
          .limit(parseInt(limit))
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        movies = rawMovies.map(movie => {
          const movieObj = movie.toObject();
          movieObj.matchScore = Math.min(Math.round(((movieObj.voteAverage || 0) / 10) * 100), 99);
          return movieObj;
        });

        meta = { industry: industry || 'all', era: era || 'all' };
        break;
      }

      // ─── SURPRISE ME ───
      case 'surprise': {
        const query = { voteAverage: { $gte: 6.5 } };
        if (industry && industry !== 'all') query.industry = industry.toLowerCase();

        const count = await Movie.countDocuments(query);
        const randomSkip = Math.max(0, Math.floor(Math.random() * (count - parseInt(limit))));

        const rawMovies = await Movie.find(query)
          .skip(randomSkip)
          .limit(parseInt(limit))
          .select('title year posterPath backdropPath industry genres voteAverage revenue popularity budget predictions director cast overview');

        movies = rawMovies.map(movie => {
          const movieObj = movie.toObject();
          movieObj.matchScore = Math.floor(Math.random() * 20) + 75; // Random 75-94 for fun
          return movieObj;
        });

        meta = { surprise: true, message: 'Here are some gems you might enjoy!' };
        break;
      }

      default:
        return res.status(400).json({ error: `Unknown mode: ${mode}. Available: mood, genre, similar, talent, top-rated, surprise` });
    }

    res.json({
      recommendations: movies,
      meta,
      total: movies.length
    });

  } catch (error) {
    console.error('Recommendation error:', error);
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
