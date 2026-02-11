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

module.exports = router;
