const express = require('express');
const router = express.Router();
const Movie = require('../models/Movie');

// Get industry trends over time
router.get('/yearly', async (req, res) => {
  try {
    const { industry, startYear, endYear } = req.query;
    
    const matchStage = {};
    if (industry) matchStage.industry = industry;
    if (startYear || endYear) {
      matchStage.year = {};
      if (startYear) matchStage.year.$gte = parseInt(startYear);
      if (endYear) matchStage.year.$lte = parseInt(endYear);
    }

    const trends = await Movie.aggregate([
      { $match: matchStage },
      {
        $group: {
          _id: '$year',
          totalMovies: { $sum: 1 },
          avgBudget: { $avg: '$budget' },
          avgRevenue: { $avg: '$revenue' },
          avgRating: { $avg: '$voteAverage' },
          totalRevenue: { $sum: '$revenue' },
          blockbusters: {
            $sum: { $cond: [{ $eq: ['$predictions.successCategory', 'Blockbuster'] }, 1, 0] }
          },
          hits: {
            $sum: { $cond: [{ $in: ['$predictions.successCategory', ['Super Hit', 'Hit']] }, 1, 0] }
          },
          flops: {
            $sum: { $cond: [{ $in: ['$predictions.successCategory', ['Flop', 'Disaster']] }, 1, 0] }
          }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    res.json(trends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get genre trends
router.get('/genres', async (req, res) => {
  try {
    const { industry, year } = req.query;
    
    const matchStage = {};
    if (industry) matchStage.industry = industry;
    if (year) matchStage.year = parseInt(year);

    const genreTrends = await Movie.aggregate([
      { $match: matchStage },
      { $unwind: '$genres' },
      {
        $group: {
          _id: '$genres',
          count: { $sum: 1 },
          avgRevenue: { $avg: '$revenue' },
          avgBudget: { $avg: '$budget' },
          avgRating: { $avg: '$voteAverage' },
          avgROI: { 
            $avg: { 
              $cond: [
                { $and: [{ $gt: ['$budget', 0] }, { $gt: ['$revenue', 0] }] },
                { $multiply: [{ $divide: [{ $subtract: ['$revenue', '$budget'] }, '$budget'] }, 100] },
                null
              ]
            }
          },
          topMovie: { $max: { revenue: '$revenue', title: '$title' } }
        }
      },
      { $sort: { count: -1 } }
    ]);

    res.json(genreTrends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get seasonal trends (monthly performance)
router.get('/seasonal', async (req, res) => {
  try {
    const { industry, genre } = req.query;
    
    const matchStage = { releaseDate: { $exists: true } };
    if (industry) matchStage.industry = industry;
    if (genre) matchStage.genres = genre;

    const seasonalTrends = await Movie.aggregate([
      { $match: matchStage },
      {
        $group: {
          _id: { $month: '$releaseDate' },
          count: { $sum: 1 },
          avgRevenue: { $avg: '$revenue' },
          avgRating: { $avg: '$voteAverage' },
          successRate: {
            $avg: {
              $cond: [
                { $in: ['$predictions.successCategory', ['Blockbuster', 'Super Hit', 'Hit']] },
                1,
                0
              ]
            }
          }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    const formattedTrends = seasonalTrends.map(t => ({
      month: monthNames[t._id - 1],
      monthNumber: t._id,
      ...t
    }));

    res.json(formattedTrends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get director/actor performance trends
router.get('/talent/:type', async (req, res) => {
  try {
    const { type } = req.params;
    const { industry, limit = 20 } = req.query;
    
    const matchStage = {};
    if (industry) matchStage.industry = industry;

    let groupField, nameField;
    if (type === 'directors') {
      groupField = '$director.name';
      nameField = 'director';
    } else {
      // For actors, we'll use the first cast member (lead actor)
      groupField = { $arrayElemAt: ['$cast.name', 0] };
      nameField = 'actor';
    }

    const talentTrends = await Movie.aggregate([
      { $match: { ...matchStage, [nameField === 'director' ? 'director.name' : 'cast']: { $exists: true, $ne: null } } },
      {
        $group: {
          _id: groupField,
          movieCount: { $sum: 1 },
          avgRevenue: { $avg: '$revenue' },
          totalRevenue: { $sum: '$revenue' },
          avgRating: { $avg: '$voteAverage' },
          hitRate: {
            $avg: {
              $cond: [
                { $in: ['$predictions.successCategory', ['Blockbuster', 'Super Hit', 'Hit']] },
                1,
                0
              ]
            }
          },
          movies: { $push: { title: '$title', year: '$year', revenue: '$revenue' } }
        }
      },
      { $match: { movieCount: { $gte: 3 } } },
      { $sort: { avgRevenue: -1 } },
      { $limit: parseInt(limit) }
    ]);

    res.json(talentTrends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get regional comparison (for Indian cinema)
router.get('/regional', async (req, res) => {
  try {
    const regionalTrends = await Movie.aggregate([
      { $match: { industry: { $in: ['bollywood', 'tollywood', 'kollywood', 'mollywood', 'sandalwood'] } } },
      {
        $group: {
          _id: '$industry',
          totalMovies: { $sum: 1 },
          avgBudget: { $avg: '$budget' },
          avgRevenue: { $avg: '$revenue' },
          avgRating: { $avg: '$voteAverage' },
          topGenres: { $push: '$genres' }
        }
      },
      { $sort: { totalMovies: -1 } }
    ]);

    res.json(regionalTrends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
