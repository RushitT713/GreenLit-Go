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
    
    const matchStage = { releaseDate: { $exists: true, $ne: null, $type: 'date' } };
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

// Get budget vs revenue scatter data
router.get('/budget-revenue', async (req, res) => {
  try {
    const { industry } = req.query;
    
    const matchStage = { budget: { $gt: 0 }, revenue: { $gt: 0 } };
    if (industry) matchStage.industry = industry;

    const data = await Movie.find(matchStage)
      .select('title budget revenue industry predictions.successCategory voteAverage year')
      .limit(500)
      .lean();

    const formatted = data.map(m => ({
      title: m.title,
      budget: m.budget,
      revenue: m.revenue,
      roi: ((m.revenue - m.budget) / m.budget * 100).toFixed(1),
      category: m.predictions?.successCategory || 'Unknown',
      rating: m.voteAverage,
      industry: m.industry,
      year: m.year
    }));

    res.json(formatted);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// YouTube Hype vs Box Office — scatter (trailerViews vs revenue)
router.get('/youtube-hype', async (req, res) => {
  try {
    const { industry } = req.query;
    const matchStage = {
      'socialMetrics.trailerViews': { $gt: 0 },
      revenue: { $gt: 0 },
    };
    if (industry) matchStage.industry = industry;

    const data = await Movie.find(matchStage)
      .select('title socialMetrics.trailerViews revenue voteAverage predictions.successCategory industry year')
      .limit(300)
      .lean();

    const formatted = data.map(m => ({
      title: m.title,
      trailerViews: m.socialMetrics?.trailerViews || 0,
      revenue: m.revenue,
      rating: m.voteAverage || 0,
      category: m.predictions?.successCategory || 'Unknown',
      industry: m.industry,
      year: m.year,
    }));

    res.json(formatted);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Production House Rankings — aggregated by company
router.get('/production-houses', async (req, res) => {
  try {
    const { industry, limit = 15 } = req.query;
    const matchStage = {
      'productionCompanies.0': { $exists: true },
      revenue: { $gt: 0 },
    };
    if (industry) matchStage.industry = industry;

    const data = await Movie.aggregate([
      { $match: matchStage },
      { $unwind: '$productionCompanies' },
      {
        $group: {
          _id: '$productionCompanies.name',
          movieCount: { $sum: 1 },
          avgRevenue: { $avg: '$revenue' },
          totalRevenue: { $sum: '$revenue' },
          avgRating: { $avg: '$voteAverage' },
          avgBudget: { $avg: '$budget' },
        },
      },
      { $match: { _id: { $ne: null }, movieCount: { $gte: 3 } } },
      { $sort: { totalRevenue: -1 } },
      { $limit: parseInt(limit) },
    ]);

    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Opening Weekend Strength — opening weekend as % of total revenue
router.get('/opening-weekend', async (req, res) => {
  try {
    const { industry } = req.query;
    const matchStage = {
      'releaseStrategy.openingWeekendRevenue': { $gt: 0 },
      revenue: { $gt: 0 },
    };
    if (industry) matchStage.industry = industry;

    const data = await Movie.find(matchStage)
      .select('title releaseStrategy.openingWeekendRevenue revenue year industry predictions.successCategory')
      .sort({ revenue: -1 })
      .limit(30)
      .lean();

    const formatted = data.map(m => ({
      title: m.title,
      openingWeekend: m.releaseStrategy?.openingWeekendRevenue || 0,
      totalRevenue: m.revenue,
      openingPct: parseFloat(((m.releaseStrategy?.openingWeekendRevenue || 0) / m.revenue * 100).toFixed(1)),
      year: m.year,
      industry: m.industry,
      category: m.predictions?.successCategory || 'Unknown',
    }));

    res.json(formatted);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Critic vs Audience Gap — RT critics - audience divergence
router.get('/critic-audience', async (req, res) => {
  try {
    const { industry } = req.query;
    const matchStage = {
      rottenTomatoesScore: { $gt: 0 },
      rottenTomatoesAudienceScore: { $gt: 0 },
    };
    if (industry) matchStage.industry = industry;

    const data = await Movie.find(matchStage)
      .select('title rottenTomatoesScore rottenTomatoesAudienceScore voteAverage year industry')
      .lean();

    const formatted = data
      .map(m => ({
        title: m.title,
        criticsScore: m.rottenTomatoesScore,
        audienceScore: m.rottenTomatoesAudienceScore,
        gap: m.rottenTomatoesScore - m.rottenTomatoesAudienceScore,
        rating: m.voteAverage || 0,
        year: m.year,
        industry: m.industry,
      }))
      .sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap))
      .slice(0, 30);

    res.json(formatted);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
