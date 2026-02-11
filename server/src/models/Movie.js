const mongoose = require('mongoose');

const movieSchema = new mongoose.Schema({
  // External IDs
  tmdbId: { type: Number, unique: true, sparse: true },
  imdbId: { type: String, unique: true, sparse: true },
  
  // Basic Info
  title: { type: String, required: true, index: true },
  originalTitle: String,
  overview: String,
  tagline: String,
  
  // Industry Classification
  industry: { 
    type: String, 
    enum: ['hollywood', 'bollywood', 'tollywood', 'kollywood', 'mollywood', 'sandalwood', 'other'],
    default: 'hollywood',
    index: true
  },
  
  // Release Info
  releaseDate: { type: Date, index: true },
  year: { type: Number, index: true },
  status: { type: String, enum: ['Released', 'Upcoming', 'In Production', 'Rumored'] },
  
  // Production Details
  budget: Number,
  revenue: Number,
  runtime: Number,
  genres: [{ type: String, index: true }],
  language: String,
  originalLanguage: String,
  spokenLanguages: [String],
  countries: [String],
  
  // Certification
  mpaaRating: String,
  certification: String,
  adult: { type: Boolean, default: false },
  
  // Production
  productionCompanies: [{
    id: Number,
    name: String,
    logoPath: String,
    originCountry: String
  }],
  
  // Franchise Info
  isSequel: { type: Boolean, default: false },
  franchiseName: String,
  collectionId: Number,
  partOfCollection: {
    id: Number,
    name: String,
    posterPath: String,
    backdropPath: String
  },
  
  // Cast & Crew
  cast: [{
    id: Number,
    name: String,
    character: String,
    order: Number,
    popularity: Number,
    profilePath: String,
    gender: Number
  }],
  
  director: {
    id: Number,
    name: String,
    popularity: Number,
    profilePath: String,
    avgRevenue: Number,
    hitRate: Number,
    totalMovies: Number
  },
  
  writers: [{
    id: Number,
    name: String,
    job: String
  }],
  
  musicDirector: {
    id: Number,
    name: String
  },
  
  crew: [{
    id: Number,
    name: String,
    job: String,
    department: String
  }],
  
  // Ratings & Scores
  voteAverage: Number,
  voteCount: Number,
  popularity: Number,
  
  imdbRating: Number,
  imdbVotes: Number,
  
  rottenTomatoesScore: Number,
  rottenTomatoesAudienceScore: Number,
  
  metacriticScore: Number,
  
  // Predictions (computed by ML models)
  predictions: {
    successCategory: {
      type: String,
      enum: ['Blockbuster', 'Super Hit', 'Hit', 'Average', 'Below Average', 'Flop', 'Disaster']
    },
    predictedRevenue: Number,
    predictedRating: Number,
    predictedROI: Number,
    confidence: Number,
    lastUpdated: Date,
    
    // Feature explanations from SHAP
    featureImportance: [{
      feature: String,
      value: Number,
      impact: Number
    }]
  },
  
  // Social Metrics
  socialMetrics: {
    trailerViews: Number,
    trailerLikes: Number,
    trailerComments: Number,
    googleTrendsScore: Number,
    googleTrendsPeak: Number,
    redditMentions: Number,
    redditSentiment: Number,
    wikiPageViews: Number,
    twitterMentions: Number,
    instagramMentions: Number,
    lastUpdated: Date
  },
  
  // Release Strategy
  releaseStrategy: {
    openingScreens: Number,
    openingWeekendRevenue: Number,
    domesticRevenue: Number,
    internationalRevenue: Number,
    holdoverPercentage: Number,
    theaterRun: Number
  },
  
  // Awards
  awards: [{
    name: String,
    category: String,
    result: { type: String, enum: ['Won', 'Nominated'] },
    year: Number
  }],
  
  // Media
  posterPath: String,
  backdropPath: String,
  trailerUrl: String,
  trailerKey: String,
  images: {
    posters: [String],
    backdrops: [String]
  },
  
  // Keywords & Tags
  keywords: [String],
  
  // Analysis Flags
  isAnalyzed: { type: Boolean, default: false },
  dataCompleteness: { type: Number, min: 0, max: 100 },
  
  // Metadata
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for ROI calculation
movieSchema.virtual('roi').get(function() {
  if (this.budget && this.revenue && this.budget > 0) {
    return ((this.revenue - this.budget) / this.budget * 100).toFixed(2);
  }
  return null;
});

// Virtual for poster URL
movieSchema.virtual('posterUrl').get(function() {
  if (this.posterPath) {
    return `https://image.tmdb.org/t/p/w500${this.posterPath}`;
  }
  return null;
});

// Virtual for backdrop URL
movieSchema.virtual('backdropUrl').get(function() {
  if (this.backdropPath) {
    return `https://image.tmdb.org/t/p/original${this.backdropPath}`;
  }
  return null;
});

// Index for text search
movieSchema.index({ title: 'text', overview: 'text', keywords: 'text' });

// Compound indexes for common queries
movieSchema.index({ industry: 1, year: -1 });
movieSchema.index({ genres: 1, year: -1 });
movieSchema.index({ 'predictions.successCategory': 1 });

const Movie = mongoose.model('Movie', movieSchema);

module.exports = Movie;
