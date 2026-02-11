import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Movies API
export const movieService = {
  getAll: (params = {}) => api.get('/movies', { params }),
  getById: (id) => api.get(`/movies/${id}`),
  getByTmdbId: (tmdbId) => api.get(`/movies/tmdb/${tmdbId}`),
  search: (query, limit = 10) => api.get('/movies/search', { params: { q: query, limit } }),
  getTrending: (industry, limit = 10) => api.get('/movies/trending', { params: { industry, limit } }),
  getByCategory: (category, params = {}) => api.get(`/movies/by-category/${category}`, { params }),
  getByGenre: (genre, params = {}) => api.get(`/movies/by-genre/${genre}`, { params }),
  getUpcoming: (params = {}) => api.get('/movies/upcoming', { params }),
  getSimilar: (id) => api.get(`/movies/${id}/similar`),
  getStats: (industry) => api.get('/movies/stats/overview', { params: { industry } }),
};

// Predictions API
export const predictionService = {
  predict: (movieData) => api.post('/predictions/predict', movieData),
  explain: (movieData) => api.post('/predictions/explain', movieData),
  simulate: (baseMovie, modifications) => api.post('/predictions/simulate', { baseMovie, modifications }),
  getOptimalRelease: (movieData) => api.post('/predictions/optimal-release', movieData),
  getCompetitiveAnalysis: (data) => api.post('/predictions/competitive-analysis', data),
};

// Trends API
export const trendsService = {
  getYearly: (params = {}) => api.get('/trends/yearly', { params }),
  getGenres: (params = {}) => api.get('/trends/genres', { params }),
  getSeasonal: (params = {}) => api.get('/trends/seasonal', { params }),
  getTalent: (type, params = {}) => api.get(`/trends/talent/${type}`, { params }),
  getRegional: () => api.get('/trends/regional'),
};

// Talents API (Director/Actor Lookup)
export const talentsService = {
  search: (query, limit = 10) => api.get('/talents/search', { params: { q: query, limit } }),
  getDirector: (id) => api.get(`/talents/director/${id}`),
  getActor: (id) => api.get(`/talents/actor/${id}`),
  getCastScore: (actorIds) => api.post('/talents/cast-score', { actorIds }),
};

export default api;
