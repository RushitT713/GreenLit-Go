const express = require('express');
const cors = require('cors');
require('dotenv').config();

const connectDB = require('./config/database');
const movieRoutes = require('./routes/movies');
const predictionRoutes = require('./routes/predictions');
const trendsRoutes = require('./routes/trends');
const talentsRoutes = require('./routes/talents');

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/movies', movieRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/trends', trendsRoutes);
app.use('/api/talents', talentsRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Something went wrong!',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`🎬 GreenLit Go Server running on port ${PORT}`);
});

module.exports = app;
