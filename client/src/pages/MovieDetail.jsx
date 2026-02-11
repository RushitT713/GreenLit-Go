import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { movieService } from '../services/api';
import './MovieDetail.css';

const MovieDetail = () => {
    const { id } = useParams();
    const [movie, setMovie] = useState(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');

    useEffect(() => {
        fetchMovie();
    }, [id]);

    const fetchMovie = async () => {
        try {
            setLoading(true);
            const response = await movieService.getById(id);
            setMovie(response.data);
        } catch (error) {
            setMovie(getDemoMovie());
        } finally {
            setLoading(false);
        }
    };

    const getDemoMovie = () => ({
        _id: id,
        title: 'John Wick: Chapter 4',
        year: 2023,
        overview: 'With the price on his head ever increasing, John Wick uncovers a path to defeating The High Table. But before he can earn his freedom, Wick must face off against a new enemy with powerful alliances across the globe.',
        posterPath: 'https://image.tmdb.org/t/p/w500/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg',
        backdropPath: 'https://image.tmdb.org/t/p/original/7I6VUdPj6tQECNHdviJkUHD2u89.jpg',
        genres: ['Action', 'Thriller', 'Crime'],
        runtime: 169,
        budget: 100000000,
        revenue: 440000000,
        voteAverage: 7.9,
        voteCount: 29000,
        industry: 'hollywood',
        releaseDate: '2023-03-22',
        director: { name: 'Chad Stahelski' },
        cast: [
            { name: 'Keanu Reeves', character: 'John Wick', profilePath: 'https://image.tmdb.org/t/p/w185/4D0PpNI0kmP58hgrwGC3wCjxhnm.jpg' },
            { name: 'Donnie Yen', character: 'Caine', profilePath: 'https://image.tmdb.org/t/p/w185/hTlhrrZMj8hZVvD17j4KzA8YS5E.jpg' },
            { name: 'Bill Skarsgård', character: 'Marquis', profilePath: 'https://image.tmdb.org/t/p/w185/oeeGlUIGAhz2vKQojCg1JxLesNE.jpg' },
        ],
        predictions: {
            successCategory: 'Blockbuster',
            predictedRevenue: 420000000,
            confidence: 92,
            featureImportance: [
                { feature: 'Franchise Power', impact: 0.24 },
                { feature: 'Lead Actor', impact: 0.20 },
                { feature: 'Director', impact: 0.16 },
                { feature: 'Genre', impact: 0.14 },
                { feature: 'Budget', impact: 0.12 },
            ]
        }
    });

    const formatCurrency = (value) => {
        if (!value) return 'N/A';
        if (value >= 1000000000) return `$${(value / 1000000000).toFixed(1)}B`;
        if (value >= 1000000) return `$${(value / 1000000).toFixed(0)}M`;
        return `$${value.toLocaleString()}`;
    };

    if (loading) {
        return (
            <div className="movie-detail-loading">
                <div className="spinner"></div>
                <p>Loading movie details...</p>
            </div>
        );
    }

    if (!movie) {
        return (
            <div className="movie-detail-error">
                <h2>Movie not found</h2>
                <Link to="/movies" className="btn btn-primary">Back to Movies</Link>
            </div>
        );
    }

    const tabs = ['overview', 'cast'];

    return (
        <div className="movie-detail">
            {/* Hero Section */}
            <section className="detail-hero">
                <div className="hero-backdrop">
                    {movie.backdropPath && (
                        <img
                            src={movie.backdropPath.startsWith('http') ? movie.backdropPath : `https://image.tmdb.org/t/p/original${movie.backdropPath}`}
                            alt=""
                        />
                    )}
                    <div className="hero-overlay"></div>
                </div>

                <div className="container">
                    <div className="detail-hero-content">
                        {/* Poster */}
                        <motion.div
                            className="detail-poster"
                            initial={{ opacity: 0, x: -30 }}
                            animate={{ opacity: 1, x: 0 }}
                        >
                            <img
                                src={movie.posterPath?.startsWith('http') ? movie.posterPath : `https://image.tmdb.org/t/p/w500${movie.posterPath}`}
                                alt={movie.title}
                            />
                        </motion.div>

                        {/* Info */}
                        <motion.div
                            className="detail-info"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="detail-badges">
                                {movie.predictions?.successCategory && (
                                    <span className={`badge category-${movie.predictions.successCategory.toLowerCase()}`}>
                                        {movie.predictions.successCategory}
                                    </span>
                                )}
                                <span className="badge industry-badge">{movie.industry}</span>
                            </div>

                            <h1>{movie.title}</h1>

                            <div className="detail-meta">
                                <span className="rating">
                                    <span className="star">⭐</span>
                                    <span className="rating-value">{movie.voteAverage?.toFixed(1)}</span>
                                    <span className="rating-count">({movie.voteCount?.toLocaleString()} votes)</span>
                                </span>
                                <span>{movie.year}</span>
                                <span>{movie.runtime} min</span>
                                <span>{movie.genres?.join(', ')}</span>
                            </div>

                            <p className="detail-overview">{movie.overview}</p>

                            <div className="detail-stats">
                                <div className="stat">
                                    <span className="stat-label">Budget</span>
                                    <span className="stat-value">{formatCurrency(movie.budget)}</span>
                                </div>
                                <div className="stat">
                                    <span className="stat-label">Revenue</span>
                                    <span className="stat-value">{formatCurrency(movie.revenue)}</span>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* Tabs */}
            <section className="detail-tabs">
                <div className="container">
                    <div className="tabs-header">
                        {tabs.map(tab => (
                            <button
                                key={tab}
                                className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                                onClick={() => setActiveTab(tab)}
                            >
                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                            </button>
                        ))}
                    </div>

                    <div className="tab-content">
                        {activeTab === 'overview' && (
                            <div className="tab-panel">
                                <div className="overview-grid">
                                    {/* Performance Summary */}
                                    {movie.predictions?.successCategory && (
                                        <div className="overview-section performance-section">
                                            <h3>🎯 Performance Summary</h3>
                                            <div className="performance-summary">
                                                <div className="success-badge-large">
                                                    <span className={`success-category category-${movie.predictions.successCategory.toLowerCase()}`}>
                                                        {movie.predictions.successCategory}
                                                    </span>
                                                </div>
                                                <div className="financial-summary">
                                                    <div className="financial-item">
                                                        <span className="fin-label">Budget</span>
                                                        <span className="fin-value">{formatCurrency(movie.budget)}</span>
                                                    </div>
                                                    <div className="financial-item">
                                                        <span className="fin-label">Box Office</span>
                                                        <span className="fin-value">{formatCurrency(movie.revenue)}</span>
                                                    </div>
                                                    {movie.budget && movie.revenue && (
                                                        <div className="financial-item">
                                                            <span className="fin-label">ROI</span>
                                                            <span className={`fin-value ${((movie.revenue - movie.budget) / movie.budget) >= 1 ? 'text-success' : 'text-warning'}`}>
                                                                {(((movie.revenue - movie.budget) / movie.budget) * 100).toFixed(0)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Details */}
                                    <div className="overview-section">
                                        <h3>📋 Details</h3>
                                        <div className="detail-list">
                                            {movie.director && <p><strong>Director:</strong> {movie.director.name}</p>}
                                            <p><strong>Release Date:</strong> {movie.releaseDate}</p>
                                            <p><strong>Industry:</strong> {movie.industry?.charAt(0).toUpperCase() + movie.industry?.slice(1)}</p>
                                            <p><strong>Runtime:</strong> {movie.runtime} minutes</p>
                                            <p><strong>Genres:</strong> {movie.genres?.join(', ')}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}



                        {activeTab === 'cast' && (
                            <div className="tab-panel">
                                <div className="cast-grid">
                                    {movie.cast?.map((actor, i) => (
                                        <div key={i} className="cast-card">
                                            <div className="cast-photo">
                                                {actor.profilePath ? (
                                                    <img
                                                        src={actor.profilePath.startsWith('http') ? actor.profilePath : `https://image.tmdb.org/t/p/w185${actor.profilePath}`}
                                                        alt={actor.name}
                                                    />
                                                ) : (
                                                    <span>👤</span>
                                                )}
                                            </div>
                                            <div className="cast-info">
                                                <span className="cast-name">{actor.name}</span>
                                                <span className="cast-character">{actor.character}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </section>
        </div>
    );
};

export default MovieDetail;
