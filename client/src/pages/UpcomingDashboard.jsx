import { useState } from 'react';
import { motion } from 'framer-motion';
import { predictionService } from '../services/api';
import { DirectorSearch, CastSearch } from '../components/common/TalentSearch';
import './UpcomingDashboard.css';

const UpcomingDashboard = () => {
    const [formData, setFormData] = useState({
        title: '',
        genres: [],
        budget: '',
        runtime: '',
        releaseMonth: '',
        industry: 'hollywood',
        directorName: '',
        directorHitRate: '',
        leadActorPopularity: '',
        isSequel: false,
        productionCompany: '',
        trailerViews: '',
        googleTrendsScore: ''
    });

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [activeSection, setActiveSection] = useState('predict');
    const [selectedDirector, setSelectedDirector] = useState(null);
    const [selectedCast, setSelectedCast] = useState([]);
    const [castScore, setCastScore] = useState(0);

    const genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation', 'Adventure', 'Crime', 'Fantasy', 'Mystery'];
    const industries = [
        { value: 'hollywood', label: '🇺🇸 Hollywood' },
        { value: 'bollywood', label: '🇮🇳 Bollywood' },
        { value: 'tollywood', label: '🎬 Tollywood' },
        { value: 'kollywood', label: '🎥 Kollywood' },
        { value: 'mollywood', label: '🌴 Mollywood' }
    ];
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleGenreToggle = (genre) => {
        setFormData(prev => ({
            ...prev,
            genres: prev.genres.includes(genre)
                ? prev.genres.filter(g => g !== genre)
                : [...prev.genres, genre]
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        try {
            // Build enriched prediction payload with talent data
            const predictionPayload = {
                ...formData,
                // Director TMDB popularity (from talent search)
                directorPopularity: selectedDirector?.popularity || 0,
                // Cast members with individual popularity scores
                castMembers: selectedCast.map(member => ({
                    name: member.name,
                    popularity: member.popularity || 0
                })),
                // Combined cast score from TalentSearch component
                castScore: castScore || 0,
                // Cast popularity as average of selected members
                castPopularity: selectedCast.length > 0
                    ? selectedCast.reduce((sum, m) => sum + (m.popularity || 0), 0) / selectedCast.length
                    : 0,
                // Current year
                year: new Date().getFullYear()
            };

            const response = await predictionService.predict(predictionPayload);
            setPrediction(response.data);
        } catch (error) {
            console.error('Prediction error:', error);
            // Use demo prediction
            setPrediction(getDemoPrediction());
        } finally {
            setLoading(false);
        }
    };

    const getDemoPrediction = () => {
        const budget = parseInt(formData.budget || 100000000);
        const predictedRevenue = budget * 3.5;
        const calculatedROI = budget > 0 ? ((predictedRevenue - budget) / budget * 100) : 0;

        return {
            predictions: {
                successCategory: formData.budget > 100000000 ? 'Blockbuster' : 'Hit',
                predictedRevenue: predictedRevenue,
                predictedROI: calculatedROI,
                confidence: 85,
                featureImportance: [
                    { feature: 'Director Track Record', value: formData.directorHitRate || 0.7, impact: 0.22 },
                    { feature: 'Lead Actor Popularity', value: formData.leadActorPopularity || 50, impact: 0.18 },
                    { feature: 'Budget', value: formData.budget, impact: 0.15 },
                    { feature: 'Genre Performance', value: formData.genres.join(', '), impact: 0.14 },
                    { feature: 'Release Timing', value: formData.releaseMonth, impact: 0.12 },
                    { feature: 'Sequel/Franchise', value: formData.isSequel ? 'Yes' : 'No', impact: 0.10 },
                    { feature: 'Pre-release Buzz', value: formData.trailerViews || 0, impact: 0.09 }
                ]
            }
        };
    };

    const getCategoryClass = (category) => {
        if (!category) return '';
        const normalized = category.toLowerCase().replace(/\s+/g, '-');
        if (['blockbuster', 'super-hit'].includes(normalized)) return 'category-blockbuster';
        if (normalized === 'hit') return 'category-hit';
        if (normalized === 'average') return 'category-average';
        return 'category-flop';
    };

    const formatCurrency = (value) => {
        if (!value) return 'N/A';
        if (value >= 1000000000) return `$${(value / 1000000000).toFixed(1)}B`;
        if (value >= 1000000) return `$${(value / 1000000).toFixed(0)}M`;
        return `$${parseInt(value).toLocaleString()}`;
    };

    return (
        <div className="upcoming-dashboard">
            {/* Header */}
            <section className="dashboard-header">
                <div className="dashboard-header-bg"></div>
                <div className="container">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="dashboard-header-content"
                    >
                        <h1 className="dashboard-title font-display">
                            Upcoming Movies <span className="text-gradient">Dashboard</span>
                        </h1>
                        <p className="dashboard-subtitle">
                            Predict success for movies in pre-production or upcoming releases
                        </p>

                        {/* Section Toggle */}
                        <div className="dashboard-tabs">
                            {[
                                { id: 'predict', label: '🔮 Predict Success', icon: '🔮' },
                                { id: 'simulate', label: '🎛️ What-If Simulation', icon: '🎛️' },
                                { id: 'release', label: '📅 Optimal Release', icon: '📅' }
                            ].map(section => (
                                <button
                                    key={section.id}
                                    className={`dashboard-tab ${activeSection === section.id ? 'active' : ''}`}
                                    onClick={() => setActiveSection(section.id)}
                                >
                                    {section.label}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Main Content */}
            <section className="dashboard-content">
                <div className="container">
                    {activeSection === 'predict' && (
                        <div className="predict-section">
                            <div className="predict-grid">
                                {/* Input Form */}
                                <motion.div
                                    className="predict-form-card glass-card"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                >
                                    <h2 className="card-title">Movie Details</h2>
                                    <form onSubmit={handleSubmit} className="predict-form">
                                        {/* Basic Info */}
                                        <div className="form-section">
                                            <h3 className="form-section-title">Basic Information</h3>

                                            <div className="form-group">
                                                <label className="form-label">Movie Title *</label>
                                                <input
                                                    type="text"
                                                    name="title"
                                                    className="input"
                                                    value={formData.title}
                                                    onChange={handleInputChange}
                                                    placeholder="Enter movie title"
                                                    required
                                                />
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">Industry *</label>
                                                <div className="industry-options">
                                                    {industries.map(ind => (
                                                        <button
                                                            key={ind.value}
                                                            type="button"
                                                            className={`industry-option ${formData.industry === ind.value ? 'active' : ''}`}
                                                            onClick={() => setFormData(prev => ({ ...prev, industry: ind.value }))}
                                                        >
                                                            {ind.label}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">Genres *</label>
                                                <div className="genre-options">
                                                    {genres.map(genre => (
                                                        <button
                                                            key={genre}
                                                            type="button"
                                                            className={`chip ${formData.genres.includes(genre) ? 'active' : ''}`}
                                                            onClick={() => handleGenreToggle(genre)}
                                                        >
                                                            {genre}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            <div className="form-row">
                                                <div className="form-group">
                                                    <label className="form-label">Budget ($) *</label>
                                                    <input
                                                        type="number"
                                                        name="budget"
                                                        className="input"
                                                        value={formData.budget}
                                                        onChange={handleInputChange}
                                                        placeholder="e.g., 150000000"
                                                        required
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Runtime (min)</label>
                                                    <input
                                                        type="number"
                                                        name="runtime"
                                                        className="input"
                                                        value={formData.runtime}
                                                        onChange={handleInputChange}
                                                        placeholder="e.g., 120"
                                                    />
                                                </div>
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">Target Release Month</label>
                                                <select
                                                    name="releaseMonth"
                                                    className="input"
                                                    value={formData.releaseMonth}
                                                    onChange={handleInputChange}
                                                >
                                                    <option value="">Select month</option>
                                                    {months.map((month, idx) => (
                                                        <option key={month} value={idx + 1}>{month}</option>
                                                    ))}
                                                </select>
                                            </div>
                                        </div>

                                        {/* Talent Info */}
                                        <div className="form-section">
                                            <h3 className="form-section-title">Talent Information</h3>

                                            <div className="form-group">
                                                <label className="form-label">Director</label>
                                                <DirectorSearch
                                                    selectedDirector={selectedDirector}
                                                    onSelect={(director) => {
                                                        setSelectedDirector(director);
                                                        setFormData(prev => ({
                                                            ...prev,
                                                            directorName: director.name,
                                                            directorHitRate: director.metrics?.hitRate || 0
                                                        }));
                                                    }}
                                                    onClear={() => {
                                                        setSelectedDirector(null);
                                                        setFormData(prev => ({
                                                            ...prev,
                                                            directorName: '',
                                                            directorHitRate: ''
                                                        }));
                                                    }}
                                                />
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">Cast Members</label>
                                                <CastSearch
                                                    selectedCast={selectedCast}
                                                    combinedScore={castScore}
                                                    onCastChange={(cast, score) => {
                                                        setSelectedCast(cast);
                                                        setCastScore(score);
                                                        setFormData(prev => ({
                                                            ...prev,
                                                            leadActorPopularity: score
                                                        }));
                                                    }}
                                                />
                                            </div>

                                            <div className="form-group">
                                                <label className="form-label">Production Company</label>
                                                <input
                                                    type="text"
                                                    name="productionCompany"
                                                    className="input"
                                                    value={formData.productionCompany}
                                                    onChange={handleInputChange}
                                                    placeholder="e.g., Warner Bros."
                                                />
                                            </div>

                                            <div className="form-group checkbox-group">
                                                <label className="checkbox-label">
                                                    <input
                                                        type="checkbox"
                                                        name="isSequel"
                                                        checked={formData.isSequel}
                                                        onChange={handleInputChange}
                                                    />
                                                    <span className="checkbox-custom"></span>
                                                    Part of a Sequel/Franchise
                                                </label>
                                            </div>
                                        </div>

                                        {/* Pre-release Buzz */}
                                        <div className="form-section">
                                            <h3 className="form-section-title">Pre-release Buzz (Optional)</h3>

                                            <div className="form-row">
                                                <div className="form-group">
                                                    <label className="form-label">Trailer Views</label>
                                                    <input
                                                        type="number"
                                                        name="trailerViews"
                                                        className="input"
                                                        value={formData.trailerViews}
                                                        onChange={handleInputChange}
                                                        placeholder="e.g., 50000000"
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Google Trends Score</label>
                                                    <input
                                                        type="number"
                                                        name="googleTrendsScore"
                                                        className="input"
                                                        value={formData.googleTrendsScore}
                                                        onChange={handleInputChange}
                                                        placeholder="0-100"
                                                        min="0"
                                                        max="100"
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        <button
                                            type="submit"
                                            className="btn btn-primary btn-block"
                                            disabled={loading || !formData.title || formData.genres.length === 0 || !formData.budget}
                                        >
                                            {loading ? (
                                                <>
                                                    <span className="btn-spinner"></span>
                                                    Predicting...
                                                </>
                                            ) : (
                                                <>
                                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                        <circle cx="12" cy="12" r="10" />
                                                        <path d="M12 16v-4M12 8h.01" />
                                                    </svg>
                                                    Get Prediction
                                                </>
                                            )}
                                        </button>
                                    </form>
                                </motion.div>

                                {/* Prediction Results */}
                                <motion.div
                                    className="predict-results"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                >
                                    {prediction ? (
                                        <div className="results-card glass-card">
                                            <h2 className="card-title">Prediction Results</h2>

                                            {/* Main Prediction */}
                                            <div className="prediction-main">
                                                <div className="prediction-category">
                                                    <span className={`category-label ${getCategoryClass(prediction.predictions?.successCategory)}`}>
                                                        {prediction.predictions?.successCategory || 'Processing...'}
                                                    </span>
                                                    <span className="confidence-label">
                                                        {prediction.predictions?.confidence || 0}% Confidence
                                                    </span>
                                                </div>

                                                <div className="prediction-metrics">
                                                    <div className="metric-card">
                                                        <span className="metric-icon">💰</span>
                                                        <div className="metric-info">
                                                            <span className="metric-value">
                                                                {formatCurrency(prediction.predictions?.predictedRevenue)}
                                                            </span>
                                                            <span className="metric-label">Predicted Revenue</span>
                                                        </div>
                                                    </div>

                                                    <div className="metric-card">
                                                        <span className="metric-icon">⭐</span>
                                                        <div className="metric-info">
                                                            <span className="metric-value">
                                                                {prediction.predictions?.confidence ? (Math.min(prediction.predictions.confidence / 10, 10)).toFixed(1) : 'N/A'}/10
                                                            </span>
                                                            <span className="metric-label">Predicted Rating</span>
                                                        </div>
                                                    </div>

                                                    <div className="metric-card">
                                                        <span className="metric-icon">📈</span>
                                                        <div className="metric-info">
                                                            <span className={`metric-value ${(prediction.predictions?.predictedROI || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                                                                {(prediction.predictions?.predictedROI || 0) >= 0 ? '+' : ''}{Math.round(prediction.predictions?.predictedROI || 0)}%
                                                            </span>
                                                            <span className="metric-label">Predicted ROI</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Feature Importance */}
                                            {prediction.predictions?.featureImportance && (
                                                <div className="feature-importance-section">
                                                    <h3 className="section-subtitle">Why This Prediction? (XAI)</h3>
                                                    <div className="feature-list">
                                                        {prediction.predictions.featureImportance.map((feature, idx) => (
                                                            <div key={idx} className="feature-item">
                                                                <div className="feature-header">
                                                                    <span className="feature-name">{feature.feature}</span>
                                                                    <span className="feature-impact">{(feature.impact * 100).toFixed(0)}%</span>
                                                                </div>
                                                                <div className="feature-bar-bg">
                                                                    <motion.div
                                                                        className="feature-bar"
                                                                        initial={{ width: 0 }}
                                                                        animate={{ width: `${feature.impact * 100}%` }}
                                                                        transition={{ delay: idx * 0.1, duration: 0.5 }}
                                                                    />
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="results-placeholder glass-card">
                                            <div className="placeholder-icon">🎯</div>
                                            <h3>Ready to Predict</h3>
                                            <p>Fill in the movie details and click "Get Prediction" to see AI-powered success analysis.</p>
                                            <div className="placeholder-features">
                                                <span>💰 Revenue Prediction</span>
                                                <span>⭐ Rating Prediction</span>
                                                <span>📊 Success Classification</span>
                                                <span>🧠 Explainable AI</span>
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            </div>
                        </div>
                    )}

                    {activeSection === 'simulate' && (
                        <motion.div
                            className="simulate-section glass-card"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <h2 className="card-title">🎛️ What-If Simulation</h2>
                            <p className="card-description">
                                Test different scenarios by modifying movie parameters and see how predictions change.
                            </p>
                            <div className="coming-soon">
                                <span className="coming-soon-icon">🚧</span>
                                <h3>Coming Soon</h3>
                                <p>This feature will allow you to test scenarios like:</p>
                                <ul>
                                    <li>What if we increase the budget by 50%?</li>
                                    <li>What if we change the lead actor?</li>
                                    <li>What if we release in summer instead of winter?</li>
                                </ul>
                            </div>
                        </motion.div>
                    )}

                    {activeSection === 'release' && (
                        <motion.div
                            className="release-section glass-card"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <h2 className="card-title">📅 Optimal Release Date Finder</h2>
                            <p className="card-description">
                                Find the best release window based on competition analysis and seasonal trends.
                            </p>
                            <div className="coming-soon">
                                <span className="coming-soon-icon">🚧</span>
                                <h3>Coming Soon</h3>
                                <p>This feature will analyze:</p>
                                <ul>
                                    <li>Competing releases in your target window</li>
                                    <li>Historical performance by month/season</li>
                                    <li>Genre-specific optimal release periods</li>
                                    <li>Holiday and event-based recommendations</li>
                                </ul>
                            </div>
                        </motion.div>
                    )}
                </div>
            </section>
        </div>
    );
};

export default UpcomingDashboard;
