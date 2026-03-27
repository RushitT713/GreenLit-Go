import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { movieService } from '../services/api';
import MovieCard from '../components/movies/MovieCard';
import FilterFab from '../components/common/FilterFab';
import './ReleasedMovies.css';

const MOODS = [
    { id: 'feel-good', label: 'Feel-Good', emoji: '😊', color: '#FFD93D' },
    { id: 'thrilling', label: 'Thrilling', emoji: '⚡', color: '#FF6B6B' },
    { id: 'mind-bending', label: 'Mind-Bending', emoji: '🌀', color: '#A855F7' },
    { id: 'romantic', label: 'Romantic', emoji: '💕', color: '#F472B6' },
    { id: 'dark-intense', label: 'Dark & Intense', emoji: '🌑', color: '#64748B' },
    { id: 'epic-adventure', label: 'Epic Adventure', emoji: '⚔️', color: '#F59E0B' },
    { id: 'laugh-out-loud', label: 'Laugh Out Loud', emoji: '😂', color: '#34D399' },
    { id: 'emotional', label: 'Emotional', emoji: '🥺', color: '#60A5FA' },
    { id: 'scary', label: 'Scary', emoji: '👻', color: '#EF4444' },
    { id: 'inspirational', label: 'Inspirational', emoji: '✨', color: '#FBBF24' },
];

const DISCOVER_MODES = [
    { id: 'mood', label: 'By Mood', emoji: '🎭', desc: 'How are you feeling today?' },
    { id: 'genre', label: 'Genre Mix', emoji: '🎬', desc: 'Pick your favorite genres' },
    { id: 'similar', label: 'Similar To', emoji: '🎥', desc: 'Find movies like one you love' },
    { id: 'talent', label: 'By Talent', emoji: '⭐', desc: 'Search by actor or director' },
    { id: 'top-rated', label: 'Top Rated', emoji: '🏆', desc: 'Best of the best' },
    { id: 'surprise', label: 'Surprise Me', emoji: '🎲', desc: 'Random hidden gems' },
];

const ALL_GENRES = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Biography', 'History', 'Music', 'Family', 'Sport'];

const ERAS = [
    { value: '2020-2026', label: '2020s' },
    { value: '2010-2019', label: '2010s' },
    { value: '2000-2009', label: '2000s' },
    { value: '1990-1999', label: '1990s' },
    { value: '1970-1989', label: 'Classics' },
];

const ReleasedMovies = () => {
    const [searchParams] = useSearchParams();
    const [activeTab, setActiveTab] = useState('library');

    // ─── Library State ───
    const [movies, setMovies] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState(searchParams.get('q') || '');
    const [filters, setFilters] = useState({
        industry: searchParams.get('industry') || '',
        genre: searchParams.get('genre') || '',
        category: searchParams.get('category') || '',
        sortBy: searchParams.get('sortBy') || 'popularity'
    });
    const [pagination, setPagination] = useState({
        page: 1,
        total: 0,
        totalPages: 0
    });

    const industries = ['All', 'Hollywood', 'Bollywood', 'Tollywood', 'Kollywood', 'Mollywood'];
    const genres = ['All', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'];
    const categories = ['All', 'Blockbuster', 'Hit', 'Average', 'Flop'];
    const sortOptions = [
        { value: 'popularity', label: 'Most Popular' },
        { value: 'voteAverage', label: 'Highest Rated' },
        { value: 'releaseDate', label: 'Latest Release' },
        { value: 'revenue', label: 'Highest Grossing' },
    ];

    // ─── Discover State ───
    const [discoverMode, setDiscoverMode] = useState(null);
    const [selectedMood, setSelectedMood] = useState('');
    const [selectedGenres, setSelectedGenres] = useState([]);
    const [similarSearch, setSimilarSearch] = useState('');
    const [similarResults, setSimilarResults] = useState([]);
    const [selectedMovie, setSelectedMovie] = useState(null);
    const [talentName, setTalentName] = useState('');
    const [talentType, setTalentType] = useState('actor');
    const [discoverIndustry, setDiscoverIndustry] = useState('all');
    const [selectedEra, setSelectedEra] = useState('');
    const [recommendations, setRecommendations] = useState([]);
    const [recMeta, setRecMeta] = useState(null);
    const [recLoading, setRecLoading] = useState(false);
    const [searchingMovies, setSearchingMovies] = useState(false);

    // ─── Library Logic ───
    useEffect(() => {
        if (activeTab === 'library') fetchMovies();
    }, [filters, pagination.page, searchQuery, activeTab]);

    const fetchMovies = async () => {
        try {
            setLoading(true);
            const params = {
                page: pagination.page,
                limit: 20,
                sortBy: filters.sortBy,
                ...(searchQuery && { search: searchQuery }),
                ...(filters.industry && filters.industry !== 'All' && { industry: filters.industry.toLowerCase() }),
                ...(filters.genre && filters.genre !== 'All' && { genre: filters.genre.toLowerCase() }),
                ...(filters.category && filters.category !== 'All' && { category: filters.category.toLowerCase() }),
            };

            const response = await movieService.getAll(params);
            if (response.data?.movies) {
                setMovies(response.data.movies);
                setPagination(prev => ({
                    ...prev,
                    total: response.data.pagination?.total || 0,
                    totalPages: response.data.pagination?.pages || 1
                }));
            } else {
                setMovies([]);
            }
        } catch (error) {
            setMovies([]);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = (e) => {
        e.preventDefault();
        setPagination(prev => ({ ...prev, page: 1 }));
    };

    const handleFilterChange = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
        setPagination(prev => ({ ...prev, page: 1 }));
    };

    const handleClearFilters = () => {
        setFilters({ industry: '', genre: '', category: '', sortBy: 'popularity' });
        setPagination(prev => ({ ...prev, page: 1 }));
    };

    // ─── Discover Logic ───
    const handleGenreToggle = (genre) => {
        setSelectedGenres(prev =>
            prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]
        );
    };

    const searchMoviesForSimilar = async (query) => {
        setSimilarSearch(query);
        if (query.length < 2) {
            setSimilarResults([]);
            return;
        }
        setSearchingMovies(true);
        try {
            const response = await movieService.getAll({ search: query, limit: 6 });
            setSimilarResults(response.data?.movies || []);
        } catch {
            setSimilarResults([]);
        } finally {
            setSearchingMovies(false);
        }
    };

    const selectMovieForSimilar = (movie) => {
        setSelectedMovie(movie);
        setSimilarSearch(movie.title);
        setSimilarResults([]);
    };

    const getRecommendations = async () => {
        if (!discoverMode) return;

        setRecLoading(true);
        setRecommendations([]);
        setRecMeta(null);

        try {
            const payload = {
                mode: discoverMode,
                industry: discoverIndustry,
                limit: 20
            };

            switch (discoverMode) {
                case 'mood':
                    if (!selectedMood) return setRecLoading(false);
                    payload.mood = selectedMood;
                    break;
                case 'genre':
                    if (selectedGenres.length === 0) return setRecLoading(false);
                    payload.genres = selectedGenres;
                    break;
                case 'similar':
                    if (!selectedMovie) return setRecLoading(false);
                    payload.movieId = selectedMovie._id;
                    break;
                case 'talent':
                    if (!talentName.trim()) return setRecLoading(false);
                    payload.talentName = talentName;
                    payload.talentType = talentType;
                    break;
                case 'top-rated':
                    if (selectedEra) payload.era = selectedEra;
                    break;
                case 'surprise':
                    break;
            }

            const response = await movieService.recommend(payload);
            setRecommendations(response.data?.recommendations || []);
            setRecMeta(response.data?.meta || {});
        } catch (error) {
            console.error('Recommendation error:', error);
        } finally {
            setRecLoading(false);
        }
    };

    const resetDiscover = () => {
        setDiscoverMode(null);
        setSelectedMood('');
        setSelectedGenres([]);
        setSimilarSearch('');
        setSelectedMovie(null);
        setTalentName('');
        setSelectedEra('');
        setRecommendations([]);
        setRecMeta(null);
    };

    // ─── Render Helpers ───
    const renderModeInputs = () => {
        switch (discoverMode) {
            case 'mood':
                return (
                    <div className="discover-input-section">
                        <h3>How are you feeling today?</h3>
                        <div className="mood-chips">
                            {MOODS.map(mood => (
                                <button
                                    key={mood.id}
                                    className={`mood-chip ${selectedMood === mood.id ? 'active' : ''}`}
                                    style={{ '--mood-color': mood.color }}
                                    onClick={() => setSelectedMood(mood.id)}
                                >
                                    <span className="mood-emoji">{mood.emoji}</span>
                                    <span>{mood.label}</span>
                                </button>
                            ))}
                        </div>
                        <div className="discover-industry-filter">
                            <label>Industry:</label>
                            <select value={discoverIndustry} onChange={e => setDiscoverIndustry(e.target.value)}>
                                <option value="all">All Industries</option>
                                {industries.filter(i => i !== 'All').map(i => (
                                    <option key={i} value={i.toLowerCase()}>{i}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                );

            case 'genre':
                return (
                    <div className="discover-input-section">
                        <h3>Pick your favorite genres</h3>
                        <div className="genre-chips">
                            {ALL_GENRES.map(genre => (
                                <button
                                    key={genre}
                                    className={`genre-chip ${selectedGenres.includes(genre) ? 'active' : ''}`}
                                    onClick={() => handleGenreToggle(genre)}
                                >
                                    {genre}
                                </button>
                            ))}
                        </div>
                        {selectedGenres.length > 0 && (
                            <p className="selected-info">Selected: {selectedGenres.join(', ')}</p>
                        )}
                    </div>
                );

            case 'similar':
                return (
                    <div className="discover-input-section">
                        <h3>Find movies similar to...</h3>
                        <div className="similar-search-wrapper">
                            <input
                                type="text"
                                className="discover-input"
                                placeholder="Search for a movie..."
                                value={similarSearch}
                                onChange={(e) => searchMoviesForSimilar(e.target.value)}
                            />
                            {searchingMovies && <div className="search-spinner"></div>}
                        </div>
                        {similarResults.length > 0 && !selectedMovie && (
                            <div className="similar-dropdown">
                                {similarResults.map(movie => (
                                    <button
                                        key={movie._id}
                                        className="similar-option"
                                        onClick={() => selectMovieForSimilar(movie)}
                                    >
                                        <img
                                            src={movie.posterPath ? `https://image.tmdb.org/t/p/w92${movie.posterPath}` : ''}
                                            alt=""
                                            className="similar-poster"
                                        />
                                        <div>
                                            <strong>{movie.title}</strong>
                                            <span className="similar-year">{movie.year}</span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                        {selectedMovie && (
                            <div className="selected-movie-badge">
                                <img
                                    src={selectedMovie.posterPath ? `https://image.tmdb.org/t/p/w92${selectedMovie.posterPath}` : ''}
                                    alt=""
                                />
                                <span>{selectedMovie.title} ({selectedMovie.year})</span>
                                <button onClick={() => { setSelectedMovie(null); setSimilarSearch(''); }}>✕</button>
                            </div>
                        )}
                    </div>
                );

            case 'talent':
                return (
                    <div className="discover-input-section">
                        <h3>Search by talent</h3>
                        <div className="talent-controls">
                            <div className="talent-type-toggle">
                                <button
                                    className={talentType === 'actor' ? 'active' : ''}
                                    onClick={() => setTalentType('actor')}
                                >Actor</button>
                                <button
                                    className={talentType === 'director' ? 'active' : ''}
                                    onClick={() => setTalentType('director')}
                                >Director</button>
                            </div>
                            <input
                                type="text"
                                className="discover-input"
                                placeholder={`Enter ${talentType} name...`}
                                value={talentName}
                                onChange={(e) => setTalentName(e.target.value)}
                            />
                        </div>
                    </div>
                );

            case 'top-rated':
                return (
                    <div className="discover-input-section">
                        <h3>Best movies of all time</h3>
                        <div className="top-rated-controls">
                            <div>
                                <label>Industry:</label>
                                <select value={discoverIndustry} onChange={e => setDiscoverIndustry(e.target.value)}>
                                    <option value="all">All Industries</option>
                                    {industries.filter(i => i !== 'All').map(i => (
                                        <option key={i} value={i.toLowerCase()}>{i}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label>Era:</label>
                                <select value={selectedEra} onChange={e => setSelectedEra(e.target.value)}>
                                    <option value="">All Time</option>
                                    {ERAS.map(era => (
                                        <option key={era.value} value={era.value}>{era.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>
                );

            case 'surprise':
                return (
                    <div className="discover-input-section">
                        <h3>Feeling lucky? 🎲</h3>
                        <p className="surprise-desc">We'll pick some hidden gems for you from our database of {pagination.total || '1,650'}+ movies!</p>
                        <div className="discover-industry-filter">
                            <label>Industry:</label>
                            <select value={discoverIndustry} onChange={e => setDiscoverIndustry(e.target.value)}>
                                <option value="all">All Industries</option>
                                {industries.filter(i => i !== 'All').map(i => (
                                    <option key={i} value={i.toLowerCase()}>{i}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="released-movies">
            {/* Header */}
            <section className="page-header">
                <div className="container">
                    <motion.div
                        className="page-header-content"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <h1 className="page-title">Movie <span className="text-gradient">{activeTab === 'library' ? 'Library' : 'Discover'}</span></h1>
                        <p className="page-subtitle">{activeTab === 'library'
                            ? 'Explore our comprehensive database of movies with AI-powered success predictions'
                            : 'Get personalized movie recommendations powered by our intelligent engine'
                        }</p>

                        {/* Tab Switcher — inside header, centered */}
                        <div className="dashboard-tabs">
                            {[
                                { id: 'library', label: '📚 Library' },
                                { id: 'discover', label: '✨ Discover' }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    className={`dashboard-tab ${activeTab === tab.id ? 'active' : ''}`}
                                    onClick={() => setActiveTab(tab.id)}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </section>

            <AnimatePresence mode="wait">
                {/* ═══════════ LIBRARY TAB ═══════════ */}
                {activeTab === 'library' && (
                    <motion.div
                        key="library"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        {/* Search and Filters */}
                        <section className="filters-section">
                            <div className="container">
                                <div className="filters-header-row">
                                    <form onSubmit={handleSearch} className="search-form">
                                        <div className="search-input-wrapper">
                                            <span className="search-icon">🔍</span>
                                            <input
                                                type="text"
                                                className="search-input"
                                                placeholder="Search movies by title, director, or actor..."
                                                value={searchQuery}
                                                onChange={(e) => setSearchQuery(e.target.value)}
                                            />
                                        </div>
                                    </form>
                                    {/* Filter Button */}
                                    <FilterFab
                                        filters={filters}
                                        onFilterChange={handleFilterChange}
                                        industries={industries}
                                        genres={genres}
                                        categories={categories}
                                        sortOptions={sortOptions}
                                        onClear={handleClearFilters}
                                    />
                                </div>
                            </div>
                        </section>

                        {/* Movies Grid */}
                        <section className="movies-section">
                            <div className="container">
                                {loading ? (
                                    <div className="movies-grid">
                                        {[...Array(12)].map((_, i) => (
                                            <div key={i} className="movie-skeleton">
                                                <div className="skeleton skeleton-poster"></div>
                                                <div className="skeleton skeleton-title"></div>
                                            </div>
                                        ))}
                                    </div>
                                ) : movies.length > 0 ? (
                                    <>
                                        <div className="movies-grid">
                                            {movies.map((movie, index) => (
                                                <motion.div
                                                    key={movie._id}
                                                    initial={{ opacity: 0, y: 20 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: index * 0.03, duration: 0.4 }}
                                                >
                                                    <MovieCard movie={movie} />
                                                </motion.div>
                                            ))}
                                        </div>
                                        {pagination.totalPages > 1 && (
                                            <div className="pagination">
                                                <button
                                                    className="btn btn-secondary"
                                                    onClick={() => setPagination(prev => ({ ...prev, page: prev.page - 1 }))}
                                                    disabled={pagination.page === 1}
                                                >← Previous</button>
                                                <span className="page-info">
                                                    Page <strong>{pagination.page}</strong> of <strong>{pagination.totalPages}</strong>
                                                </span>
                                                <button
                                                    className="btn btn-secondary"
                                                    onClick={() => setPagination(prev => ({ ...prev, page: prev.page + 1 }))}
                                                    disabled={pagination.page >= pagination.totalPages}
                                                >Next →</button>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="no-results">
                                        <span className="no-results-icon">🎬</span>
                                        <h3>No movies found</h3>
                                        <p>Try adjusting your filters or search query</p>
                                    </div>
                                )}
                            </div>
                        </section>

                        {/* Filter FAB moved to filters-section */}
                    </motion.div>
                )}

                {/* ═══════════ DISCOVER TAB ═══════════ */}
                {activeTab === 'discover' && (
                    <motion.div
                        key="discover"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3 }}
                    >
                        <section className="discover-section">
                            <div className="container">
                                {/* Mode Selector */}
                                {!discoverMode ? (
                                    <div className="discover-modes">
                                        <h2 className="discover-title">How would you like to discover movies?</h2>
                                        <div className="mode-grid">
                                            {DISCOVER_MODES.map((mode, index) => (
                                                <motion.button
                                                    key={mode.id}
                                                    className="mode-card"
                                                    onClick={() => setDiscoverMode(mode.id)}
                                                    initial={{ opacity: 0, y: 30 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: index * 0.08, duration: 0.4 }}
                                                    whileHover={{ scale: 1.05, y: -5 }}
                                                    whileTap={{ scale: 0.97 }}
                                                >
                                                    <span className="mode-emoji">{mode.emoji}</span>
                                                    <span className="mode-label">{mode.label}</span>
                                                    <span className="mode-desc">{mode.desc}</span>
                                                </motion.button>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="discover-active">
                                        {/* Back + Mode Title */}
                                        <div className="discover-header">
                                            <button className="back-btn" onClick={resetDiscover}>
                                                ← Back
                                            </button>
                                            <h2>
                                                {DISCOVER_MODES.find(m => m.id === discoverMode)?.emoji}{' '}
                                                {DISCOVER_MODES.find(m => m.id === discoverMode)?.label}
                                            </h2>
                                        </div>

                                        {/* Mode-specific inputs */}
                                        {renderModeInputs()}

                                        {/* Get Recommendations Button */}
                                        <div className="discover-action">
                                            <motion.button
                                                className="btn-recommend"
                                                onClick={getRecommendations}
                                                disabled={recLoading}
                                                whileHover={{ scale: 1.03 }}
                                                whileTap={{ scale: 0.97 }}
                                            >
                                                {recLoading ? (
                                                    <><span className="btn-spinner"></span> Finding movies...</>
                                                ) : (
                                                    <>🎯 Get Recommendations</>
                                                )}
                                            </motion.button>
                                        </div>

                                        {/* Results */}
                                        {recMeta && (
                                            <div className="rec-meta">
                                                {recMeta.description && <p className="rec-description">{recMeta.description}</p>}
                                                {recMeta.sourceMovie && (
                                                    <p className="rec-source">Similar to: <strong>{recMeta.sourceMovie.title}</strong> ({recMeta.sourceMovie.year})</p>
                                                )}
                                                {recMeta.talentName && (
                                                    <p className="rec-source">Movies featuring: <strong>{recMeta.talentName}</strong> ({recMeta.talentType})</p>
                                                )}
                                                <p className="rec-count">{recommendations.length} movies found</p>
                                            </div>
                                        )}

                                        {recLoading && (
                                            <div className="movies-grid">
                                                {[...Array(8)].map((_, i) => (
                                                    <div key={i} className="movie-skeleton">
                                                        <div className="skeleton skeleton-poster"></div>
                                                        <div className="skeleton skeleton-title"></div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {!recLoading && recommendations.length > 0 && (
                                            <div className="movies-grid">
                                                {recommendations.map((movie, index) => (
                                                    <motion.div
                                                        key={movie._id}
                                                        className="rec-card-wrapper"
                                                        initial={{ opacity: 0, y: 20 }}
                                                        animate={{ opacity: 1, y: 0 }}
                                                        transition={{ delay: index * 0.04, duration: 0.4 }}
                                                    >
                                                        <div className="match-badge">{movie.matchScore}% match</div>
                                                        <MovieCard movie={movie} />
                                                    </motion.div>
                                                ))}
                                            </div>
                                        )}

                                        {!recLoading && recMeta && recommendations.length === 0 && (
                                            <div className="no-results">
                                                <span className="no-results-icon">😕</span>
                                                <h3>No movies found</h3>
                                                <p>Try different preferences or another mode</p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </section>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default ReleasedMovies;
