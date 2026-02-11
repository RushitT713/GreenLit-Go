import { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { movieService } from '../services/api';
import MovieCard from '../components/movies/MovieCard';
import FilterFab from '../components/common/FilterFab';
import './ReleasedMovies.css';

const ReleasedMovies = () => {
    const [searchParams] = useSearchParams();
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

    useEffect(() => {
        fetchMovies();
    }, [filters, pagination.page, searchQuery]);

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
                setMovies(getDemoMovies());
            }
        } catch (error) {
            setMovies(getDemoMovies());
        } finally {
            setLoading(false);
        }
    };

    const getDemoMovies = () => [
        { _id: '1', title: 'Dune: Part Two', year: 2024, voteAverage: 8.3, genres: ['Sci-Fi'], industry: 'hollywood', posterPath: 'https://image.tmdb.org/t/p/w500/8b8R8l88Qje9dn9OE8PY05Nxl1X.jpg', predictions: { successCategory: 'Blockbuster' } },
        { _id: '2', title: 'Oppenheimer', year: 2023, voteAverage: 8.4, genres: ['Drama'], industry: 'hollywood', posterPath: 'https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg', predictions: { successCategory: 'Blockbuster' } },
        { _id: '3', title: 'John Wick 4', year: 2023, voteAverage: 7.9, genres: ['Action'], industry: 'hollywood', posterPath: 'https://image.tmdb.org/t/p/w500/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg', predictions: { successCategory: 'Hit' } },
        { _id: '4', title: 'Pathaan', year: 2023, voteAverage: 7.5, genres: ['Action'], industry: 'bollywood', posterPath: 'https://image.tmdb.org/t/p/w500/deZvgbD5SUmuhHQxnQQiQesbHkz.jpg', predictions: { successCategory: 'Blockbuster' } },
        { _id: '5', title: 'Inside Out 2', year: 2024, voteAverage: 7.6, genres: ['Animation'], industry: 'hollywood', posterPath: 'https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg', predictions: { successCategory: 'Blockbuster' } },
        { _id: '6', title: 'RRR', year: 2022, voteAverage: 8.0, genres: ['Action'], industry: 'tollywood', posterPath: 'https://image.tmdb.org/t/p/w500/nEufeZlyAOLqO2brrs0yeF1lgXO.jpg', predictions: { successCategory: 'Blockbuster' } },
    ];

    const handleSearch = (e) => {
        e.preventDefault();
        setPagination(prev => ({ ...prev, page: 1 }));
    };

    const handleFilterChange = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
        setPagination(prev => ({ ...prev, page: 1 }));
    };

    const handleClearFilters = () => {
        setFilters({
            industry: '',
            genre: '',
            category: '',
            sortBy: 'popularity'
        });
        setPagination(prev => ({ ...prev, page: 1 }));
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
                        <div className="page-header-text">
                            <h1>Movie <span>Library</span></h1>
                            <p>Explore our comprehensive database of movies with AI-powered success predictions and box office analytics</p>
                        </div>
                        <div className="header-stats">
                            <div className="header-stat">
                                <div className="header-stat-value">{pagination.total || '575'}+</div>
                                <div className="header-stat-label">Movies</div>
                            </div>
                            <div className="header-stat">
                                <div className="header-stat-value">5+</div>
                                <div className="header-stat-label">Industries</div>
                            </div>
                            <div className="header-stat">
                                <div className="header-stat-value">10+</div>
                                <div className="header-stat-label">Genres</div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Search Bar Section */}
            <section className="filters-section">
                <div className="container">
                    {/* Search */}
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

                            {/* Pagination */}
                            {pagination.totalPages > 1 && (
                                <div className="pagination">
                                    <button
                                        className="btn btn-secondary"
                                        onClick={() => setPagination(prev => ({ ...prev, page: prev.page - 1 }))}
                                        disabled={pagination.page === 1}
                                    >
                                        ← Previous
                                    </button>
                                    <span className="page-info">
                                        Page <strong>{pagination.page}</strong> of <strong>{pagination.totalPages}</strong>
                                    </span>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={() => setPagination(prev => ({ ...prev, page: prev.page + 1 }))}
                                        disabled={pagination.page >= pagination.totalPages}
                                    >
                                        Next →
                                    </button>
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

            {/* Filter FAB */}
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
    );
};

export default ReleasedMovies;
