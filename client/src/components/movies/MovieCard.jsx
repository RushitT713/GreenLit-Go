import { Link } from 'react-router-dom';
import './MovieCard.css';

const MovieCard = ({ movie }) => {
    const getPosterUrl = (path) => {
        if (!path) return null;
        if (path.startsWith('http')) return path;
        return `https://image.tmdb.org/t/p/w500${path}`;
    };

    const getCategoryClass = (category) => {
        if (!category) return '';
        const normalized = category.toLowerCase().replace(/\s+/g, '-');
        if (['blockbuster', 'super-hit'].includes(normalized)) return 'blockbuster';
        if (normalized === 'hit') return 'hit';
        if (normalized === 'average') return 'average';
        return 'flop';
    };

    const posterUrl = getPosterUrl(movie.posterPath);
    const rating = movie.voteAverage?.toFixed(1) || 'N/A';
    const category = movie.predictions?.successCategory || movie.successCategory;

    return (
        <Link to={`/movies/${movie._id}`} className="movie-card">
            <div className="movie-card-poster">
                {posterUrl ? (
                    <img src={posterUrl} alt={movie.title} loading="lazy" />
                ) : (
                    <div className="poster-placeholder">
                        <span>🎬</span>
                    </div>
                )}

                {/* Rating Badge */}
                <div className="card-rating">
                    <span className="star">⭐</span>
                    {rating}
                </div>

                {/* Overlay on Hover */}
                <div className="card-overlay">
                    <div className="overlay-content">
                        {category && (
                            <span className={`category-tag ${getCategoryClass(category)}`}>
                                {category}
                            </span>
                        )}
                        <span className="overlay-year">{movie.year}</span>
                        {movie.genres && (
                            <p className="overlay-genres">{movie.genres.slice(0, 2).join(' • ')}</p>
                        )}
                    </div>
                    <button className="overlay-btn">View Details</button>
                </div>
            </div>

            <div className="movie-card-info">
                <h3 className="movie-card-title">{movie.title}</h3>
                <div className="movie-card-meta">
                    <span>{movie.year}</span>
                    {movie.industry && (
                        <span className="meta-dot">•</span>
                    )}
                    {movie.industry && (
                        <span className="industry-tag">{movie.industry}</span>
                    )}
                </div>
            </div>
        </Link>
    );
};

export default MovieCard;
