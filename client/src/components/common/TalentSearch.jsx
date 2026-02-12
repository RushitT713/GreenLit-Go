import { useState, useEffect, useRef } from 'react';
import { talentsService } from '../../services/api';
import './TalentSearch.css';

/**
 * Director Search Component with auto-complete and metrics display
 */
export const DirectorSearch = ({ onSelect, selectedDirector, onClear }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [loadingMetrics, setLoadingMetrics] = useState(false);
    const inputRef = useRef(null);
    const dropdownRef = useRef(null);

    // Search for directors
    useEffect(() => {
        if (query.length < 2) {
            setResults([]);
            return;
        }

        const timer = setTimeout(async () => {
            setLoading(true);
            try {
                const response = await talentsService.search(query, 8);
                // Filter for people known for directing
                const directors = response.data.filter(
                    p => p.knownFor === 'Directing' || p.knownFor === 'Production'
                );
                setResults(directors.length > 0 ? directors : response.data.slice(0, 5));
            } catch (err) {
                console.error('Search error:', err);
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [query]);

    // Handle selection
    const handleSelect = async (person) => {
        setLoadingMetrics(true);
        setShowDropdown(false);
        setQuery('');

        try {
            const response = await talentsService.getDirector(person.id);
            onSelect(response.data);
        } catch (err) {
            console.error('Director fetch error:', err);
            // Fallback with basic info
            onSelect({
                id: person.id,
                name: person.name,
                profilePath: person.profilePath,
                metrics: { powerScore: Math.round(person.popularity), hitRate: 0.5 }
            });
        } finally {
            setLoadingMetrics(false);
        }
    };

    // Click outside to close
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
                inputRef.current && !inputRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    if (selectedDirector) {
        return (
            <div className="talent-selected">
                <div className="talent-card selected">
                    {selectedDirector.profilePath && (
                        <img src={selectedDirector.profilePath} alt={selectedDirector.name} />
                    )}
                    <div className="talent-info">
                        <span className="talent-name">{selectedDirector.name}</span>
                        <div className="talent-metrics">
                            <span className="metric power-score">
                                Power: {selectedDirector.metrics?.powerScore || 0}/100
                            </span>
                            <span className="metric hit-rate">
                                Hit Rate: {((selectedDirector.metrics?.hitRate || 0) * 100).toFixed(0)}%
                            </span>
                        </div>
                    </div>
                    <button type="button" className="remove-btn" onClick={onClear}>✕</button>
                </div>
            </div>
        );
    }

    return (
        <div className="talent-search">
            <div className="search-input-wrapper">
                <input
                    ref={inputRef}
                    type="text"
                    placeholder="🔍 Search director by name..."
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        setShowDropdown(true);
                    }}
                    onFocus={() => setShowDropdown(true)}
                    className="talent-search-input"
                />
                {loading && <span className="search-loading">⏳</span>}
            </div>

            {showDropdown && results.length > 0 && (
                <div className="search-dropdown" ref={dropdownRef}>
                    {results.map(person => (
                        <div
                            key={person.id}
                            className="search-result"
                            onClick={() => handleSelect(person)}
                        >
                            {person.profilePath ? (
                                <img src={person.profilePath} alt={person.name} />
                            ) : (
                                <div className="no-photo">👤</div>
                            )}
                            <div className="result-info">
                                <span className="result-name">{person.name}</span>
                                <span className="result-known">
                                    {person.knownForMovies?.map(m => m.title).join(', ') || person.knownFor}
                                </span>
                            </div>
                            <span className="result-popularity">
                                ⭐ {Math.round(person.popularity)}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {loadingMetrics && (
                <div className="loading-metrics">Loading director metrics...</div>
            )}
        </div>
    );
};

/**
 * Cast Multi-Select Component with combined score
 */
export const CastSearch = ({ selectedCast = [], onCastChange, combinedScore = 0 }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const inputRef = useRef(null);
    const dropdownRef = useRef(null);

    // Search for actors
    useEffect(() => {
        if (query.length < 2) {
            setResults([]);
            return;
        }

        const timer = setTimeout(async () => {
            setLoading(true);
            try {
                const response = await talentsService.search(query, 8);
                // Filter for actors
                const actors = response.data.filter(
                    p => p.knownFor === 'Acting'
                );
                setResults(actors.length > 0 ? actors : response.data.slice(0, 5));
            } catch (err) {
                console.error('Search error:', err);
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [query]);

    // Handle actor selection
    const handleSelect = async (person) => {
        // Don't add duplicates
        if (selectedCast.some(c => c.id === person.id)) {
            setQuery('');
            setShowDropdown(false);
            return;
        }

        try {
            const response = await talentsService.getActor(person.id);
            const newCast = [...selectedCast, response.data];

            // Calculate combined score
            const actorIds = newCast.map(a => a.id);
            const scoreResponse = await talentsService.getCastScore(actorIds);

            onCastChange(newCast, scoreResponse.data.combinedScore);
        } catch (err) {
            console.error('Actor fetch error:', err);
            // Fallback
            const newCast = [...selectedCast, {
                id: person.id,
                name: person.name,
                profilePath: person.profilePath,
                metrics: { starPower: Math.round(person.popularity) }
            }];
            onCastChange(newCast, combinedScore);
        }

        setQuery('');
        setShowDropdown(false);
    };

    // Remove actor
    const handleRemove = async (actorId) => {
        const newCast = selectedCast.filter(c => c.id !== actorId);

        if (newCast.length > 0) {
            try {
                const actorIds = newCast.map(a => a.id);
                const scoreResponse = await talentsService.getCastScore(actorIds);
                onCastChange(newCast, scoreResponse.data.combinedScore);
            } catch (err) {
                onCastChange(newCast, 0);
            }
        } else {
            onCastChange([], 0);
        }
    };

    // Click outside to close
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target) &&
                inputRef.current && !inputRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="cast-search">
            {/* Selected Cast */}
            {selectedCast.length > 0 && (
                <div className="selected-cast">
                    {selectedCast.map((actor, index) => (
                        <div key={actor.id} className="cast-chip">
                            {actor.profilePath && (
                                <img src={actor.profilePath} alt={actor.name} />
                            )}
                            <span className="cast-name">{actor.name}</span>
                            <span className="cast-score">{actor.metrics?.starPower || 0}</span>
                            <button type="button" className="remove-chip" onClick={() => handleRemove(actor.id)}>✕</button>
                        </div>
                    ))}
                </div>
            )}

            {/* Combined Score Display */}
            {selectedCast.length > 0 && (
                <div className="combined-score">
                    ⭐ Combined Cast Score: <strong>{combinedScore}/100</strong>
                </div>
            )}

            {/* Search Input */}
            {selectedCast.length < 10 && (
                <div className="search-input-wrapper">
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="🔍 Add cast member..."
                        value={query}
                        onChange={(e) => {
                            setQuery(e.target.value);
                            setShowDropdown(true);
                        }}
                        onFocus={() => setShowDropdown(true)}
                        className="talent-search-input"
                    />
                    {loading && <span className="search-loading">⏳</span>}
                </div>
            )}

            {/* Dropdown Results */}
            {showDropdown && results.length > 0 && (
                <div className="search-dropdown" ref={dropdownRef}>
                    {results.map(person => (
                        <div
                            key={person.id}
                            className={`search-result ${selectedCast.some(c => c.id === person.id) ? 'disabled' : ''}`}
                            onClick={() => handleSelect(person)}
                        >
                            {person.profilePath ? (
                                <img src={person.profilePath} alt={person.name} />
                            ) : (
                                <div className="no-photo">👤</div>
                            )}
                            <div className="result-info">
                                <span className="result-name">{person.name}</span>
                                <span className="result-known">
                                    {person.knownForMovies?.map(m => m.title).join(', ') || 'Actor'}
                                </span>
                            </div>
                            <span className="result-popularity">
                                ⭐ {Math.round(person.popularity)}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
