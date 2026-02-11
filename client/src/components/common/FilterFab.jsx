import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './FilterFab.css';

/**
 * Floating Action Button with Filter Popup
 * Shows filter options in a card popup when clicked
 */
const FilterFab = ({
    filters,
    onFilterChange,
    industries,
    genres,
    categories,
    sortOptions,
    onClear
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const popupRef = useRef(null);

    // Count active filters
    const activeFilterCount = [
        filters.industry,
        filters.genre,
        filters.category,
        filters.sortBy !== 'popularity' ? filters.sortBy : null
    ].filter(Boolean).length;

    // Close popup when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (popupRef.current && !popupRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isOpen]);

    const handleClear = () => {
        onClear();
        setIsOpen(false);
    };

    return (
        <div className="filter-fab-container" ref={popupRef}>
            {/* Popup Card */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        className="filter-popup"
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div className="filter-popup-header">
                            <h3>Filters</h3>
                            {activeFilterCount > 0 && (
                                <button className="clear-btn" onClick={handleClear}>
                                    Clear All
                                </button>
                            )}
                        </div>

                        <div className="filter-popup-content">
                            {/* Industry Filter */}
                            <div className="fab-filter-group">
                                <label>🎬 Industry</label>
                                <div className="fab-chips">
                                    {industries.map(ind => (
                                        <button
                                            key={ind}
                                            className={`fab-chip ${filters.industry === ind || (ind === 'All' && !filters.industry) ? 'active' : ''}`}
                                            onClick={() => onFilterChange('industry', ind === 'All' ? '' : ind)}
                                        >
                                            {ind}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Genre Filter */}
                            <div className="fab-filter-group">
                                <label>🎭 Genre</label>
                                <select
                                    className="fab-select"
                                    value={filters.genre}
                                    onChange={(e) => onFilterChange('genre', e.target.value)}
                                >
                                    {genres.map(g => (
                                        <option key={g} value={g === 'All' ? '' : g}>{g}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Success Filter */}
                            <div className="fab-filter-group">
                                <label>⭐ Success Category</label>
                                <div className="fab-chips">
                                    {categories.map(c => (
                                        <button
                                            key={c}
                                            className={`fab-chip ${filters.category === c || (c === 'All' && !filters.category) ? 'active' : ''}`}
                                            onClick={() => onFilterChange('category', c === 'All' ? '' : c)}
                                        >
                                            {c}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Sort */}
                            <div className="fab-filter-group">
                                <label>📊 Sort By</label>
                                <select
                                    className="fab-select"
                                    value={filters.sortBy}
                                    onChange={(e) => onFilterChange('sortBy', e.target.value)}
                                >
                                    {sortOptions.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="filter-popup-footer">
                            <button className="apply-btn" onClick={() => setIsOpen(false)}>
                                Apply Filters
                            </button>
                            <button className="reset-btn" onClick={handleClear}>
                                Reset All Filters
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* FAB Button */}
            <motion.button
                className={`filter-fab ${isOpen ? 'open' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
            >
                <span className="fab-icon">
                    {isOpen ? <i className="fa-solid fa-xmark"></i> : <i className="fa-solid fa-filter"></i>}
                </span>
                {activeFilterCount > 0 && !isOpen && (
                    <span className="fab-badge">{activeFilterCount}</span>
                )}
            </motion.button>
        </div>
    );
};

export default FilterFab;
