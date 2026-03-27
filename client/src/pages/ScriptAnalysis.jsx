import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { scriptService } from '../services/api';
import './ScriptAnalysis.css';

const ScriptAnalysis = () => {
    const [file, setFile] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [result, setResult] = useState(null);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => setIsDragging(false);

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) validateAndSetFile(droppedFile);
    };

    const handleFileSelect = (e) => {
        const selected = e.target.files[0];
        if (selected) validateAndSetFile(selected);
    };

    const validateAndSetFile = (f) => {
        const allowed = ['application/pdf', 'text/plain'];
        if (!allowed.includes(f.type)) {
            setError('Only PDF and TXT files are supported.');
            return;
        }
        if (f.size > 10 * 1024 * 1024) {
            setError('File must be smaller than 10 MB.');
            return;
        }
        setError('');
        setFile(f);
        setResult(null);
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const response = await scriptService.analyze(file);
            setResult(response.data);
        } catch (err) {
            const msg = err.response?.data?.error || 'Failed to analyze script. Please try again.';
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setResult(null);
        setError('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const analysis = result?.analysis;

    return (
        <div className="script-analysis-page">
            {/* Hero Header */}
            <section className="sa-header">
                <div className="container">
                    <motion.div
                        className="sa-header-content"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <h1 className="sa-title">Script <span className="text-gradient">Analysis</span></h1>
                        <p className="sa-subtitle">
                            Upload your movie script and let our AI deliver a comprehensive 360° analysis — plot, tone, pacing, audience demographics, and commercial potential.
                        </p>
                    </motion.div>
                </div>
            </section>

            <div className="container sa-body">
                {/* Upload Section */}
                {!result && (
                    <motion.div
                        className="sa-upload-section"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2, duration: 0.5 }}
                    >
                        <div
                            className={`sa-dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileSelect}
                                accept=".pdf,.txt"
                                hidden
                            />
                            {file ? (
                                <div className="sa-file-info">
                                    <span className="sa-file-icon">📄</span>
                                    <span className="sa-file-name">{file.name}</span>
                                    <span className="sa-file-size">({(file.size / 1024).toFixed(1)} KB)</span>
                                </div>
                            ) : (
                                <>
                                    <span className="sa-drop-icon">📜</span>
                                    <p className="sa-drop-text">Drag & drop your script here</p>
                                    <p className="sa-drop-subtext">or click to browse · PDF, TXT · Max 10 MB</p>
                                </>
                            )}
                        </div>

                        {error && <p className="sa-error">{error}</p>}

                        <div className="sa-actions">
                            <motion.button
                                className="sa-btn-analyze"
                                onClick={handleAnalyze}
                                disabled={!file || loading}
                                whileHover={{ scale: 1.03 }}
                                whileTap={{ scale: 0.97 }}
                            >
                                {loading ? (
                                    <><span className="sa-spinner"></span> Analyzing Script...</>
                                ) : (
                                    <>🧠 Analyze Script</>
                                )}
                            </motion.button>
                            {file && !loading && (
                                <button className="sa-btn-reset" onClick={handleReset}>Clear</button>
                            )}
                        </div>

                        {loading && (
                            <div className="sa-loading-hint">
                                <p>⏳ The AI is reading your script. This can take 30–60 seconds...</p>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Results Section */}
                <AnimatePresence>
                    {analysis && (
                        <motion.div
                            className="sa-results"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="sa-results-header">
                                <h2>📊 Analysis Results</h2>
                                <div className="sa-results-meta">
                                    <span>📄 {result.filename}</span>
                                    <span>📝 {result.textLength?.toLocaleString()} characters extracted</span>
                                </div>
                                <button className="sa-btn-new" onClick={handleReset}>Analyze Another Script</button>
                            </div>

                            {/* Overall Score */}
                            {analysis.overall_score && (
                                <motion.div
                                    className="sa-card sa-score-card"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: 0.1 }}
                                >
                                    <div className="sa-score-ring">
                                        <span className="sa-score-value">{analysis.overall_score}</span>
                                        <span className="sa-score-label">/10</span>
                                    </div>
                                    <h3>Overall Script Score</h3>
                                </motion.div>
                            )}

                            {/* Plot Summary */}
                            <motion.div
                                className="sa-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.15 }}
                            >
                                <h3>📖 Plot Summary</h3>
                                <p>{analysis.plot_summary}</p>
                            </motion.div>

                            {/* Tone & Pacing Row */}
                            <div className="sa-row">
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                >
                                    <h3>🎭 Tone</h3>
                                    <div className="sa-tag-row">
                                        <span className="sa-tag primary">{analysis.tone?.primary}</span>
                                        {analysis.tone?.secondary && (
                                            <span className="sa-tag secondary">{analysis.tone?.secondary}</span>
                                        )}
                                    </div>
                                    <p className="sa-desc">{analysis.tone?.description}</p>
                                </motion.div>
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.25 }}
                                >
                                    <h3>⚡ Pacing</h3>
                                    <span className="sa-tag primary">{analysis.pacing?.rating}</span>
                                    <p className="sa-desc">{analysis.pacing?.description}</p>
                                </motion.div>
                            </div>

                            {/* Genres & Themes */}
                            <div className="sa-row">
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 }}
                                >
                                    <h3>🎬 Predicted Genres</h3>
                                    <div className="sa-tag-row">
                                        {analysis.genre_prediction?.map((g, i) => (
                                            <span key={i} className="sa-tag genre">{g}</span>
                                        ))}
                                    </div>
                                </motion.div>
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.35 }}
                                >
                                    <h3>💡 Key Themes</h3>
                                    <div className="sa-tag-row">
                                        {analysis.themes?.map((t, i) => (
                                            <span key={i} className="sa-tag theme">{t}</span>
                                        ))}
                                    </div>
                                </motion.div>
                            </div>

                            {/* Strengths & Weaknesses */}
                            <div className="sa-row">
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    <h3>💪 Strengths</h3>
                                    <ul className="sa-list strengths">
                                        {analysis.strengths?.map((s, i) => (
                                            <li key={i}>{s}</li>
                                        ))}
                                    </ul>
                                </motion.div>
                                <motion.div
                                    className="sa-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.45 }}
                                >
                                    <h3>⚠️ Weaknesses</h3>
                                    <ul className="sa-list weaknesses">
                                        {analysis.weaknesses?.map((w, i) => (
                                            <li key={i}>{w}</li>
                                        ))}
                                    </ul>
                                </motion.div>
                            </div>

                            {/* Demographics */}
                            {analysis.demographics && (
                                <motion.div
                                    className="sa-card sa-demographics"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.5 }}
                                >
                                    <h3>👥 Audience Demographics</h3>
                                    <div className="sa-demo-grid">
                                        <div className="sa-demo-item">
                                            <span className="sa-demo-label">Target Age</span>
                                            <span className="sa-demo-value">{analysis.demographics.target_age}</span>
                                        </div>
                                        <div className="sa-demo-item">
                                            <span className="sa-demo-label">Gender Appeal</span>
                                            <span className="sa-demo-value">{analysis.demographics.target_gender}</span>
                                        </div>
                                        <div className="sa-demo-item">
                                            <span className="sa-demo-label">Market Appeal</span>
                                            <span className="sa-demo-value">{analysis.demographics.market_appeal}</span>
                                        </div>
                                    </div>
                                    <div className="sa-demo-section">
                                        <span className="sa-demo-label">Target Interests</span>
                                        <div className="sa-tag-row">
                                            {analysis.demographics.target_interests?.map((t, i) => (
                                                <span key={i} className="sa-tag interest">{t}</span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="sa-demo-section">
                                        <span className="sa-demo-label">Comparable Films</span>
                                        <div className="sa-tag-row">
                                            {analysis.demographics.comparable_films?.map((f, i) => (
                                                <span key={i} className="sa-tag film">{f}</span>
                                            ))}
                                        </div>
                                    </div>
                                </motion.div>
                            )}

                            {/* Success Indicators */}
                            {analysis.success_indicators && (
                                <motion.div
                                    className="sa-card sa-success-card"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.55 }}
                                >
                                    <h3>🎯 Success Indicators</h3>
                                    <div className="sa-indicators">
                                        <div className={`sa-indicator ${analysis.success_indicators.commercial_potential?.toLowerCase()}`}>
                                            <span className="sa-ind-label">Commercial</span>
                                            <span className="sa-ind-value">{analysis.success_indicators.commercial_potential}</span>
                                        </div>
                                        <div className={`sa-indicator ${analysis.success_indicators.critical_potential?.toLowerCase()}`}>
                                            <span className="sa-ind-label">Critical</span>
                                            <span className="sa-ind-value">{analysis.success_indicators.critical_potential}</span>
                                        </div>
                                        <div className={`sa-indicator ${analysis.success_indicators.audience_engagement?.toLowerCase()}`}>
                                            <span className="sa-ind-label">Engagement</span>
                                            <span className="sa-ind-value">{analysis.success_indicators.audience_engagement}</span>
                                        </div>
                                    </div>
                                    <p className="sa-reasoning">{analysis.success_indicators.reasoning}</p>
                                </motion.div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default ScriptAnalysis;
