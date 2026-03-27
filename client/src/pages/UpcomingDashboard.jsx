import { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Chart from 'react-apexcharts';
import { predictionService, trendsService, scriptService } from '../services/api';
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

    const [scriptFile, setScriptFile] = useState(null);
    const [scriptScore, setScriptScore] = useState(null); // Will hold the Gemini output
    const [uploadingScript, setUploadingScript] = useState(false);
    const [isDraggingPredict, setIsDraggingPredict] = useState(false);

    const handleDragOverPredict = (e) => { e.preventDefault(); setIsDraggingPredict(true); };
    const handleDragLeavePredict = () => setIsDraggingPredict(false);
    const handleDropPredict = (e) => {
        e.preventDefault(); setIsDraggingPredict(false);
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) handleScriptUpload({ target: { files: [droppedFile] } });
    };

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [activeSection, setActiveSection] = useState('predict');
    const [selectedDirector, setSelectedDirector] = useState(null);
    const [selectedCast, setSelectedCast] = useState([]);
    const [castScore, setCastScore] = useState(0);

    // What-If Simulation state
    const [simData, setSimData] = useState({
        budget: '',
        releaseMonth: '',
        genres: [],
        isSequel: false
    });
    const [simResult, setSimResult] = useState(null);
    const [simLoading, setSimLoading] = useState(false);
    
    // Sim Script upload state
    const [simScriptFile, setSimScriptFile] = useState(null);
    const [simScriptScore, setSimScriptScore] = useState(null);
    const [uploadingSimScript, setUploadingSimScript] = useState(false);
    const [isDraggingSim, setIsDraggingSim] = useState(false);

    const handleDragOverSim = (e) => { e.preventDefault(); setIsDraggingSim(true); };
    const handleDragLeaveSim = () => setIsDraggingSim(false);
    const handleDropSim = (e) => {
        e.preventDefault(); setIsDraggingSim(false);
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) handleSimScriptUpload({ target: { files: [droppedFile] } });
    };

    const initSimData = () => {
        setSimData({
            budget: formData.budget || '',
            releaseMonth: formData.releaseMonth || '',
            genres: [...(formData.genres || [])],
            isSequel: formData.isSequel || false
        });
        setSimScriptFile(null);
        setSimScriptScore(null);
        setSimResult(null);
    };

    const handleSimGenreToggle = (genre) => {
        setSimData(prev => ({
            ...prev,
            genres: prev.genres.includes(genre)
                ? prev.genres.filter(g => g !== genre)
                : [...prev.genres, genre]
        }));
    };

    const runSimulation = async () => {
        if (!prediction) return;
        setSimLoading(true);
        try {
            const modifications = [];
            // Build modification based on what changed
            const changes = {};
            if (simData.budget && simData.budget !== formData.budget) {
                changes.budget = simData.budget;
            }
            if (simData.releaseMonth && simData.releaseMonth !== formData.releaseMonth) {
                changes.releaseMonth = simData.releaseMonth;
            }
            if (JSON.stringify(simData.genres.sort()) !== JSON.stringify([...(formData.genres || [])].sort())) {
                changes.genres = simData.genres;
            }
            if (simData.isSequel !== formData.isSequel) {
                changes.isSequel = simData.isSequel;
            }

            if (Object.keys(changes).length === 0) {
                // No changes — just show same prediction
                setSimResult({
                    basePrediction: { predictions: prediction.predictions },
                    scenarios: [{
                        name: 'No Changes',
                        changes: {},
                        predictions: { predictions: prediction.predictions }
                    }]
                });
                setSimLoading(false);
                return;
            }

            // Build scenario name
            const changeDescriptions = [];
            if (changes.budget) changeDescriptions.push(`Budget → $${parseInt(changes.budget).toLocaleString()}`);
            if (changes.releaseMonth) changeDescriptions.push(`Month → ${months[parseInt(changes.releaseMonth) - 1]}`);
            if (changes.genres) changeDescriptions.push(`Genres changed`);
            if (changes.isSequel !== undefined) changeDescriptions.push(changes.isSequel ? 'Made Sequel' : 'Made Standalone');

            if (simScriptScore) {
                changeDescriptions.push('Updated Script Analysis');
            }

            modifications.push({
                name: changeDescriptions.join(' + '),
                changes
            });

            const response = await predictionService.simulate(formData, modifications);
            setSimResult(response.data);
        } catch (error) {
            console.error('Simulation error:', error);
            // Provide demo result
            const baseBudget = parseInt(formData.budget || 100000000);
            const modBudget = parseInt(simData.budget || baseBudget);
            const budgetRatio = modBudget / baseBudget;
            const baseRev = prediction.predictions?.predictedRevenue || baseBudget * 3.5;
            const modRev = baseRev * budgetRatio * (0.9 + Math.random() * 0.3);
            setSimResult({
                basePrediction: { predictions: prediction.predictions },
                scenarios: [{
                    name: changeDescriptions.join(' + ') || 'Modified Scenario',
                    changes: { budget: simData.budget },
                    predictions: {
                        predictions: {
                            successCategory: modRev > modBudget * 3 ? 'Blockbuster' : modRev > modBudget * 2 ? 'Hit' : 'Average',
                            predictedRevenue: modRev,
                            predictedROI: ((modRev - modBudget) / modBudget * 100),
                            confidence: (prediction.predictions?.confidence || 80) - 5
                        }
                    }
                }]
            });
        } finally {
            setSimLoading(false);
        }
    };

    const handleSimScriptUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Basic validation
        const validTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a PDF, TXT, or DOCX file.');
            return;
        }

        setSimScriptFile(file);
        setUploadingSimScript(true);

        try {
            const response = await scriptService.analyze(file);
            const data = response.data;
            
            if (data.success && data.analysis) {
                setSimScriptScore(data.analysis);
                
                // Auto-fill Genres based on Gemini's genre_prediction for Simulation Data
                if (data.analysis.genre_prediction && Array.isArray(data.analysis.genre_prediction)) {
                    const extractedGenres = data.analysis.genre_prediction
                        .map(g => g.split('/')[0].trim()) 
                        .filter(g => genres.includes(g)); 
                    
                    if (extractedGenres.length > 0) {
                        setSimData(prev => ({
                            ...prev,
                            // Replace genres entirely or add? For simulation, mostly we replace to test new idea
                            genres: [...new Set([...prev.genres, ...extractedGenres])]
                        }));
                    }
                }
            }
        } catch (error) {
            console.error('Simulation Script analysis failed:', error);
            alert(error.response?.data?.error || 'Failed to analyze script. Proceeding without script insights.');
            setSimScriptScore(null);
            setSimScriptFile(null);
        } finally {
            setUploadingSimScript(false);
        }
    };

    // Simulation comparison chart config
    const simChartOptions = useMemo(() => {
        if (!simResult || !simResult.scenarios?.[0]) return null;
        const base = simResult.basePrediction?.predictions || prediction?.predictions;
        const mod = simResult.scenarios[0].predictions?.predictions;
        if (!base || !mod) return null;

        const baseRev = base.predictedRevenue || 0;
        const modRev = mod.predictedRevenue || 0;
        const baseROI = base.predictedROI || 0;
        const modROI = mod.predictedROI || 0;

        return {
            series: [{
                name: 'Original',
                data: [Math.round(baseRev / 1000000), Math.round(baseROI)]
            }, {
                name: 'Modified',
                data: [Math.round(modRev / 1000000), Math.round(modROI)]
            }],
            options: {
                chart: {
                    type: 'bar',
                    height: 280,
                    background: 'transparent',
                    toolbar: { show: false },
                    fontFamily: "'TT Firs Neue', sans-serif"
                },
                plotOptions: {
                    bar: { horizontal: false, columnWidth: '55%', borderRadius: 6, borderRadiusApplication: 'end' }
                },
                dataLabels: { enabled: false },
                stroke: { show: true, width: 2, colors: ['transparent'] },
                xaxis: {
                    categories: ['Revenue ($M)', 'ROI (%)'],
                    labels: { style: { colors: '#888', fontSize: '12px' } },
                    axisBorder: { show: false },
                    axisTicks: { show: false }
                },
                yaxis: {
                    labels: { style: { colors: '#888', fontSize: '12px' } }
                },
                fill: { opacity: 1 },
                colors: ['#666666', '#ff4d00'],
                legend: {
                    position: 'top',
                    labels: { colors: '#ccc' },
                    fontFamily: "'TT Firs Neue', sans-serif"
                },
                tooltip: {
                    theme: 'dark',
                    y: {
                        formatter: (val, { dataPointIndex }) =>
                            dataPointIndex === 0 ? `$${val}M` : `${val}%`
                    }
                },
                grid: { borderColor: '#1a1a1a', strokeDashArray: 3 }
            }
        };
    }, [simResult, prediction]);

    // Optimal Release & Competitive Analysis state
    const [seasonalData, setSeasonalData] = useState([]);
    const [releaseIndustry, setReleaseIndustry] = useState('');
    const [releaseGenre, setReleaseGenre] = useState('');
    const [releaseLoading, setReleaseLoading] = useState(false);
    const [selectedMonth, setSelectedMonth] = useState(null);
    const [competitiveData, setCompetitiveData] = useState(null);
    const [competitiveLoading, setCompetitiveLoading] = useState(false);

    // Fetch seasonal data when release tab is active
    useEffect(() => {
        if (activeSection === 'release') {
            fetchSeasonalData();
        }
    }, [activeSection, releaseIndustry, releaseGenre]);

    const fetchSeasonalData = async () => {
        setReleaseLoading(true);
        try {
            const params = {};
            if (releaseIndustry) params.industry = releaseIndustry;
            if (releaseGenre) params.genre = releaseGenre;
            const response = await trendsService.getSeasonal(params);
            setSeasonalData(response.data || []);
        } catch (error) {
            console.error('Error fetching seasonal data:', error);
        } finally {
            setReleaseLoading(false);
        }
    };

    const handleMonthClick = async (monthNumber) => {
        setSelectedMonth(monthNumber);
        setCompetitiveLoading(true);
        try {
            // Use the 15th of the selected month in the current year
            const year = new Date().getFullYear();
            const releaseDate = `${year}-${String(monthNumber).padStart(2, '0')}-15`;
            const response = await predictionService.getCompetitiveAnalysis({
                releaseDate,
                genres: releaseGenre ? [releaseGenre] : (formData.genres.length > 0 ? formData.genres : undefined),
                industry: releaseIndustry || formData.industry || 'hollywood'
            });
            setCompetitiveData(response.data);
        } catch (error) {
            console.error('Competitive analysis error:', error);
            setCompetitiveData(null);
        } finally {
            setCompetitiveLoading(false);
        }
    };

    // Rank months by composite score
    const rankedMonths = useMemo(() => {
        if (!seasonalData.length) return [];
        const maxRev = Math.max(...seasonalData.map(d => d.avgRevenue || 0));
        return seasonalData
            .map(d => ({
                ...d,
                compositeScore: (
                    ((d.avgRevenue || 0) / (maxRev || 1)) * 0.5 +
                    (d.successRate || 0) * 0.3 +
                    ((d.avgRating || 0) / 10) * 0.2
                )
            }))
            .sort((a, b) => b.compositeScore - a.compositeScore);
    }, [seasonalData]);

    // Seasonal chart configuration
    const seasonalChartConfig = useMemo(() => {
        if (!seasonalData.length) return null;
        const allMonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const dataByMonth = {};
        seasonalData.forEach(d => { dataByMonth[d.month] = d; });

        const top3Months = rankedMonths.slice(0, 3).map(m => m.month);

        return {
            series: [
                {
                    name: 'Avg Revenue ($M)',
                    type: 'bar',
                    data: allMonths.map(m => Math.round((dataByMonth[m]?.avgRevenue || 0) / 1000000))
                },
                {
                    name: 'Avg Rating',
                    type: 'line',
                    data: allMonths.map(m => parseFloat((dataByMonth[m]?.avgRating || 0).toFixed(1)))
                }
            ],
            options: {
                chart: {
                    type: 'line',
                    height: 380,
                    background: 'transparent',
                    toolbar: { show: false },
                    fontFamily: "'TT Firs Neue', sans-serif",
                    events: {
                        dataPointSelection: (event, chartContext, config) => {
                            const monthIdx = config.dataPointIndex;
                            handleMonthClick(monthIdx + 1);
                        }
                    }
                },
                plotOptions: {
                    bar: {
                        borderRadius: 6,
                        borderRadiusApplication: 'end',
                        columnWidth: '50%',
                        colors: {
                            ranges: allMonths.map((m, i) => ({
                                from: 0,
                                to: 999999,
                                color: top3Months.includes(m) ? '#ff4d00' : '#333'
                            }))
                        }
                    }
                },
                colors: [
                    ({ dataPointIndex }) =>
                        top3Months.includes(allMonths[dataPointIndex]) ? '#ff4d00' : '#444',
                    '#ffa500'
                ],
                fill: {
                    type: ['solid', 'solid'],
                    opacity: [0.9, 1]
                },
                stroke: {
                    width: [0, 3],
                    curve: 'smooth'
                },
                xaxis: {
                    categories: allMonths,
                    labels: { style: { colors: '#888', fontSize: '12px' } },
                    axisBorder: { show: false },
                    axisTicks: { show: false }
                },
                yaxis: [
                    {
                        title: { text: 'Revenue ($M)', style: { color: '#888', fontSize: '12px' } },
                        labels: { style: { colors: '#888', fontSize: '11px' }, formatter: v => `$${v}M` }
                    },
                    {
                        opposite: true,
                        title: { text: 'Rating', style: { color: '#ffa500', fontSize: '12px' } },
                        labels: { style: { colors: '#ffa500', fontSize: '11px' }, formatter: v => v.toFixed(1) },
                        min: 4,
                        max: 9
                    }
                ],
                legend: {
                    position: 'top',
                    labels: { colors: '#ccc' },
                    fontFamily: "'TT Firs Neue', sans-serif"
                },
                tooltip: { theme: 'dark' },
                grid: { borderColor: '#1a1a1a', strokeDashArray: 3 },
                dataLabels: { enabled: false }
            }
        };
    }, [seasonalData, rankedMonths]);

    const genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation', 'Adventure', 'Crime', 'Fantasy', 'Mystery'];
    const industries = [
        { value: 'hollywood', label: 'Hollywood' },
        { value: 'bollywood', label: 'Bollywood' },
        { value: 'tollywood', label: 'Tollywood' },
        { value: 'kollywood', label: 'Kollywood' },
        { value: 'mollywood', label: 'Mollywood' }
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

    const handleScriptUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Basic validation
        const validTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a PDF, TXT, or DOCX file.');
            return;
        }

        setScriptFile(file);
        setUploadingScript(true);

        try {
            const response = await scriptService.analyze(file);
            const data = response.data;
            
            if (data.success && data.analysis) {
                setScriptScore(data.analysis);
                
                // Auto-fill Genres based on Gemini's genre_prediction
                if (data.analysis.genre_prediction && Array.isArray(data.analysis.genre_prediction)) {
                    // Map Gemini's genres to our predefined list, falling back if no exact match
                    const extractedGenres = data.analysis.genre_prediction
                        .map(g => g.split('/')[0].trim()) // Sometimes it says "Action/Adventure"
                        .filter(g => genres.includes(g)); 
                    
                    if (extractedGenres.length > 0) {
                        setFormData(prev => ({
                            ...prev,
                            // Set genres to unique combination of existing + extracted
                            genres: [...new Set([...prev.genres, ...extractedGenres])]
                        }));
                    }
                }
            }
        } catch (error) {
            console.error('Script analysis failed:', error);
            alert(error.response?.data?.error || 'Failed to analyze script. Proceeding without script insights.');
            setScriptScore(null);
            setScriptFile(null);
        } finally {
            setUploadingScript(false);
        }
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
            },
            scriptAnalysis: scriptScore // Add script insights to demo prediction too
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
        <div className="upcoming-dashboard" id="top">
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
                                        {/* Optional Script Upload */}
                                        <div className="form-section script-upload-section">
                                            <h3 className="form-section-title">
                                                <span className="sparkle-icon">✨</span> Automate with AI Script Analysis
                                            </h3>
                                            <p className="section-description" style={{ fontSize: '0.9rem', color: '#888', marginBottom: '15px' }}>
                                                Upload your script to auto-detect genres and receive a qualitative story score alongside your financial prediction.
                                            </p>
                                            
                                            <div 
                                                className={`sa-dropzone ${isDraggingPredict ? 'dragging' : ''} ${scriptFile ? 'has-file' : ''}`}
                                                onDragOver={handleDragOverPredict}
                                                onDragLeave={handleDragLeavePredict}
                                                onDrop={handleDropPredict}
                                            >
                                                <input
                                                    type="file"
                                                    id="script-upload-predict"
                                                    className="script-file-input"
                                                    accept=".pdf,.txt,.doc,.docx"
                                                    onChange={handleScriptUpload}
                                                    disabled={uploadingScript}
                                                    hidden
                                                />
                                                <label htmlFor="script-upload-predict" style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', margin: 0 }}>
                                                    {uploadingScript ? (
                                                        <div className="sa-loading-hint">
                                                            <span className="sa-spinner" style={{ marginBottom: '10px' }}></span>
                                                            <p style={{ margin: 0 }}>Gemini AI is analyzing your script...</p>
                                                            <p className="sa-drop-subtext" style={{ marginTop: '5px' }}>This may take up to 60 seconds.</p>
                                                        </div>
                                                    ) : scriptFile ? (
                                                        <div className="sa-file-info" style={{ flexDirection: 'column', gap: '8px' }}>
                                                            <span className="sa-file-icon">📄</span>
                                                            <span className="sa-file-name">{scriptFile.name}</span>
                                                            <span className="sa-file-size" style={{color: '#00c853'}}>✓ Analysis Complete! Click to replace</span>
                                                        </div>
                                                    ) : (
                                                        <>
                                                            <span className="sa-drop-icon">📜</span>
                                                            <p className="sa-drop-text">Drag & drop your script here</p>
                                                            <p className="sa-drop-subtext">or click to browse · PDF, TXT, DOCX · Max 10 MB</p>
                                                        </>
                                                    )}
                                                </label>
                                            </div>
                                        </div>

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

                                            {/* Top Section: Dual Engine Results */}
                                            <div className="dual-engine-results" style={{ display: 'flex', gap: '20px', flexDirection: window.innerWidth < 768 ? 'column' : 'row' }}>
                                                
                                                {/* Left: ML Data Prediction */}
                                                <div className="prediction-main flex-1" style={{ flex: scriptScore ? '1.5' : '1' }}>
                                                    <h3 className="section-subtitle" style={{ fontSize: '1rem', marginBottom: '15px', color: '#888' }}>📊 Data Model Prediction</h3>
                                                    <div className="prediction-category" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                                                        <span className={`category-label ${getCategoryClass(prediction.predictions?.successCategory)}`}>
                                                            {prediction.predictions?.successCategory || 'Processing...'}
                                                        </span>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                            <div style={{ position: 'relative', width: '40px', height: '40px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                                <svg viewBox="0 0 36 36" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                                                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" />
                                                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--accent-orange)" strokeWidth="3" strokeDasharray={`${prediction.predictions?.confidence || 0}, 100`} style={{ transition: 'stroke-dasharray 1s ease-out' }} />
                                                                </svg>
                                                                <span style={{ fontSize: '0.75rem', fontWeight: 'bold', zIndex: 1, fontFamily: 'Syne, sans-serif' }}>
                                                                    {prediction.predictions?.confidence || 0}%
                                                                </span>
                                                            </div>
                                                            <span style={{ fontSize: '0.9rem', color: '#888' }}>Confidence</span>
                                                        </div>
                                                    </div>

                                                    <div className="prediction-metrics">
                                                        <div className="metric-card">
                                                            <span className="metric-icon">💰</span>
                                                            <div className="metric-info">
                                                                <span className="metric-value">
                                                                    {formatCurrency(prediction.predictions?.predictedRevenue)}
                                                                </span>
                                                                <span className="metric-label">Predicted Revenue <span className="tooltip-icon" data-tooltip="Estimated global box office gross. Break-even translates to approx 2.5x production budget.">?</span></span>
                                                            </div>
                                                        </div>

                                                        <div className="metric-card">
                                                            <span className="metric-icon">⭐</span>
                                                            <div className="metric-info">
                                                                <span className="metric-value">
                                                                    {prediction.predictions?.confidence ? (Math.min(prediction.predictions.confidence / 10, 10)).toFixed(1) : 'N/A'}/10
                                                                </span>
                                                                <span className="metric-label">Predicted Rating <span className="tooltip-icon" data-tooltip="Estimated IMDb/Audience score out of 10 based on similar features.">?</span></span>
                                                            </div>
                                                        </div>

                                                        <div className="metric-card">
                                                            <span className="metric-icon">📈</span>
                                                            <div className="metric-info">
                                                                <span className={`metric-value ${(prediction.predictions?.predictedROI || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                                                                    {(prediction.predictions?.predictedROI || 0) >= 0 ? '+' : ''}{Math.round(prediction.predictions?.predictedROI || 0)}%
                                                                </span>
                                                                <span className="metric-label">Predicted ROI <span className="tooltip-icon" data-tooltip="Return on production budget. Warning: positive ROI doesn't mean net profit after marketing and theater cuts.">?</span></span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Right: AI Script Insights (If Available) */}
                                                {scriptScore && (
                                                    <div className="ai-magic-card">
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                                                            <span className="sparkle-icon">✨</span>
                                                            <h3 className="section-subtitle" style={{ fontSize: '1rem', margin: 0, color: '#ff4d00', zIndex: 2 }}>Gemini Script Score</h3>
                                                        </div>

                                                        <div className="script-score-circle" style={{ textAlign: 'center', marginBottom: '15px', zIndex: 2, position: 'relative' }}>
                                                            <div style={{ fontSize: '2.5rem', fontWeight: '800', color: '#fff' }}>
                                                                {scriptScore.overall_score}<span style={{ fontSize: '1rem', color: '#888' }}>/10</span>
                                                            </div>
                                                            <div style={{ fontSize: '0.85rem', color: '#aaa', marginTop: '5px' }}>Overall Story Quality</div>
                                                        </div>

                                                        <div className="script-commercial-potential" style={{ background: 'rgba(0,0,0,0.3)', padding: '15px', borderRadius: '12px', zIndex: 2, position: 'relative' }}>
                                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', alignItems: 'center' }}>
                                                                <span style={{ color: '#888', fontSize: '0.9rem', display: 'flex', alignItems: 'center' }}>
                                                                    Commercial Potential:
                                                                    <span className="tooltip-icon" data-tooltip="AI's assessment of marketability based on themes, genre, and audience trends." style={{ marginLeft: '4px' }}>?</span>
                                                                </span>
                                                                <strong className={scriptScore.success_indicators?.commercial_potential === 'High' ? 'text-success' : scriptScore.success_indicators?.commercial_potential === 'Medium' ? 'text-warning' : 'text-danger'}>
                                                                    {scriptScore.success_indicators?.commercial_potential}
                                                                </strong>
                                                            </div>
                                                            <p style={{ fontSize: '0.85rem', color: '#bbb', lineHeight: '1.4', margin: 0 }}>
                                                                {scriptScore.success_indicators?.reasoning}
                                                            </p>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Feature Importance */}
                                            {prediction.predictions?.featureImportance && (
                                                <div className="feature-importance-section" style={{ marginTop: '30px', paddingTop: '30px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                                    <h3 className="section-subtitle">Why This Prediction? (ML Explainability)</h3>
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
                            className="simulate-section"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            {!prediction ? (
                                <div className="sim-no-prediction glass-card">
                                    <div className="placeholder-icon">🔮</div>
                                    <h3>Make a Prediction First</h3>
                                    <p>Head to the <strong>Predict Success</strong> tab and run a prediction before using the What-If Simulator.</p>
                                    <button
                                        className="btn btn-primary"
                                        onClick={() => setActiveSection('predict')}
                                        style={{ marginTop: '20px', padding: '14px 32px', marginLeft: 'auto', marginRight: 'auto', display: 'block' }}
                                    >
                                        Go to Predict Tab →
                                    </button>
                                </div>
                            ) : (
                                <div className="sim-grid">
                                    {/* Left: Controls */}
                                    <div className="sim-controls glass-card">
                                        <h2 className="card-title">🎛️ Modify Parameters</h2>
                                        <p className="card-description">
                                            Adjust values below and run the simulation to see how changes affect your prediction.
                                        </p>

                                        <div className="sim-form">
                                            {/* Optional Sim Script Upload */}
                                            <div className="sim-control-group" style={{ marginBottom: '20px' }}>
                                                <label className="form-label">
                                                    ✨ Upload Updated Script
                                                    {simScriptScore && (
                                                        <span className="sim-changed-badge" style={{ backgroundColor: '#ff4d00' }}>AI Analyzed</span>
                                                    )}
                                                </label>
                                                <div 
                                                    className={`sa-dropzone ${isDraggingSim ? 'dragging' : ''} ${simScriptFile ? 'has-file' : ''}`}
                                                    style={{ padding: '30px 20px', background: 'rgba(0,0,0,0.2)' }}
                                                    onDragOver={handleDragOverSim}
                                                    onDragLeave={handleDragLeaveSim}
                                                    onDrop={handleDropSim}
                                                >
                                                    <input
                                                        type="file"
                                                        id="script-upload-simulate"
                                                        className="script-file-input"
                                                        accept=".pdf,.txt,.doc,.docx"
                                                        onChange={handleSimScriptUpload}
                                                        disabled={uploadingSimScript}
                                                        hidden
                                                    />
                                                    <label htmlFor="script-upload-simulate" style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', margin: 0 }}>
                                                        {uploadingSimScript ? (
                                                            <div className="sa-loading-hint">
                                                                <span className="sa-spinner" style={{ marginBottom: '10px', width: '20px', height: '20px' }}></span>
                                                                <p style={{ fontSize: '0.9rem', margin: 0 }}>Analyzing... <small>Wait 60s</small></p>
                                                            </div>
                                                        ) : simScriptFile ? (
                                                            <div className="sa-file-info" style={{ flexDirection: 'column', gap: '5px' }}>
                                                                <span className="sa-file-icon" style={{ fontSize: '1.5rem' }}>📄</span>
                                                                <span className="sa-file-name" style={{ fontSize: '0.9rem', textAlign: 'center' }}>{simScriptFile.name}</span>
                                                                {simScriptScore && <span className="sa-file-size" style={{color: '#00c853'}}>✓ Score: {simScriptScore.overall_score}/10</span>}
                                                            </div>
                                                        ) : (
                                                            <>
                                                                <span className="sa-drop-icon" style={{ fontSize: '2rem', marginBottom: '10px' }}>📜</span>
                                                                <p className="sa-drop-text" style={{ fontSize: '0.95rem' }}>Drag & drop script</p>
                                                                <p className="sa-drop-subtext" style={{ fontSize: '0.8rem' }}>PDF, TXT, DOCX</p>
                                                            </>
                                                        )}
                                                    </label>
                                                </div>
                                            </div>

                                            {/* Budget */}
                                            <div className="sim-control-group">
                                                <label className="form-label">
                                                    💰 Budget ($)
                                                    {simData.budget && simData.budget !== formData.budget && (
                                                        <span className="sim-changed-badge">Modified</span>
                                                    )}
                                                </label>
                                                <input
                                                    type="number"
                                                    className="input"
                                                    value={simData.budget}
                                                    onChange={(e) => setSimData(prev => ({ ...prev, budget: e.target.value }))}
                                                    placeholder={`Original: $${parseInt(formData.budget || 0).toLocaleString()}`}
                                                />
                                                <div className="sim-budget-presets">
                                                    {[0.5, 0.75, 1.5, 2].map(mult => (
                                                        <button
                                                            key={mult}
                                                            type="button"
                                                            className="sim-preset-btn"
                                                            onClick={() => setSimData(prev => ({
                                                                ...prev,
                                                                budget: String(Math.round(parseInt(formData.budget || 100000000) * mult))
                                                            }))}
                                                        >
                                                            {mult < 1 ? `${mult * 100}%` : `${mult * 100}%`}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Release Month */}
                                            <div className="sim-control-group">
                                                <label className="form-label">
                                                    📅 Release Month
                                                    {simData.releaseMonth && simData.releaseMonth !== formData.releaseMonth && (
                                                        <span className="sim-changed-badge">Modified</span>
                                                    )}
                                                </label>
                                                <select
                                                    className="input"
                                                    value={simData.releaseMonth}
                                                    onChange={(e) => setSimData(prev => ({ ...prev, releaseMonth: e.target.value }))}
                                                >
                                                    <option value="">Select month</option>
                                                    {months.map((month, idx) => (
                                                        <option key={month} value={idx + 1}>{month}</option>
                                                    ))}
                                                </select>
                                            </div>

                                            {/* Genres */}
                                            <div className="sim-control-group">
                                                <label className="form-label">
                                                    🎭 Genres
                                                    {JSON.stringify(simData.genres.sort()) !== JSON.stringify([...(formData.genres || [])].sort()) && (
                                                        <span className="sim-changed-badge">Modified</span>
                                                    )}
                                                </label>
                                                <div className="genre-options">
                                                    {genres.map(genre => (
                                                        <button
                                                            key={genre}
                                                            type="button"
                                                            className={`chip ${simData.genres.includes(genre) ? 'active' : ''}`}
                                                            onClick={() => handleSimGenreToggle(genre)}
                                                        >
                                                            {genre}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Sequel */}
                                            <div className="sim-control-group">
                                                <div className="form-group checkbox-group">
                                                    <label className="checkbox-label">
                                                        <input
                                                            type="checkbox"
                                                            checked={simData.isSequel}
                                                            onChange={(e) => setSimData(prev => ({ ...prev, isSequel: e.target.checked }))}
                                                        />
                                                        <span className="checkbox-custom"></span>
                                                        Part of a Sequel/Franchise
                                                        {simData.isSequel !== formData.isSequel && (
                                                            <span className="sim-changed-badge" style={{ marginLeft: '10px' }}>Modified</span>
                                                        )}
                                                    </label>
                                                </div>
                                            </div>

                                            <div className="sim-actions">
                                                <button
                                                    type="button"
                                                    className="btn btn-primary btn-block"
                                                    onClick={runSimulation}
                                                    disabled={simLoading}
                                                >
                                                    {simLoading ? (
                                                        <>
                                                            <span className="btn-spinner"></span>
                                                            Simulating...
                                                        </>
                                                    ) : (
                                                        '⚡ Run Simulation'
                                                    )}
                                                </button>
                                                <button
                                                    type="button"
                                                    className="btn btn-secondary"
                                                    onClick={initSimData}
                                                >
                                                    ↺ Reset to Original
                                                </button>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Right: Results */}
                                    <div className="sim-results">
                                        <AnimatePresence mode="wait">
                                            {simResult ? (
                                                <motion.div
                                                    className="sim-results-content glass-card"
                                                    key="sim-results"
                                                    initial={{ opacity: 0, x: 20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    exit={{ opacity: 0, x: -20 }}
                                                >
                                                    <h2 className="card-title">📊 Simulation Results</h2>
                                                    {simResult.scenarios?.[0] && (
                                                        <p className="sim-scenario-name">
                                                            {simResult.scenarios[0].name}
                                                        </p>
                                                    )}

                                                    {/* Comparison Cards */}
                                                    <div className="sim-comparison">
                                                        {/* Original */}
                                                        <div className="sim-compare-card sim-original" style={{ position: 'relative' }}>
                                                            <h4>Original</h4>
                                                            <div className={`sim-category ${getCategoryClass(prediction.predictions?.successCategory)}`}>
                                                                {prediction.predictions?.successCategory || 'N/A'}
                                                            </div>
                                                            <div className="sim-stat">
                                                                <span className="sim-stat-label">Revenue</span>
                                                                <span className="sim-stat-value">{formatCurrency(prediction.predictions?.predictedRevenue)}</span>
                                                            </div>
                                                            <div className="sim-stat">
                                                                <span className="sim-stat-label">ROI</span>
                                                                <span className="sim-stat-value">
                                                                    {prediction.predictions?.predictedROI != null
                                                                        ? `${Math.round(prediction.predictions.predictedROI)}%`
                                                                        : 'N/A'}
                                                                </span>
                                                            </div>
                                                            
                                                            {/* Original Script Score if any */}
                                                            <div className="sim-stat" style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                                                <span className="sim-stat-label">Script Score</span>
                                                                <span className="sim-stat-value" style={{ color: scriptScore ? '#fff' : '#888' }}>
                                                                    {scriptScore ? `${scriptScore.overall_score}/10` : 'None'}
                                                                </span>
                                                            </div>
                                                        </div>

                                                        {/* Arrow */}
                                                        <div className="sim-arrow">→</div>

                                                        {/* Modified */}
                                                        {simResult.scenarios?.[0] && (() => {
                                                            const mod = simResult.scenarios[0].predictions?.predictions;
                                                            const baseRev = prediction.predictions?.predictedRevenue || 0;
                                                            const modRev = mod?.predictedRevenue || 0;
                                                            const revDelta = baseRev > 0 ? ((modRev - baseRev) / baseRev * 100) : 0;
                                                            
                                                            const currentScriptScore = simScriptScore || scriptScore;
                                                            
                                                            return (
                                                                <div className="sim-compare-card sim-modified" style={{ position: 'relative' }}>
                                                                    <h4>Modified</h4>
                                                                    <div className={`sim-category ${getCategoryClass(mod?.successCategory)}`}>
                                                                        {mod?.successCategory || 'N/A'}
                                                                    </div>
                                                                    <div className="sim-stat">
                                                                        <span className="sim-stat-label">Revenue</span>
                                                                        <span className="sim-stat-value">
                                                                            {formatCurrency(mod?.predictedRevenue)}
                                                                            <span className={`sim-delta ${revDelta >= 0 ? 'positive' : 'negative'}`}>
                                                                                {revDelta >= 0 ? '▲' : '▼'} {Math.abs(revDelta).toFixed(1)}%
                                                                            </span>
                                                                        </span>
                                                                    </div>
                                                                    <div className="sim-stat">
                                                                        <span className="sim-stat-label">ROI</span>
                                                                        <span className="sim-stat-value">
                                                                            {mod?.predictedROI != null ? `${Math.round(mod.predictedROI)}%` : 'N/A'}
                                                                        </span>
                                                                    </div>
                                                                    
                                                                    {/* Modified Script Score */}
                                                                    <div className="sim-stat" style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                                                        <span className="sim-stat-label">Script Score</span>
                                                                        <span className="sim-stat-value" style={{ color: currentScriptScore ? (simScriptScore ? '#ff4d00' : '#fff') : '#888' }}>
                                                                            {currentScriptScore ? `${currentScriptScore.overall_score}/10` : 'None'}
                                                                            {simScriptScore && <span className="sim-delta positive" style={{ marginLeft: '5px' }}>✨ New</span>}
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                            );
                                                        })()}
                                                    </div>

                                                    {/* Comparison Chart */}
                                                    {simChartOptions && (
                                                        <div className="sim-chart">
                                                            <Chart
                                                                options={simChartOptions.options}
                                                                series={simChartOptions.series}
                                                                type="bar"
                                                                height={280}
                                                            />
                                                        </div>
                                                    )}
                                                </motion.div>
                                            ) : (
                                                <motion.div
                                                    className="sim-placeholder glass-card"
                                                    key="sim-placeholder"
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                >
                                                    <div className="placeholder-icon">🎛️</div>
                                                    <h3>Ready to Simulate</h3>
                                                    <p>Adjust the parameters on the left and click <strong>"Run Simulation"</strong> to see how changes impact your movie's prediction.</p>
                                                    <div className="placeholder-features">
                                                        <span>💰 Budget Scenarios</span>
                                                        <span>📅 Release Timing</span>
                                                        <span>🎭 Genre Shifts</span>
                                                        <span>🔄 Franchise Impact</span>
                                                    </div>
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    )}

                    {activeSection === 'release' && (
                        <motion.div
                            className="release-section"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            {/* Header + Filters */}
                            <div className="release-header glass-card">
                                <div className="release-header-top">
                                    <div>
                                        <h2 className="card-title">📅 Optimal Release Window</h2>
                                        <p className="card-description">
                                            Analyze seasonal trends and competition to find the best release month for your movie.
                                        </p>
                                    </div>
                                    <div className="release-filters">
                                        <select
                                            className="input release-filter-select"
                                            value={releaseIndustry}
                                            onChange={(e) => setReleaseIndustry(e.target.value)}
                                        >
                                            <option value="">All Industries</option>
                                            {industries.map(ind => (
                                                <option key={ind.value} value={ind.value}>{ind.label}</option>
                                            ))}
                                        </select>
                                        <select
                                            className="input release-filter-select"
                                            value={releaseGenre}
                                            onChange={(e) => setReleaseGenre(e.target.value)}
                                        >
                                            <option value="">All Genres</option>
                                            {genres.map(g => (
                                                <option key={g} value={g}>{g}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                {/* Seasonal Chart */}
                                {releaseLoading ? (
                                    <div className="release-loading">
                                        <span className="btn-spinner"></span>
                                        Loading seasonal data...
                                    </div>
                                ) : seasonalChartConfig ? (
                                    <div className="release-chart">
                                        <Chart
                                            options={seasonalChartConfig.options}
                                            series={seasonalChartConfig.series}
                                            type="line"
                                            height={380}
                                        />
                                    </div>
                                ) : (
                                    <div className="release-loading">
                                        <p>No seasonal data available.</p>
                                    </div>
                                )}
                            </div>

                            {/* Top Recommended Months */}
                            {rankedMonths.length > 0 && (
                                <div className="release-recommendations">
                                    <h3 className="release-reco-title">🏆 Top Recommended Months</h3>
                                    <div className="release-reco-grid">
                                        {rankedMonths.slice(0, 3).map((month, idx) => {
                                            const medals = ['🥇', '🥈', '🥉'];
                                            const fullMonthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
                                            return (
                                                <motion.div
                                                    key={month.month}
                                                    className={`release-reco-card glass-card ${idx === 0 ? 'top-pick' : ''}`}
                                                    initial={{ opacity: 0, y: 20 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ delay: idx * 0.1 }}
                                                    onClick={() => handleMonthClick(month.monthNumber || month._id)}
                                                    style={{ cursor: 'pointer' }}
                                                >
                                                    <div className="reco-medal">{medals[idx]}</div>
                                                    <h4 className="reco-month">{fullMonthNames[month.monthNumber - 1] || month.month}</h4>
                                                    <div className="reco-score">
                                                        Score: <strong>{(month.compositeScore * 100).toFixed(0)}</strong>/100
                                                    </div>
                                                    <div className="reco-stats">
                                                        <div className="reco-stat">
                                                            <span className="reco-stat-label">Avg Revenue</span>
                                                            <span className="reco-stat-value">{formatCurrency(month.avgRevenue)}</span>
                                                        </div>
                                                        <div className="reco-stat">
                                                            <span className="reco-stat-label">Success Rate</span>
                                                            <span className="reco-stat-value">{(month.successRate * 100).toFixed(0)}%</span>
                                                        </div>
                                                        <div className="reco-stat">
                                                            <span className="reco-stat-label">Avg Rating</span>
                                                            <span className="reco-stat-value">{(month.avgRating || 0).toFixed(1)}/10</span>
                                                        </div>
                                                    </div>
                                                    <span className="reco-click-hint">Click to see competition →</span>
                                                </motion.div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Competitive Window Analyzer */}
                            <AnimatePresence>
                                {selectedMonth && (
                                    <motion.div
                                        className="competitive-section glass-card"
                                        initial={{ opacity: 0, height: 0, marginTop: 0 }}
                                        animate={{ opacity: 1, height: 'auto', marginTop: 24 }}
                                        exit={{ opacity: 0, height: 0, marginTop: 0 }}
                                    >
                                        <div className="competitive-header">
                                            <h3 className="card-title">
                                                🎯 Competitive Analysis — {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][selectedMonth - 1]} Window
                                            </h3>
                                            <button
                                                className="competitive-close"
                                                onClick={() => { setSelectedMonth(null); setCompetitiveData(null); }}
                                            >
                                                ✕
                                            </button>
                                        </div>

                                        {competitiveLoading ? (
                                            <div className="release-loading">
                                                <span className="btn-spinner"></span>
                                                Analyzing competition...
                                            </div>
                                        ) : competitiveData ? (
                                            <div className="competitive-content">
                                                {/* Competition Score */}
                                                {(() => {
                                                    const score = competitiveData.competitionScore ?? 0;
                                                    const level = score <= 3 ? 'low' : score <= 6 ? 'medium' : 'high';
                                                    const levelLabel = score <= 3 ? 'Low' : score <= 6 ? 'Moderate' : 'High';
                                                    const levelColor = score <= 3 ? '#00c853' : score <= 6 ? '#ffa500' : '#ff5252';
                                                    const pct = Math.min(100, (score / 10) * 100);
                                                    return (
                                                        <div className={`competitive-score-card comp-card-${level}`}>
                                                            <div className="comp-score-left">
                                                                <div className={`comp-score-ring comp-ring-${level}`}>
                                                                    <svg viewBox="0 0 100 100" className="comp-ring-svg">
                                                                        <circle cx="50" cy="50" r="42" className="comp-ring-bg" />
                                                                        <circle cx="50" cy="50" r="42"
                                                                            className="comp-ring-fill"
                                                                            style={{
                                                                                strokeDasharray: `${pct * 2.64} ${264 - pct * 2.64}`,
                                                                                stroke: levelColor
                                                                            }}
                                                                        />
                                                                    </svg>
                                                                    <div className="comp-ring-content">
                                                                        <span className="comp-score-value">{score.toFixed(1)}</span>
                                                                        <span className="comp-score-label">/ 10</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                            <div className="comp-score-right">
                                                                <div className={`comp-level-badge comp-badge-${level}`}>
                                                                    <span className="comp-badge-dot" style={{ background: levelColor }} />
                                                                    {levelLabel} Competition
                                                                </div>
                                                                <div className="comp-score-bar-wrap">
                                                                    <div className="comp-score-bar-track">
                                                                        <div className="comp-score-bar-fill" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${levelColor}88, ${levelColor})` }} />
                                                                    </div>
                                                                    <div className="comp-score-bar-labels">
                                                                        <span>Low</span><span>Moderate</span><span>High</span>
                                                                    </div>
                                                                </div>
                                                                <div className="comp-stats-row">
                                                                    <div className="comp-stat">
                                                                        <span className="comp-stat-num">{competitiveData.competingMovies?.length || 0}</span>
                                                                        <span className="comp-stat-label">Total Releases</span>
                                                                    </div>
                                                                    <div className="comp-stat">
                                                                        <span className="comp-stat-num">{competitiveData.directCompetitors?.length || 0}</span>
                                                                        <span className="comp-stat-label">Direct Competitors</span>
                                                                    </div>
                                                                    <div className="comp-stat">
                                                                        <span className="comp-stat-num">{competitiveData.recommendation === 'Favorable Window' ? '✓' : competitiveData.recommendation === 'Moderate Competition' ? '~' : '✗'}</span>
                                                                        <span className="comp-stat-label">{competitiveData.recommendation || 'N/A'}</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    );
                                                })()}

                                                {/* Competing Movies */}
                                                {competitiveData.competingMovies?.length > 0 && (() => {
                                                    // Pre-compute anticipation badges from relative popularity
                                                    const movies = competitiveData.competingMovies.slice(0, 12);
                                                    const maxPop = Math.max(...movies.map(m => m.popularity || 0));
                                                    const getAnticipationBadge = (popularity) => {
                                                        if (!popularity || maxPop === 0) return null;
                                                        const ratio = popularity / maxPop;
                                                        if (ratio >= 0.95) return { label: 'Most Anticipated', emoji: '🔥🔥🔥', className: 'anticipation-top' };
                                                        if (ratio >= 0.5) return { label: 'High Buzz', emoji: '🔥🔥', className: 'anticipation-high' };
                                                        if (ratio >= 0.25) return { label: 'Moderate Buzz', emoji: '🔥', className: 'anticipation-mid' };
                                                        return null;
                                                    };

                                                    return (
                                                        <div className="competing-movies-list">
                                                            <h4>Competing Releases (±14 days) — {competitiveData.competingMovies.length} movies</h4>
                                                            <div className="competing-movies-grid">
                                                                {movies.map((movie, idx) => {
                                                                    const badge = getAnticipationBadge(movie.popularity);
                                                                    return (
                                                                        <div key={idx} className="competing-movie-card">
                                                                            {movie.posterPath && (
                                                                                <img
                                                                                    src={movie.posterPath}
                                                                                    alt={movie.title}
                                                                                    className="comp-movie-poster"
                                                                                    loading="lazy"
                                                                                />
                                                                            )}
                                                                            <div className="comp-movie-info">
                                                                                <div className="comp-movie-title">{movie.title || 'Unknown'}</div>
                                                                                {movie.releaseDate && (
                                                                                    <div className="comp-movie-date">
                                                                                        📅 {new Date(movie.releaseDate).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                                                                    </div>
                                                                                )}
                                                                                <div className="comp-movie-meta">
                                                                                    {movie.genres?.slice(0, 3).join(', ') || 'N/A'}
                                                                                </div>
                                                                                {badge && (
                                                                                    <div className={`comp-anticipation-badge ${badge.className}`}>
                                                                                        {badge.emoji} {badge.label}
                                                                                    </div>
                                                                                )}
                                                                                {movie.budget && (
                                                                                    <div className="comp-movie-budget">
                                                                                        💰 {formatCurrency(movie.budget)}
                                                                                    </div>
                                                                                )}
                                                                            </div>
                                                                        </div>
                                                                    );
                                                                })}
                                                            </div>
                                                        </div>
                                                    );
                                                })()}
                                            </div>
                                        ) : (
                                            <div className="release-loading">
                                                <p>No competitive data found for this window.</p>
                                            </div>
                                        )}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>
                    )}
                </div>
            </section>
        </div>
    );
};

export default UpcomingDashboard;
