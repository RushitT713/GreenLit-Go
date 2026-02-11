import React, { useEffect, useRef, useState } from 'react';
import './Home.css';

// Counter Animation Hook
const useCountUp = (end, duration = 2000, startCounting = false) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        if (!startCounting) return;

        let startTime = null;
        const startValue = 0;

        const animate = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);

            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            setCount(Math.floor(easeOutQuart * (end - startValue) + startValue));

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }, [end, duration, startCounting]);

    return count;
};

const Home = () => {
    const [statsVisible, setStatsVisible] = useState(false);
    const statsRef = useRef(null);

    // Counter values
    const moviesCount = useCountUp(1600, 2000, statsVisible);
    const accuracyCount = useCountUp(93, 2000, statsVisible);
    const industriesCount = useCountUp(5, 1500, statsVisible);

    // Capabilities data
    const capabilities = [
        {
            icon: '💰',
            title: 'Revenue Prediction',
            description: 'Predict box office gross with ML models trained on 575+ movies. Forecast opening weekend, domestic, and worldwide revenue.'
        },
        {
            icon: '📊',
            title: 'Success Classification',
            description: 'Categorize movies as Hit, Average, or Flop based on ROI analysis and comprehensive performance metrics.'
        },
        {
            icon: '⭐',
            title: 'Rating Prediction',
            description: 'Forecast IMDb scores and Rotten Tomatoes ratings before release using sentiment and buzz analysis.'
        },
        {
            icon: '📅',
            title: 'Optimal Release Date',
            description: 'Identify the perfect release window by analyzing seasonal trends, competition, and genre patterns.'
        },
        {
            icon: '🎭',
            title: 'Cast & Crew Impact',
            description: 'Understand how director, actors, and production company choices affect movie performance.'
        },
        {
            icon: '🌍',
            title: 'Dual Market Analysis',
            description: 'Specialized models for both Hollywood and Indian cinema (Bollywood, Tollywood, Kollywood).'
        }
    ];

    // Scroll Reveal Animation using Intersection Observer
    useEffect(() => {
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, observerOptions);

        // Observe all elements with scroll-reveal class
        const revealElements = document.querySelectorAll('.scroll-reveal');
        revealElements.forEach(el => observer.observe(el));

        return () => observer.disconnect();
    }, []);

    // Stats section observer for counter animation
    useEffect(() => {
        const statsObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting && !statsVisible) {
                        setStatsVisible(true);
                    }
                });
            },
            { threshold: 0.3 }
        );

        if (statsRef.current) {
            statsObserver.observe(statsRef.current);
        }

        return () => statsObserver.disconnect();
    }, [statsVisible]);

    return (
        <div className="home">
            {/* Hero Section */}
            <section className="hero">
                {/* Vertical Grey Dividers (4 lines) - z-index: 1 (BEHIND image) */}
                <div className="hero-divider divider-1"></div>
                <div className="hero-divider divider-2"></div>
                <div className="hero-divider divider-3"></div>
                <div className="hero-divider divider-4"></div>

                {/* Background Image Layer - z-index: 2 (IN FRONT of lines) */}
                <div className="hero-bg-image"></div>

                {/* Gradient Overlay for text visibility - z-index: 3 */}
                <div className="hero-gradient-overlay"></div>

                {/* Hero Text Content (on top of everything) - z-index: 10 */}
                <div className="hero-content fade-in">
                    <h1 className="hero-title">
                        <span className="title-white">DATA-DRIVEN</span>
                        <br />
                        <span className="title-orange">BLOCKBUSTERS</span>
                        <span className="title-dot">.</span>
                    </h1>
                    <p className="hero-subtitle-description">
                        Leverage machine learning to analyze Hollywood and Indian cinema. Get data-driven insights on revenue, ratings, and success factors before the cameras roll.
                    </p>
                </div>

                {/* Bottom Horizontal Line */}
                <div className="hero-bottom-line"></div>
            </section>

            {/* Capabilities Section */}
            <section className="capabilities">
                {/* Vertical Grey Dividers - continuing from Hero */}
                <div className="capabilities-divider divider-1"></div>
                <div className="capabilities-divider divider-2"></div>
                <div className="capabilities-divider divider-3"></div>
                <div className="capabilities-divider divider-4"></div>

                {/* Section Header */}
                <div className="section-header scroll-reveal fade-up">
                    <div className="section-label">CAPABILITIES</div>
                    <h2 className="section-title">Powerful Analytics for Filmmakers</h2>
                    <p className="section-description">
                        From revenue prediction to optimal release timing, our platform provides comprehensive insights
                    </p>
                </div>


                <div className="capabilities-grid">
                    {capabilities.map((cap, index) => (
                        <div
                            key={index}
                            className="capability-card scroll-reveal fade-up"
                            style={{ transitionDelay: `${index * 0.1}s` }}
                        >
                            <span className="capability-icon">{cap.icon}</span>
                            <h3 className="capability-name">{cap.title}</h3>
                            <p className="capability-desc">{cap.description}</p>
                        </div>
                    ))}
                </div>

                {/* Bottom Horizontal Line */}
                <div className="capabilities-bottom-line"></div>
            </section>

            {/* Stats Section */}
            <section className="stats" ref={statsRef}>
                <div className="stats-grid">
                    <div className="stat-item scroll-reveal fade-up" style={{ transitionDelay: '0s' }}>
                        <div className="stat-number">{moviesCount}+</div>
                        <div className="stat-label">Movies Analyzed</div>
                    </div>
                    <div className="stat-item scroll-reveal fade-up" style={{ transitionDelay: '0.1s' }}>
                        <div className="stat-number">{accuracyCount}%</div>
                        <div className="stat-label">Prediction Accuracy</div>
                    </div>
                    <div className="stat-item scroll-reveal fade-up" style={{ transitionDelay: '0.2s' }}>
                        <div className="stat-number">2015-24</div>
                        <div className="stat-label">Data Coverage</div>
                    </div>
                    <div className="stat-item scroll-reveal fade-up" style={{ transitionDelay: '0.3s' }}>
                        <div className="stat-number">{industriesCount}+</div>
                        <div className="stat-label">Film Industries</div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Home;
