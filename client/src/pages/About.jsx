import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import rushitImg from '../assets/rushit-img.jpeg';
import vanditImg from '../assets/vandit-img.jpeg';
import './About.css';

const About = () => {
    const navigate = useNavigate();

    // Stats data with SVG icons
    const stats = [
        {
            icon: <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff8300" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="2" width="20" height="20" rx="2" /><path d="M7 2v20M17 2v20M2 12h20M2 7h5M2 17h5M17 7h5M17 17h5" /></svg>,
            value: '1,600+', label: 'Movies Analyzed'
        },
        {
            icon: <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff8300" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" /></svg>,
            value: '5', label: 'Industries Covered'
        },
        {
            icon: <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff8300" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" /><path d="M10 21h4M10 17v-1a2 2 0 012-2h0a2 2 0 012 2v1" /></svg>,
            value: '40+', label: 'ML Features'
        },
        {
            icon: <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ff8300" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></svg>,
            value: '93%', label: 'Model Accuracy'
        }
    ];

    // Tech stack data with real logos
    const techStack = [
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/react/react-original.svg" alt="React" width="36" height="36" />,
            name: 'React', role: 'Frontend UI'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/nodejs/nodejs-original.svg" alt="Node.js" width="36" height="36" />,
            name: 'Node.js', role: 'Backend API'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" alt="Python" width="36" height="36" />,
            name: 'Python', role: 'ML Pipeline'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/mongodb/mongodb-original.svg" alt="MongoDB" width="36" height="36" />,
            name: 'MongoDB', role: 'Database'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/flask/flask-original.svg" alt="Flask" width="36" height="36" style={{ filter: 'invert(1)' }} />,
            name: 'Flask', role: 'ML Service'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" alt="Scikit-Learn" width="36" height="36" />,
            name: 'Scikit-Learn', role: 'ML Models'
        },
        {
            icon: <img src="https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png" alt="XGBoost" width="36" height="36" style={{ objectFit: 'contain' }} />,
            name: 'XGBoost', role: 'Ensemble Models'
        },
        {
            icon: <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/json/json-original.svg" alt="TMDB API" width="36" height="36" />,
            name: 'TMDB API', role: 'Movie Data'
        }
    ];

    // Capabilities for timeline
    const capabilities = [
        {
            title: 'Revenue Prediction',
            description: 'Predict box office gross using ensemble ML models trained on 1,600+ movies across multiple industries.'
        },
        {
            title: 'Success Classification',
            description: 'Automatically categorize movies into Blockbuster, Super Hit, Hit, Average, Below Average, or Flop based on ROI analysis.'
        },
        {
            title: 'Explainable AI (SHAP)',
            description: 'Understand exactly WHY a prediction was made with SHAP-based feature importance — full model transparency.'
        },
        {
            title: 'Optimal Release Timing',
            description: 'Discover the best release month using historical seasonal trends, revenue patterns, and success rate analysis.'
        },
        {
            title: 'Competition Analysis',
            description: 'Real-time TMDB-powered analysis of competing movies in your chosen release window with threat assessment.'
        },
        {
            title: 'What-If Simulator',
            description: 'Interactively tweak budget, genre, release month, and cast to see how predictions change in real-time.'
        }
    ];

    // Team members
    const team = [
        {
            name: 'Rushit Trambadia',
            role: 'ML & Backend Developer',
            description: 'Built the machine learning pipeline, trained prediction models, and engineered 40+ features from multi-source movie data.',
            photo: rushitImg,
            github: '',
            linkedin: ''
        },
        {
            name: 'Vandit Doshi',
            role: 'Frontend Developer',
            description: 'Brought the website to life with dynamic React interfaces, interactive dashboards, and premium visual design.',
            photo: vanditImg,
            github: '',
            linkedin: ''
        },
        {
            name: 'Prof. Priyanka Mangi',
            role: 'Project Guide',
            description: 'Provided academic mentorship, guided the research methodology, and ensured the project met scholarly standards.',
            photo: null,
            github: '',
            linkedin: ''
        }
    ];

    // Scroll Reveal Animation
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('revealed');
                    }
                });
            },
            { threshold: 0.1 }
        );

        const revealElements = document.querySelectorAll('.scroll-reveal');
        revealElements.forEach(el => observer.observe(el));
        return () => observer.disconnect();
    }, []);

    return (
        <div className="about-page">

            {/* ===== HERO SECTION ===== */}
            <section className="about-hero">
                <div className="about-hero-glow"></div>
                <div className="about-hero-content fade-in">
                    <span className="about-label">INTRODUCTION</span>
                    <h1 className="about-hero-title">
                        About <span className="text-orange">GreenLit GO</span>
                    </h1>
                    <p className="about-hero-subtitle">
                        Empowering filmmakers and studios with data-driven movie analytics
                        and pre-release predictions powered by machine learning.
                    </p>
                </div>
                <div className="about-hero-bottom-line"></div>
            </section>

            {/* ===== MISSION SECTION ===== */}
            <section className="about-mission">
                <div className="about-mission-inner">
                    <div className="about-mission-text scroll-reveal fade-up">
                        <h2 className="about-section-title">Our Mission</h2>
                        <p className="about-mission-desc">
                            We're building an intelligent platform that bridges the gap between
                            data science and the film industry. By analyzing historical performance
                            data from over 1,600 movies, our ML models help filmmakers, producers,
                            and studios make informed decisions — from choosing the right release
                            window to understanding which factors drive box office success.
                        </p>
                        <p className="about-mission-desc">
                            GreenLit GO covers <strong>Hollywood, Bollywood, Tollywood, Kollywood,</strong> and
                            <strong> Mollywood</strong> — making it one of the most comprehensive multi-industry
                            movie prediction platforms.
                        </p>
                    </div>
                    <div className="about-stats-grid scroll-reveal fade-up" style={{ transitionDelay: '0.2s' }}>
                        {stats.map((stat, index) => (
                            <div key={index} className="about-stat-card">
                                <span className="about-stat-icon">{stat.icon}</span>
                                <span className="about-stat-value">{stat.value}</span>
                                <span className="about-stat-label">{stat.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <div className="about-section-line-2"></div>

            {/* ===== TECH STACK SECTION ===== */}
            <section className="about-tech">
                <div className="about-tech-inner">
                    <div className="about-section-header scroll-reveal fade-up">
                        <h2 className="about-section-title">Built With Precision</h2>
                        <p className="about-section-subtitle">
                            A modern full-stack architecture combining React, Node.js, Python, and MongoDB
                            for scalable, real-time movie analytics.
                        </p>
                    </div>
                    <div className="about-tech-grid scroll-reveal fade-up" style={{ transitionDelay: '0.15s' }}>
                        {techStack.map((tech, index) => (
                            <div key={index} className="about-tech-card">
                                <span className="about-tech-icon">{tech.icon}</span>
                                <span className="about-tech-name">{tech.name}</span>
                                <span className="about-tech-role">{tech.role}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <div className="about-section-line-3"></div>

            {/* ===== WHAT WE OFFER SECTION ===== */}
            <section className="about-offer">
                <div className="about-offer-inner">
                    <div className="about-section-header scroll-reveal fade-up">
                        <h2 className="about-section-title">What We Offer</h2>
                        <p className="about-section-subtitle">
                            End-to-end tools for film industry decision-making.
                        </p>
                    </div>
                    <div className="about-timeline scroll-reveal fade-up" style={{ transitionDelay: '0.15s' }}>
                        {capabilities.map((cap, index) => (
                            <div key={index} className="about-timeline-item">
                                <div className="about-timeline-dot"></div>
                                <div className="about-timeline-content">
                                    <h3 className="about-timeline-title">{cap.title}</h3>
                                    <p className="about-timeline-desc">{cap.description}</p>
                                </div>
                            </div>
                        ))}
                        <div className="about-timeline-line"></div>
                    </div>
                </div>
            </section>

            <div className="about-section-line-4"></div>

            {/* ===== MEET THE BUILDERS SECTION ===== */}
            <section className="about-team">
                <div className="about-team-inner">
                    <div className="about-section-header scroll-reveal fade-up">
                        <h2 className="about-section-title">Meet the Builders</h2>
                        <p className="about-section-subtitle">
                            The minds behind GreenLit GO.
                        </p>
                    </div>
                    <div className="about-team-grid scroll-reveal fade-up" style={{ transitionDelay: '0.15s' }}>
                        {team.map((member, index) => (
                            <div key={index} className="about-team-card">
                                <div className="about-team-photo">
                                    {member.photo ? (
                                        <img src={member.photo} alt={member.name} />
                                    ) : (
                                        <div className="about-team-placeholder">
                                            <span>{member.name.split(' ').map(n => n[0]).join('')}</span>
                                        </div>
                                    )}
                                </div>
                                <div className="about-team-info">
                                    <div className="about-team-name-row">
                                        <div>
                                            <h3 className="about-team-name">{member.name}</h3>
                                            <p className="about-team-role">{member.role}</p>
                                        </div>
                                        <a
                                            href={member.github || '#'}
                                            className="about-team-arrow"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            aria-label={`Visit ${member.name}'s profile`}
                                        >
                                            ↗
                                        </a>
                                    </div>
                                    <p className="about-team-desc">{member.description}</p>
                                    <div className="about-team-socials">
                                        {member.github && (
                                            <a href={member.github} target="_blank" rel="noopener noreferrer" className="about-social-link">
                                                GitHub
                                            </a>
                                        )}
                                        {member.linkedin && (
                                            <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="about-social-link">
                                                LinkedIn
                                            </a>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <div className="about-section-line-5"></div>

            {/* ===== CTA SECTION ===== */}
            <section className="about-cta scroll-reveal fade-up">
                <div className="about-cta-inner">
                    <h2 className="about-cta-title">Ready to Greenlight Your Next Hit?</h2>
                    <p className="about-cta-desc">
                        Start predicting movie success with our data-driven analytics platform.
                    </p>
                    <div className="about-cta-buttons">
                        <button className="about-cta-btn primary" onClick={() => navigate('/upcoming')}>
                            Get Started
                        </button>
                        <button className="about-cta-btn secondary" onClick={() => navigate('/library')}>
                            View Library
                        </button>
                    </div>
                </div>
            </section>

        </div>
    );
};

export default About;
