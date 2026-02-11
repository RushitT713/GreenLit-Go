import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faClapperboard } from '@fortawesome/free-solid-svg-icons';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="footer">
            {/* Top Border Gradient */}
            <div className="footer-gradient-border"></div>

            <div className="footer-container">
                {/* Main Footer Content */}
                <div className="footer-main">
                    {/* Brand Section */}
                    <div className="footer-brand">
                        <Link to="/" className="footer-logo">
                            {/* <FontAwesomeIcon icon={faClapperboard} className="logo-icon" /> */}
                            <span className="logo-white">GreenLit</span>
                            <span className="logo-orange">GO</span>
                        </Link>
                        <p className="footer-tagline">
                            Predicting movie success with machine learning.
                            Data-driven insights for filmmakers, studios, and investors.
                        </p>
                        <div className="footer-social">
                            <a href="#" className="social-link" aria-label="GitHub">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                                </svg>
                            </a>
                            <a href="#" className="social-link" aria-label="LinkedIn">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                                </svg>
                            </a>
                            <a href="#" className="social-link" aria-label="Twitter">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                                </svg>
                            </a>
                        </div>
                    </div>

                    {/* Navigation Links */}
                    <div className="footer-nav">
                        <div className="footer-column">
                            <h4 className="footer-heading">Navigation</h4>
                            <ul className="footer-list">
                                <li><Link to="/">Home</Link></li>
                                <li><Link to="/movies">Released Movies</Link></li>
                                <li><Link to="/upcoming">Upcoming Movies</Link></li>
                                <li><Link to="/about">About Us</Link></li>
                            </ul>
                        </div>

                        <div className="footer-column">
                            <h4 className="footer-heading">Industries</h4>
                            <ul className="footer-list">
                                <li><span>Hollywood</span></li>
                                <li><span>Bollywood</span></li>
                                <li><span>Tollywood</span></li>
                                <li><span>Kollywood</span></li>
                                <li><span>Mollywood</span></li>
                            </ul>
                        </div>

                        <div className="footer-column">
                            <h4 className="footer-heading">Data Sources</h4>
                            <ul className="footer-list">
                                <li><span>TMDB API</span></li>
                                <li><span>OMDB API</span></li>
                                <li><span>Box Office Mojo</span></li>
                                <li><span>YouTube Analytics</span></li>
                            </ul>
                        </div>

                        <div className="footer-column">
                            <h4 className="footer-heading">Technology</h4>
                            <ul className="footer-list">
                                <li><span>React.js</span></li>
                                <li><span>Python + FastAPI</span></li>
                                <li><span>Scikit-learn</span></li>
                                <li><span>MongoDB</span></li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Divider with gradient */}
                <div className="footer-divider"></div>

                {/* Bottom Section */}
                <div className="footer-bottom">
                    <p className="footer-copyright">
                        © {currentYear} <span className="copyright-brand">GreenLit<span className="orange">GO</span></span>. All rights reserved.
                    </p>
                    <p className="footer-credits">
                        Powered by Machine Learning • Built with ❤️
                    </p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
