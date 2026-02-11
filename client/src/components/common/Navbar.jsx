import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faClapperboard } from '@fortawesome/free-solid-svg-icons';

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();

    const navItems = [
        { name: 'Home', path: '/' },
        { name: 'Released Movies', path: '/movies' },
        { name: 'Upcoming Movies', path: '/upcoming' },
        { name: 'About Us', path: '/about' }
    ];

    const isActive = (path) => {
        if (path === '/') return location.pathname === '/';
        return location.pathname.startsWith(path);
    };

    return (
        <nav className="navbar">
            <div className="navbar-container">
                {/* Column 1: Logo */}
                <div className="navbar-column navbar-logo-section">
                    <Link to="/" className="navbar-logo">
                        {/* <FontAwesomeIcon icon={faClapperboard} className="logo-icon" /> */}
                        <span className="logo-white">GreenLit</span>
                        <span className="logo-orange">GO</span>
                    </Link>
                </div>

                {/* Vertical Divider */}
                <div className="navbar-divider"></div>

                {/* Column 2: Navigation Links */}
                <div className="navbar-column navbar-nav-section">
                    <ul className={`navbar-nav ${isOpen ? 'open' : ''}`}>
                        {navItems.map((item) => (
                            <li key={item.path}>
                                <Link
                                    to={item.path}
                                    className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
                                    onClick={() => setIsOpen(false)}
                                >
                                    {item.name}
                                </Link>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Vertical Divider */}
                <div className="navbar-divider"></div>

                {/* Column 3: Empty Space (same width as logo) */}
                <div className="navbar-column navbar-empty-section"></div>

                {/* Mobile Toggle */}
                <button
                    className={`navbar-toggle ${isOpen ? 'open' : ''}`}
                    onClick={() => setIsOpen(!isOpen)}
                    aria-label="Toggle navigation"
                >
                    <span></span>
                    <span></span>
                    <span></span>
                </button>
            </div>
        </nav>

    );
};

export default Navbar;
