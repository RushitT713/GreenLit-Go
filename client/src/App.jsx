import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import Footer from './components/common/Footer';
import Home from './pages/Home';
import ReleasedMovies from './pages/ReleasedMovies';
import MovieDetail from './pages/MovieDetail';
import UpcomingDashboard from './pages/UpcomingDashboard';
import About from './pages/About';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/movies" element={<ReleasedMovies />} />
            <Route path="/movies/:id" element={<MovieDetail />} />
            <Route path="/upcoming" element={<UpcomingDashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
