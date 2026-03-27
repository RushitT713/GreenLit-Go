# Active Context: GreenLit GO Project

This document serves as a persistent record of the project state, progress, and pending tasks to ensure continuity across development sessions.

## 🚀 Progress Summary

### 1. Personalized Recommendation System (Completed)
- **Backend Implementation**: Added `/api/movies/recommend` in `server/src/routes/movies.js`.
  - Supports 6 discovery modes: Mood, Genre Mix, Similarity, Talent, Top-Rated, and Surprise Me.
  - Implemented scoring logic based on genre overlap, popularity, and talent matching.
- **Frontend Implementation**: Rewrote `ReleasedMovies.jsx` to include a dual-mode interface.
  - **Library**: Browse existing movies with filters and search.
  - **Discover**: Interactive recommendation engine with mood chips, talent search, and similarity inputs.
- **UI/UX & Styling**: 
  - Aligned "Released Movies" tabs with the "Upcoming Dashboard" design.
  - Features centered, pill-shaped tabs (`border-radius: 50px`) with an orange gradient for active states.
  - Responsive layout improvements for mobile and desktop.
  - Integrated `framer-motion` for smooth tab transitions.

### 2. Infrastructure & Data
- **Database**: Migrated to MongoDB Atlas (handled via `migrate_to_atlas.py`).
- **Services**: Centralized API calls in `client/src/services/api.js`.

---

## 📂 Key File Structure

```text
e:/ml-web-app-project/
├── client/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ReleasedMovies.jsx      # Main UI for Library & Discover
│   │   │   ├── ReleasedMovies.css      # Premium styling for recommendations
│   │   │   ├── UpcomingDashboard.jsx   # Reference for tab/prediction UI
│   │   │   └── UpcomingDashboard.css   # Reference for global design tokens
│   │   └── services/
│   │       └── api.js                  # Frontend API client (movieService.recommend)
├── server/
│   └── src/
│       └── routes/
│           ├── movies.js               # Recommendation engine logic
│           └── predictions.js          # Success prediction endpoints
└── ml-service/
    └── app.py                          # ML Python service (Success Prediction & SHAP)
```

---

## 📝 Planned Features & Logic

### Feature 2: Script Analysis
- **Goal**: Allow users to upload movie scripts for AI-driven analysis.
- **Components**:
  - **360° Analysis**: Summary of plot, tone, and pacing.
  - **Audience Demographics**: Identify target age groups, interests, and markets.
  - **Prediction Bridge**: Explain how specific script elements (e.g., "high intensity", "dark tone") affect the movie's success category/revenue prediction.
- **Approach**:
  - **Gemini API (Primary)**: Use Gemini 1.5 Flash (free tier) for high-quality text analysis.
  - **Local Approach (Fallback/Hybrid)**: Basic NLP or local LLM for cost-free offline analysis.
  - **Requirement**: Must spend $0 — leverage free tiers and efficient local processing.
  - **UI**: Add a "Script Analysis" section to the dashboard or a dedicated page.

---

## 🛠️ Pending Tasks

- [ ] **Implementation of Script Analysis (Feature 2)**
  - [ ] Setup script upload handler in Node.js server.
  - [ ] Integrate Gemini API for text processing.
  - [ ] Create UI for displaying demographic and analysis cards.
  - [ ] Link script insights to the Success Prediction model.
- [ ] **Comparative Analysis Mode**: Implement a UI element that compares Gemini API findings with local NLP results.
- [ ] **Testing**: Verify Script Analysis on various file formats (.pdf, .txt).
- [ ] **More Data Collection**: Collect more data for Indian Cinema and Hollywood.
- [ ] **Work on Model and Improve Model**: Improvements in ML Model after data collection.
- [ ] **Mobile Responsiveness**: Further optimize for mobile screens.

---

## 🔮 Future Work (Time Permitting / Post-Review 2)

- [ ] **Real-time Data Refresh**: Automated periodic re-collection from TMDB API.
- [ ] **User Authentication**: Login system for saved predictions and history.
- [ ] **Prediction History**: Track and compare past predictions vs actual results.

---

## 🔍 How to Continue
If starting a new session:
1. Read this file (`active_context.md`) to sync state.
2. Check `ReleasedMovies.jsx` for the current tab implementation pattern.
3. Start the servers:
   - Backend: `npm run dev` in `server/`
   - Frontend: `npm run dev` in `client/`
   - ML: `python -m app` in `ml-service/`
