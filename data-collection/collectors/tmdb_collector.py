"""
TMDB Data Collector
Fetches movie data from The Movie Database API.
"""

import os
import time
import requests
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

class TMDBCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key or TMDB_API_KEY
        if not self.api_key:
            raise ValueError("TMDB API key is required. Set TMDB_API_KEY in .env")
        self.base_url = TMDB_BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, endpoint, params=None):
        """Make API request with rate limiting."""
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Rate limiting - TMDB allows ~40 requests/10 seconds
            time.sleep(0.25)
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information."""
        return self._make_request(
            f"/movie/{movie_id}",
            params={'append_to_response': 'credits,videos,keywords,release_dates'}
        )
    
    def get_popular_movies(self, page=1, region=None):
        """Get popular movies."""
        params = {'page': page}
        if region:
            params['region'] = region
        return self._make_request('/movie/popular', params)
    
    def get_top_rated_movies(self, page=1):
        """Get top rated movies."""
        return self._make_request('/movie/top_rated', {'page': page})
    
    def get_movies_by_year(self, year, page=1):
        """Get movies released in a specific year."""
        return self._make_request('/discover/movie', {
            'primary_release_year': year,
            'sort_by': 'revenue.desc',
            'page': page
        })
    
    def get_indian_movies(self, page=1, language='hi'):
        """Get Indian movies (Bollywood, Tollywood, etc.)."""
        return self._make_request('/discover/movie', {
            'with_original_language': language,
            'sort_by': 'revenue.desc',
            'page': page
        })
    
    def get_upcoming_movies(self, page=1, region=None):
        """Get upcoming movies."""
        params = {'page': page}
        if region:
            params['region'] = region
        return self._make_request('/movie/upcoming', params)
    
    def search_movies(self, query, page=1):
        """Search for movies by title."""
        return self._make_request('/search/movie', {
            'query': query,
            'page': page
        })
    
    def get_person_details(self, person_id):
        """Get person (actor/director) details."""
        return self._make_request(
            f"/person/{person_id}",
            params={'append_to_response': 'movie_credits'}
        )
    
    def process_movie_for_db(self, movie_data, industry='hollywood'):
        """
        Process TMDB movie data into our database schema format.
        """
        if not movie_data:
            return None
        
        credits = movie_data.get('credits', {})
        videos = movie_data.get('videos', {}).get('results', [])
        
        # Extract director
        director = None
        for crew in credits.get('crew', []):
            if crew.get('job') == 'Director':
                director = {
                    'id': crew.get('id'),
                    'name': crew.get('name'),
                    'popularity': crew.get('popularity', 0)
                }
                break
        
        # Extract top cast
        cast = []
        for actor in credits.get('cast', [])[:10]:
            cast.append({
                'id': actor.get('id'),
                'name': actor.get('name'),
                'character': actor.get('character'),
                'order': actor.get('order'),
                'popularity': actor.get('popularity', 0),
                'profilePath': actor.get('profile_path'),
                'gender': actor.get('gender')
            })
        
        # Extract writers
        writers = []
        for crew in credits.get('crew', []):
            if crew.get('department') == 'Writing':
                writers.append({
                    'id': crew.get('id'),
                    'name': crew.get('name'),
                    'job': crew.get('job')
                })
        
        # Find trailer
        trailer = None
        for video in videos:
            if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
                trailer = {
                    'key': video.get('key'),
                    'url': f"https://www.youtube.com/watch?v={video.get('key')}"
                }
                break
        
        # Extract genres
        genres = [g.get('name') for g in movie_data.get('genres', [])]
        
        # Extract production companies
        production_companies = []
        for company in movie_data.get('production_companies', [])[:5]:
            production_companies.append({
                'id': company.get('id'),
                'name': company.get('name'),
                'logoPath': company.get('logo_path'),
                'originCountry': company.get('origin_country')
            })
        
        # Check if sequel (part of collection)
        is_sequel = movie_data.get('belongs_to_collection') is not None
        
        # Parse release date
        release_date = None
        year = None
        if movie_data.get('release_date'):
            try:
                release_date = datetime.strptime(movie_data['release_date'], '%Y-%m-%d')
                year = release_date.year
            except:
                pass
        
        return {
            'tmdbId': movie_data.get('id'),
            'imdbId': movie_data.get('imdb_id'),
            'title': movie_data.get('title'),
            'originalTitle': movie_data.get('original_title'),
            'overview': movie_data.get('overview'),
            'tagline': movie_data.get('tagline'),
            'industry': industry,
            'releaseDate': release_date,
            'year': year,
            'status': movie_data.get('status'),
            'budget': movie_data.get('budget'),
            'revenue': movie_data.get('revenue'),
            'runtime': movie_data.get('runtime'),
            'genres': genres,
            'language': movie_data.get('original_language'),
            'spokenLanguages': [l.get('english_name') for l in movie_data.get('spoken_languages', [])],
            'countries': [c.get('name') for c in movie_data.get('production_countries', [])],
            'adult': movie_data.get('adult', False),
            'productionCompanies': production_companies,
            'isSequel': is_sequel,
            'partOfCollection': movie_data.get('belongs_to_collection'),
            'cast': cast,
            'director': director,
            'writers': writers,
            'voteAverage': movie_data.get('vote_average'),
            'voteCount': movie_data.get('vote_count'),
            'popularity': movie_data.get('popularity'),
            'posterPath': movie_data.get('poster_path'),
            'backdropPath': movie_data.get('backdrop_path'),
            'trailerUrl': trailer.get('url') if trailer else None,
            'trailerKey': trailer.get('key') if trailer else None,
            'keywords': [k.get('name') for k in movie_data.get('keywords', {}).get('keywords', [])]
        }
    
    def collect_hollywood_movies(self, years_range=(2010, 2024), max_per_year=100):
        """
        Collect Hollywood movies for a range of years.
        """
        all_movies = []
        
        for year in range(years_range[0], years_range[1] + 1):
            print(f"\nCollecting movies from {year}...")
            page = 1
            year_movies = []
            
            while len(year_movies) < max_per_year:
                result = self.get_movies_by_year(year, page)
                if not result or not result.get('results'):
                    break
                
                for movie in result['results']:
                    if len(year_movies) >= max_per_year:
                        break
                    
                    # Get detailed info
                    details = self.get_movie_details(movie['id'])
                    if details:
                        processed = self.process_movie_for_db(details, 'hollywood')
                        if processed:
                            year_movies.append(processed)
                
                page += 1
                if page > result.get('total_pages', 1):
                    break
            
            print(f"  Collected {len(year_movies)} movies from {year}")
            all_movies.extend(year_movies)
        
        return all_movies
    
    def collect_indian_movies(self, languages=None, max_per_language=200):
        """
        Collect Indian cinema movies.
        Languages: hi (Hindi/Bollywood), te (Telugu/Tollywood), 
                   ta (Tamil/Kollywood), ml (Malayalam/Mollywood), kn (Kannada/Sandalwood)
        """
        if languages is None:
            languages = [
                ('hi', 'bollywood'),
                ('te', 'tollywood'),
                ('ta', 'kollywood'),
                ('ml', 'mollywood'),
                ('kn', 'sandalwood')
            ]
        
        all_movies = []
        
        for lang_code, industry in languages:
            print(f"\nCollecting {industry} movies...")
            page = 1
            lang_movies = []
            
            while len(lang_movies) < max_per_language:
                result = self.get_indian_movies(page, lang_code)
                if not result or not result.get('results'):
                    break
                
                for movie in result['results']:
                    if len(lang_movies) >= max_per_language:
                        break
                    
                    details = self.get_movie_details(movie['id'])
                    if details:
                        processed = self.process_movie_for_db(details, industry)
                        if processed:
                            lang_movies.append(processed)
                
                page += 1
                if page > min(result.get('total_pages', 1), 50):
                    break
            
            print(f"  Collected {len(lang_movies)} {industry} movies")
            all_movies.extend(lang_movies)
        
        return all_movies


if __name__ == '__main__':
    # Example usage
    collector = TMDBCollector()
    
    # Test single movie
    movie = collector.get_movie_details(550)  # Fight Club
    if movie:
        processed = collector.process_movie_for_db(movie)
        print(f"Title: {processed['title']}")
        print(f"Director: {processed['director']['name'] if processed['director'] else 'N/A'}")
        print(f"Genres: {', '.join(processed['genres'])}")
