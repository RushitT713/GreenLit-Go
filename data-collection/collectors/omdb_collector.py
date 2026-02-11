"""
OMDB API Collector
Fetches IMDb ratings, awards, Metascore, Rotten Tomatoes
"""

import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

class OMDBCollector:
    """Collects movie data from OMDB API"""
    
    def __init__(self):
        self.api_key = os.getenv('OMDB_API_KEY')
        self.base_url = "http://www.omdbapi.com/"
        
        if not self.api_key:
            raise ValueError("OMDB_API_KEY not found in environment")
        
        print(f"✓ OMDB Collector initialized")
    
    def get_by_imdb_id(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """Fetch movie data by IMDb ID"""
        try:
            params = {
                'apikey': self.api_key,
                'i': imdb_id,
                'plot': 'short'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') == 'True':
                return data
            return None
            
        except Exception as e:
            return None
    
    def get_by_title(self, title: str, year: int = None) -> Optional[Dict[str, Any]]:
        """Fetch movie data by title and optional year"""
        try:
            params = {
                'apikey': self.api_key,
                't': title,
                'plot': 'short'
            }
            
            if year:
                params['y'] = year
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') == 'True':
                return data
            return None
            
        except Exception as e:
            return None
    
    def extract_enrichment_data(self, omdb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data for enrichment"""
        enrichment = {
            'omdb': {
                'imdbID': omdb_data.get('imdbID'),
                'imdbRating': self._parse_float(omdb_data.get('imdbRating')),
                'imdbVotes': self._parse_int(omdb_data.get('imdbVotes')),
                'metascore': self._parse_int(omdb_data.get('Metascore')),
                'rated': omdb_data.get('Rated'),
                'awards': omdb_data.get('Awards'),
                'boxOffice': omdb_data.get('BoxOffice'),
                'rottenTomatoes': None
            }
        }
        
        # Extract Rotten Tomatoes rating
        ratings = omdb_data.get('Ratings', [])
        for rating in ratings:
            if rating.get('Source') == 'Rotten Tomatoes':
                rt_value = rating.get('Value', '').replace('%', '')
                enrichment['omdb']['rottenTomatoes'] = self._parse_int(rt_value)
                break
        
        return enrichment
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse string to float, handling N/A"""
        if not value or value == 'N/A':
            return None
        try:
            return float(value)
        except:
            return None
    
    def _parse_int(self, value: str) -> Optional[int]:
        """Parse string to int, handling N/A and commas"""
        if not value or value == 'N/A':
            return None
        try:
            return int(str(value).replace(',', ''))
        except:
            return None


if __name__ == '__main__':
    # Test
    collector = OMDBCollector()
    result = collector.get_by_title("Inception", 2010)
    if result:
        print(f"Found: {result.get('Title')}")
        enrichment = collector.extract_enrichment_data(result)
        print(f"IMDb Rating: {enrichment['omdb']['imdbRating']}")
        print(f"Metascore: {enrichment['omdb']['metascore']}")
        print(f"Rotten Tomatoes: {enrichment['omdb']['rottenTomatoes']}")
