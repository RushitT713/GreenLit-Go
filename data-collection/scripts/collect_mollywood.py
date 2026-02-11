"""
Mollywood (Malayalam) Collection
"""

import os
import time
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

def run_mollywood_collection():
    print("=" * 60)
    print("🎬 Mollywood (Malayalam) Collection")
    print("=" * 60)
    
    # MongoDB
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    initial_count = movies_collection.count_documents({})
    print(f"✓ Connected (current: {initial_count} movies)")
    
    total = 0
    pages = 5  # ~100 movies
    
    for page in range(1, pages + 1):
        # Fetch Malayalam movies
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': 'ml',
            'sort_by': 'revenue.desc',
            'page': page
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        
        data = response.json()
        movies = data.get('results', [])
        print(f"\n📄 Page {page}: {len(movies)} movies")
        
        for m in movies:
            try:
                # Get details
                url = f"{TMDB_BASE_URL}/movie/{m['id']}"
                params = {'api_key': TMDB_API_KEY, 'append_to_response': 'credits'}
                resp = requests.get(url, params=params)
                if resp.status_code != 200:
                    continue
                
                movie = resp.json()
                credits = movie.get('credits', {})
                
                # Director
                director = None
                for crew in credits.get('crew', []):
                    if crew.get('job') == 'Director':
                        director = {'id': crew.get('id'), 'name': crew.get('name')}
                        break
                
                # Cast
                cast = [{'id': a.get('id'), 'name': a.get('name')} for a in credits.get('cast', [])[:5]]
                
                # Genres
                genres = [g.get('name') for g in movie.get('genres', [])]
                
                # Year
                release_date = movie.get('release_date')
                year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None
                
                processed = {
                    'tmdbId': movie.get('id'),
                    'title': movie.get('title'),
                    'originalTitle': movie.get('original_title'),
                    'overview': movie.get('overview'),
                    'industry': 'mollywood',
                    'releaseDate': release_date,
                    'year': year,
                    'budget': movie.get('budget'),
                    'revenue': movie.get('revenue'),
                    'runtime': movie.get('runtime'),
                    'genres': genres,
                    'originalLanguage': movie.get('original_language'),
                    'director': director,
                    'cast': cast,
                    'voteAverage': movie.get('vote_average'),
                    'voteCount': movie.get('vote_count'),
                    'popularity': movie.get('popularity'),
                    'posterPath': movie.get('poster_path'),
                    'createdAt': datetime.now(),
                    'updatedAt': datetime.now()
                }
                
                movies_collection.update_one(
                    {'tmdbId': processed['tmdbId']},
                    {'$set': processed},
                    upsert=True
                )
                total += 1
                print(f"   ✓ {processed['title'][:45]}")
                time.sleep(0.15)
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue
    
    # Stats
    final_count = movies_collection.count_documents({})
    mw_count = movies_collection.count_documents({'industry': 'mollywood'})
    
    print("\n" + "=" * 60)
    print("✅ MOLLYWOOD COLLECTION COMPLETE!")
    print(f"   Collected: {total} movies")
    print(f"   Total Mollywood: {mw_count}")
    print(f"   Total in database: {final_count}")
    client.close()

if __name__ == '__main__':
    run_mollywood_collection()
