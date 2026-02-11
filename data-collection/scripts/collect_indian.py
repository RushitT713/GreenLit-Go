"""
Indian Cinema Collection - Simplified Version
Directly uses TMDB API with minimal processing to avoid errors
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

def get_movies_by_language(language, page=1):
    """Get movies by original language"""
    url = f"{TMDB_BASE_URL}/discover/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'with_original_language': language,
        'sort_by': 'revenue.desc',
        'page': page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_movie_details(movie_id):
    """Get detailed movie info"""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,videos'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def process_movie(movie, industry):
    """Process movie data for database - simplified"""
    try:
        credits = movie.get('credits', {})
        
        # Director
        director = None
        for crew in credits.get('crew', []):
            if crew.get('job') == 'Director':
                director = {'id': crew.get('id'), 'name': crew.get('name')}
                break
        
        # Cast (top 5)
        cast = []
        for actor in credits.get('cast', [])[:5]:
            cast.append({
                'id': actor.get('id'),
                'name': actor.get('name'),
                'character': actor.get('character')
            })
        
        # Genres
        genres = [g.get('name') for g in movie.get('genres', [])]
        
        # Parse year
        year = None
        release_date = movie.get('release_date')
        if release_date and len(release_date) >= 4:
            year = int(release_date[:4])
        
        return {
            'tmdbId': movie.get('id'),
            'imdbId': movie.get('imdb_id'),
            'title': movie.get('title'),
            'originalTitle': movie.get('original_title'),
            'overview': movie.get('overview'),
            'industry': industry,
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
            'backdropPath': movie.get('backdrop_path'),
            'createdAt': datetime.now(),
            'updatedAt': datetime.now()
        }
    except Exception as e:
        print(f"      Error processing: {e}")
        return None

def run_collection():
    print("=" * 60)
    print("🎬 Indian Cinema Collection (Simplified)")
    print("=" * 60)
    
    # MongoDB setup
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    initial_count = movies_collection.count_documents({})
    print(f"✓ Connected (current: {initial_count} movies)")
    
    # Industries to collect
    industries = [
        ('hi', 'bollywood', 'Bollywood', 8),   # ~160 movies
        ('te', 'tollywood', 'Tollywood', 5),   # ~100 movies
        ('ta', 'kollywood', 'Kollywood', 5),   # ~100 movies
    ]
    
    total = 0
    
    for lang, industry, name, pages in industries:
        print(f"\n📽️ {name}:")
        count = 0
        
        for page in range(1, pages + 1):
            result = get_movies_by_language(lang, page)
            if not result or not result.get('results'):
                break
            
            movies = result.get('results', [])
            print(f"   Page {page}: {len(movies)} movies")
            
            for m in movies:
                try:
                    details = get_movie_details(m['id'])
                    if not details:
                        continue
                    
                    processed = process_movie(details, industry)
                    if not processed:
                        continue
                    
                    # Upsert
                    movies_collection.update_one(
                        {'tmdbId': processed['tmdbId']},
                        {'$set': processed},
                        upsert=True
                    )
                    count += 1
                    total += 1
                    
                    # Brief output
                    print(f"      ✓ {processed['title'][:40]}")
                    time.sleep(0.15)
                    
                except Exception as e:
                    print(f"      ✗ Error: {e}")
                    continue
        
        print(f"   → {name}: {count} movies")
    
    # Stats
    final_count = movies_collection.count_documents({})
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print(f"   Total collected: {total}")
    print(f"   Before: {initial_count} | After: {final_count}")
    
    print("\n📊 By Industry:")
    for ind in ['hollywood', 'bollywood', 'tollywood', 'kollywood']:
        c = movies_collection.count_documents({'industry': ind})
        if c > 0:
            print(f"   {ind.capitalize()}: {c}")
    
    client.close()

if __name__ == '__main__':
    run_collection()
