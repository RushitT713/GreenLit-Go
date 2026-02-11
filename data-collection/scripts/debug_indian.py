"""Debug - test single movie collection"""
import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

print("Testing Indian movie collection debug...\n")

# 1. Get a Hindi movie
print("1. Fetching Hindi movies list...")
url = f"{TMDB_BASE_URL}/discover/movie"
params = {
    'api_key': TMDB_API_KEY,
    'with_original_language': 'hi',
    'sort_by': 'revenue.desc',
    'page': 1
}
response = requests.get(url, params=params)
data = response.json()
movies = data.get('results', [])
print(f"   Found {len(movies)} movies")

if movies:
    movie_info = movies[0]
    movie_id = movie_info['id']
    print(f"   First movie: {movie_info['title']} (ID: {movie_id})")
    
    # 2. Get movie details
    print("\n2. Fetching movie details...")
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,videos'
    }
    response = requests.get(url, params=params)
    details = response.json()
    print(f"   Title: {details.get('title')}")
    print(f"   Budget: {details.get('budget')}")
    print(f"   Revenue: {details.get('revenue')}")
    print(f"   Release: {details.get('release_date')}")
    
    # 3. Process movie
    print("\n3. Processing movie data...")
    try:
        credits = details.get('credits', {})
        
        director = None
        for crew in credits.get('crew', []):
            if crew.get('job') == 'Director':
                director = {'id': crew.get('id'), 'name': crew.get('name')}
                break
        
        cast = []
        for actor in credits.get('cast', [])[:3]:
            cast.append({'id': actor.get('id'), 'name': actor.get('name')})
        
        genres = [g.get('name') for g in details.get('genres', [])]
        
        release_date = details.get('release_date')
        year = None
        if release_date and len(release_date) >= 4:
            year = int(release_date[:4])
        
        processed = {
            'tmdbId': details.get('id'),
            'title': details.get('title'),
            'industry': 'bollywood',
            'year': year,
            'budget': details.get('budget'),
            'revenue': details.get('revenue'),
            'genres': genres,
            'director': director,
            'cast': cast,
            'posterPath': details.get('poster_path'),
            'createdAt': datetime.now(),
            'updatedAt': datetime.now()
        }
        
        print(f"   Processed: {processed['title']}")
        print(f"   Year: {processed['year']}")
        print(f"   Director: {processed['director']}")
        
        # 4. Save to MongoDB
        print("\n4. Saving to MongoDB...")
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
        client = MongoClient(mongo_uri)
        db = client.get_database()
        movies_collection = db['movies']
        
        result = movies_collection.update_one(
            {'tmdbId': processed['tmdbId']},
            {'$set': processed},
            upsert=True
        )
        
        print(f"   Matched: {result.matched_count}")
        print(f"   Modified: {result.modified_count}")
        print(f"   Upserted ID: {result.upserted_id}")
        
        # Verify
        saved = movies_collection.find_one({'tmdbId': processed['tmdbId']})
        print(f"\n5. Verification: Found in DB: {saved is not None}")
        if saved:
            print(f"   Title: {saved.get('title')}")
            print(f"   Industry: {saved.get('industry')}")
        
        client.close()
        print("\n✅ Debug complete - everything works!")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
