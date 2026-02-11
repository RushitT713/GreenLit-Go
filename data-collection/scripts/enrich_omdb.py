"""
OMDB Enrichment Script
Enriches existing movies with OMDB data (IMDb ratings, awards, etc.)
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.omdb_collector import OMDBCollector
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def run_omdb_enrichment():
    print("=" * 60)
    print("🎬 OMDB Enrichment")
    print("   Adding IMDb ratings, awards, Metascore to movies")
    print("=" * 60)
    
    # Initialize
    collector = OMDBCollector()
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    # Get movies without OMDB data
    movies_to_enrich = list(movies_collection.find({
        'omdb': {'$exists': False}
    }))
    
    total = len(movies_to_enrich)
    print(f"✓ Found {total} movies to enrich")
    
    enriched_count = 0
    failed_count = 0
    
    for i, movie in enumerate(movies_to_enrich):
        title = movie.get('title', 'Unknown')
        year = movie.get('year')
        imdb_id = movie.get('imdbId')
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"\n📊 Progress: {i + 1}/{total} ({enriched_count} enriched, {failed_count} failed)")
        
        try:
            # Try by IMDb ID first, then by title
            omdb_data = None
            
            if imdb_id:
                omdb_data = collector.get_by_imdb_id(imdb_id)
            
            if not omdb_data:
                omdb_data = collector.get_by_title(title, year)
            
            if omdb_data:
                enrichment = collector.extract_enrichment_data(omdb_data)
                
                # Update movie in database
                movies_collection.update_one(
                    {'_id': movie['_id']},
                    {
                        '$set': {
                            'omdb': enrichment['omdb'],
                            'imdbId': omdb_data.get('imdbID'),
                            'updatedAt': datetime.now()
                        }
                    }
                )
                
                enriched_count += 1
                print(f"   ✓ {title[:40]} (IMDb: {enrichment['omdb'].get('imdbRating')})")
            else:
                failed_count += 1
                
            # Rate limiting (OMDB has 1000 requests/day limit on free tier)
            time.sleep(0.1)
            
        except Exception as e:
            failed_count += 1
            continue
    
    print("\n" + "=" * 60)
    print("✅ OMDB ENRICHMENT COMPLETE!")
    print("=" * 60)
    print(f"   Movies enriched: {enriched_count}")
    print(f"   Movies not found: {failed_count}")
    
    # Stats
    with_imdb = movies_collection.count_documents({'omdb.imdbRating': {'$exists': True, '$ne': None}})
    with_metascore = movies_collection.count_documents({'omdb.metascore': {'$exists': True, '$ne': None}})
    with_rt = movies_collection.count_documents({'omdb.rottenTomatoes': {'$exists': True, '$ne': None}})
    
    print(f"\n📊 Data Coverage:")
    print(f"   With IMDb Rating: {with_imdb}")
    print(f"   With Metascore: {with_metascore}")
    print(f"   With Rotten Tomatoes: {with_rt}")
    
    client.close()
    return enriched_count

if __name__ == '__main__':
    run_omdb_enrichment()
