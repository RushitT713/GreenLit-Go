"""
TMDB Hollywood Collection - Comprehensive
Fetches 1500+ Hollywood movies (2010-2024)
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.tmdb_collector import TMDBCollector
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def run_hollywood_collection():
    print("=" * 60)
    print("🎬 TMDB Hollywood Collection")
    print("   Target: 1500+ movies (2010-2024)")
    print("=" * 60)
    
    # Initialize
    collector = TMDBCollector()
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    initial_count = movies_collection.count_documents({})
    print(f"✓ Connected to MongoDB (current: {initial_count} movies)")
    
    # Collection config
    years = list(range(2024, 2009, -1))  # 2024 down to 2010
    pages_per_year = 5  # ~100 movies per year
    
    total_fetched = 0
    total_inserted = 0
    
    for year in years:
        print(f"\n📅 Year {year}:")
        year_count = 0
        
        for page in range(1, pages_per_year + 1):
            try:
                # Fetch discover movies sorted by revenue
                result = collector.get_movies_by_year(year, page=page)
                
                if not result or not result.get('results'):
                    print(f"   No more results for page {page}")
                    break
                
                for movie in result['results']:
                    try:
                        # Get full movie details
                        details = collector.get_movie_details(movie['id'])
                        
                        if not details:
                            continue
                        
                        # Process for database
                        processed = collector.process_movie_for_db(details, 'hollywood')
                        
                        if not processed:
                            continue
                        
                        # Add timestamps  
                        processed['createdAt'] = datetime.now()
                        processed['updatedAt'] = datetime.now()
                        
                        # Calculate success category based on ROI
                        if processed.get('budget') and processed.get('revenue') and processed['budget'] > 0:
                            roi = ((processed['revenue'] - processed['budget']) / processed['budget']) * 100
                            if roi >= 400:
                                processed['predictions'] = {'successCategory': 'Blockbuster'}
                            elif roi >= 200:
                                processed['predictions'] = {'successCategory': 'Super Hit'}
                            elif roi >= 100:
                                processed['predictions'] = {'successCategory': 'Hit'}
                            elif roi >= 0:
                                processed['predictions'] = {'successCategory': 'Average'}
                            else:
                                processed['predictions'] = {'successCategory': 'Flop'}
                        
                        # Upsert to MongoDB
                        movies_collection.update_one(
                            {'tmdbId': processed['tmdbId']},
                            {'$set': processed},
                            upsert=True
                        )
                        
                        total_fetched += 1
                        year_count += 1
                        
                        # Progress indicator
                        title = processed['title'][:35]
                        print(f"   ✓ {title}")
                        
                        # Rate limiting - be nice to API
                        time.sleep(0.15)
                        
                    except Exception as e:
                        continue
                
            except Exception as e:
                print(f"   ⚠ Page {page} error: {str(e)[:50]}")
                time.sleep(1)  # Wait longer on errors
                continue
        
        print(f"   → Year {year}: {year_count} movies")
    
    # Final stats
    final_count = movies_collection.count_documents({})
    new_movies = final_count - initial_count
    
    print("\n" + "=" * 60)
    print("✅ TMDB COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"   Movies fetched this session: {total_fetched}")
    print(f"   New movies added: {new_movies}")
    print(f"   Total in database: {final_count}")
    
    # Distribution
    print("\n📊 By Success Category:")
    for cat in ['Blockbuster', 'Super Hit', 'Hit', 'Average', 'Flop']:
        count = movies_collection.count_documents({'predictions.successCategory': cat})
        if count > 0:
            print(f"   {cat}: {count}")
    
    client.close()
    return final_count

if __name__ == '__main__':
    run_hollywood_collection()
