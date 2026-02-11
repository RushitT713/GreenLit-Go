"""
Run TMDB Data Collection - Quick version
Fetches a subset of movies to populate the database
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.tmdb_collector import TMDBCollector
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def run_collection():
    print("=" * 50)
    print("GreenLit Go - TMDB Data Collection")
    print("=" * 50)
    
    # Initialize collector
    collector = TMDBCollector()
    print("✓ TMDB Collector initialized")
    
    # Connect to MongoDB
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    print(f"✓ Connected to MongoDB")
    
    all_movies = []
    
    # Collect Hollywood movies (top popular from recent years)
    print("\n📽️  Collecting Hollywood movies...")
    for year in [2024, 2023, 2022, 2021]:
        print(f"  Fetching year {year}...")
        try:
            result = collector.get_movies_by_year(year, page=1)
            if result and result.get('results'):
                for movie in result['results'][:15]:  # Top 15 per year
                    try:
                        details = collector.get_movie_details(movie['id'])
                        if details and details.get('budget', 0) > 0:
                            processed = collector.process_movie_for_db(details, 'hollywood')
                            if processed:
                                all_movies.append(processed)
                                print(f"    ✓ {processed['title']} ({processed['year']})")
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
        except Exception as e:
            print(f"  ✗ Error fetching {year}: {e}")
    
    print(f"\n  Total Hollywood: {len(all_movies)} movies")
    
    # Collect Bollywood movies
    print("\n🎬 Collecting Bollywood movies...")
    try:
        result = collector.get_indian_movies(page=1, language='hi')
        if result and result.get('results'):
            for movie in result['results'][:20]:
                try:
                    details = collector.get_movie_details(movie['id'])
                    if details:
                        processed = collector.process_movie_for_db(details, 'bollywood')
                        if processed:
                            all_movies.append(processed)
                            print(f"    ✓ {processed['title']}")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Collect Telugu movies
    print("\n🎥 Collecting Tollywood movies...")
    try:
        result = collector.get_indian_movies(page=1, language='te')
        if result and result.get('results'):
            for movie in result['results'][:10]:
                try:
                    details = collector.get_movie_details(movie['id'])
                    if details:
                        processed = collector.process_movie_for_db(details, 'tollywood')
                        if processed:
                            all_movies.append(processed)
                            print(f"    ✓ {processed['title']}")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"Total collected: {len(all_movies)} movies")
    
    # Insert into MongoDB
    if all_movies:
        print("\n💾 Inserting into MongoDB...")
        
        inserted_count = 0
        for movie in all_movies:
            movie['createdAt'] = datetime.now()
            movie['updatedAt'] = datetime.now()
            
            # Calculate success category based on ROI if budget and revenue exist
            if movie.get('budget') and movie.get('revenue') and movie['budget'] > 0:
                roi = ((movie['revenue'] - movie['budget']) / movie['budget']) * 100
                if roi >= 400:
                    movie['predictions'] = {'successCategory': 'Blockbuster'}
                elif roi >= 200:
                    movie['predictions'] = {'successCategory': 'Super Hit'}
                elif roi >= 100:
                    movie['predictions'] = {'successCategory': 'Hit'}
                elif roi >= 0:
                    movie['predictions'] = {'successCategory': 'Average'}
                else:
                    movie['predictions'] = {'successCategory': 'Flop'}
            
            # Upsert to handle duplicates
            try:
                movies_collection.update_one(
                    {'tmdbId': movie['tmdbId']},
                    {'$set': movie},
                    upsert=True
                )
                inserted_count += 1
            except Exception as e:
                print(f"  ✗ Error inserting {movie.get('title')}: {e}")
        
        print(f"✓ Inserted/Updated {inserted_count} movies")
    
    # Create indexes (with error handling for existing indexes)
    print("\n📇 Creating indexes...")
    try:
        # Drop existing indexes first (except _id)
        existing_indexes = movies_collection.index_information()
        for index_name in existing_indexes:
            if index_name != '_id_':
                try:
                    movies_collection.drop_index(index_name)
                except:
                    pass
        
        # Create fresh indexes
        movies_collection.create_index('tmdbId', unique=True, sparse=True)
        movies_collection.create_index('title')
        movies_collection.create_index('industry')
        movies_collection.create_index('year')
        movies_collection.create_index('genres')
        print("✓ Indexes created")
    except Exception as e:
        print(f"  Index warning (not critical): {e}")
    
    print(f"\n{'=' * 50}")
    print("✅ Data collection complete!")
    print(f"Database now has {movies_collection.count_documents({})} movies")
    print("=" * 50)
    
    client.close()

if __name__ == '__main__':
    run_collection()
