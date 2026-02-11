"""
Expanded TMDB Data Collection
Fetches 500+ movies for better ML model training
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

def run_expanded_collection():
    print("=" * 60)
    print("GreenLit Go - Expanded TMDB Data Collection")
    print("Target: 500+ movies for ML training")
    print("=" * 60)
    
    # Initialize
    collector = TMDBCollector()
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    print(f"✓ Connected to MongoDB")
    print(f"  Current movies in DB: {movies_collection.count_documents({})}")
    
    all_movies = []
    
    # =========================================
    # HOLLYWOOD - Multiple years, multiple pages
    # =========================================
    print("\n" + "=" * 40)
    print("📽️  HOLLYWOOD MOVIES")
    print("=" * 40)
    
    hollywood_years = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]
    
    for year in hollywood_years:
        print(f"\n📅 Year {year}:")
        for page in range(1, 4):  # 3 pages per year = ~60 movies/year
            try:
                result = collector.get_movies_by_year(year, page=page)
                if not result or not result.get('results'):
                    break
                    
                for movie in result['results']:
                    try:
                        # Only fetch movies with budget info (quality filter)
                        details = collector.get_movie_details(movie['id'])
                        if details and details.get('revenue', 0) > 0:
                            processed = collector.process_movie_for_db(details, 'hollywood')
                            if processed:
                                all_movies.append(processed)
                                print(f"   ✓ {processed['title'][:40]}")
                        time.sleep(0.1)  # Rate limiting
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"   ✗ Error page {page}: {e}")
                continue
    
    print(f"\n  Hollywood total: {len(all_movies)} movies")
    
    # =========================================
    # BOLLYWOOD - Hindi movies
    # =========================================
    print("\n" + "=" * 40)
    print("🎬 BOLLYWOOD MOVIES (Hindi)")
    print("=" * 40)
    
    bollywood_count = 0
    for page in range(1, 6):  # 5 pages
        try:
            result = collector.get_indian_movies(page=page, language='hi')
            if not result or not result.get('results'):
                break
                
            for movie in result['results']:
                try:
                    details = collector.get_movie_details(movie['id'])
                    if details:
                        processed = collector.process_movie_for_db(details, 'bollywood')
                        if processed:
                            all_movies.append(processed)
                            bollywood_count += 1
                            print(f"   ✓ {processed['title'][:40]}")
                    time.sleep(0.1)
                except:
                    continue
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    print(f"\n  Bollywood total: {bollywood_count} movies")
    
    # =========================================
    # TOLLYWOOD - Telugu movies
    # =========================================
    print("\n" + "=" * 40)
    print("🎥 TOLLYWOOD MOVIES (Telugu)")
    print("=" * 40)
    
    tollywood_count = 0
    for page in range(1, 4):  # 3 pages
        try:
            result = collector.get_indian_movies(page=page, language='te')
            if not result or not result.get('results'):
                break
                
            for movie in result['results']:
                try:
                    details = collector.get_movie_details(movie['id'])
                    if details:
                        processed = collector.process_movie_for_db(details, 'tollywood')
                        if processed:
                            all_movies.append(processed)
                            tollywood_count += 1
                            print(f"   ✓ {processed['title'][:40]}")
                    time.sleep(0.1)
                except:
                    continue
        except Exception as e:
            continue
    
    print(f"\n  Tollywood total: {tollywood_count} movies")
    
    # =========================================
    # KOLLYWOOD - Tamil movies
    # =========================================
    print("\n" + "=" * 40)
    print("🎞️  KOLLYWOOD MOVIES (Tamil)")
    print("=" * 40)
    
    kollywood_count = 0
    for page in range(1, 3):  # 2 pages
        try:
            result = collector.get_indian_movies(page=page, language='ta')
            if not result or not result.get('results'):
                break
                
            for movie in result['results']:
                try:
                    details = collector.get_movie_details(movie['id'])
                    if details:
                        processed = collector.process_movie_for_db(details, 'kollywood')
                        if processed:
                            all_movies.append(processed)
                            kollywood_count += 1
                            print(f"   ✓ {processed['title'][:40]}")
                    time.sleep(0.1)
                except:
                    continue
        except Exception as e:
            continue
    
    print(f"\n  Kollywood total: {kollywood_count} movies")
    
    # =========================================
    # INSERT INTO MONGODB
    # =========================================
    print("\n" + "=" * 60)
    print(f"💾 INSERTING {len(all_movies)} MOVIES INTO MONGODB")
    print("=" * 60)
    
    inserted_count = 0
    updated_count = 0
    
    for movie in all_movies:
        movie['createdAt'] = datetime.now()
        movie['updatedAt'] = datetime.now()
        
        # Calculate success category
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
        
        try:
            result = movies_collection.update_one(
                {'tmdbId': movie['tmdbId']},
                {'$set': movie},
                upsert=True
            )
            if result.upserted_id:
                inserted_count += 1
            else:
                updated_count += 1
        except Exception as e:
            continue
    
    print(f"\n✓ New movies inserted: {inserted_count}")
    print(f"✓ Existing movies updated: {updated_count}")
    
    # Final count
    final_count = movies_collection.count_documents({})
    
    print(f"\n" + "=" * 60)
    print(f"✅ DATA COLLECTION COMPLETE!")
    print(f"   Total movies in database: {final_count}")
    print("=" * 60)
    
    # Distribution by industry
    print("\n📊 Movies by Industry:")
    for industry in ['hollywood', 'bollywood', 'tollywood', 'kollywood']:
        count = movies_collection.count_documents({'industry': industry})
        print(f"   {industry.capitalize()}: {count}")
    
    client.close()

if __name__ == '__main__':
    run_expanded_collection()
