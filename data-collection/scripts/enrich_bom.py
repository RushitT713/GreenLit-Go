"""
Box Office Mojo Enrichment Script
Scrapes BOM and enriches existing movies with detailed financial data
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.bom_scraper import BoxOfficeMojoScraper
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from difflib import SequenceMatcher

load_dotenv()

def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_matching_movie(bom_title: str, bom_year: int, movies_collection):
    """Find matching movie in database by title and year"""
    # Try exact match first
    movie = movies_collection.find_one({
        'title': {'$regex': f'^{bom_title}$', '$options': 'i'},
        'year': bom_year
    })
    
    if movie:
        return movie
    
    # Try fuzzy match
    candidates = list(movies_collection.find({'year': bom_year}))
    best_match = None
    best_score = 0
    
    for m in candidates:
        score = similarity(bom_title, m.get('title', ''))
        if score > best_score and score > 0.8:  # 80% similarity threshold
            best_score = score
            best_match = m
    
    return best_match

def run_bom_enrichment():
    print("=" * 60)
    print("🎬 Box Office Mojo Enrichment")
    print("   Scraping years 2010-2024")
    print("=" * 60)
    
    # Initialize
    scraper = BoxOfficeMojoScraper()
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    years = list(range(2024, 2009, -1))  # 2024 down to 2010
    
    total_matched = 0
    total_scraped = 0
    
    for year in years:
        print(f"\n📅 Year {year}:")
        
        try:
            bom_movies = scraper.get_year_movies(year)
            print(f"   Scraped {len(bom_movies)} movies from BOM")
            total_scraped += len(bom_movies)
            
            year_matched = 0
            
            for bom in bom_movies:
                bom_title = bom.get('title', '')
                
                # Find matching movie in our database
                db_movie = find_matching_movie(bom_title, year, movies_collection)
                
                if db_movie:
                    # Update with BOM data
                    movies_collection.update_one(
                        {'_id': db_movie['_id']},
                        {
                            '$set': {
                                'bom': {
                                    'rank': bom.get('bomRank'),
                                    'openingGross': bom.get('openingGross'),
                                    'openingTheaters': bom.get('openingTheaters'),
                                    'totalGross': bom.get('totalGross'),
                                    'distributor': bom.get('distributor'),
                                    'releaseDate': bom.get('bomReleaseDate')
                                },
                                'updatedAt': datetime.now()
                            }
                        }
                    )
                    year_matched += 1
            
            print(f"   Matched {year_matched} movies")
            total_matched += year_matched
            
            # Be nice - don't hammer the server
            time.sleep(1)
            
        except Exception as e:
            print(f"   Error: {str(e)[:50]}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ BOM ENRICHMENT COMPLETE!")
    print("=" * 60)
    print(f"   Total scraped: {total_scraped}")
    print(f"   Total matched: {total_matched}")
    
    # Stats
    with_bom = movies_collection.count_documents({'bom': {'$exists': True}})
    print(f"\n📊 Movies with BOM data: {with_bom}")
    
    client.close()
    return total_matched

if __name__ == '__main__':
    run_bom_enrichment()
