"""
YouTube Enrichment Script
Adds trailer view counts to movies
FREE TIER LIMIT: ~100 movies per day (10,000 quota / 101 per search)
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.youtube_collector import YouTubeCollector
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Free tier limit: 10,000 units/day, each movie costs ~101 units
MAX_MOVIES_PER_RUN = 95  # Stay under limit

def run_youtube_enrichment():
    print("=" * 60)
    print("🎬 YouTube Trailer Enrichment")
    print(f"   FREE TIER: Processing up to {MAX_MOVIES_PER_RUN} movies")
    print("=" * 60)
    
    # Initialize
    collector = YouTubeCollector()
    mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/greenlit_go')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    movies_collection = db['movies']
    
    # Get movies without YouTube data (prioritize by revenue)
    movies_to_enrich = list(movies_collection.find({
        'youtube': {'$exists': False}
    }).sort('revenue', -1).limit(MAX_MOVIES_PER_RUN))
    
    total = len(movies_to_enrich)
    print(f"✓ Found {total} movies to enrich (of {MAX_MOVIES_PER_RUN} max)")
    
    enriched_count = 0
    failed_count = 0
    
    for i, movie in enumerate(movies_to_enrich):
        title = movie.get('title', 'Unknown')
        year = movie.get('year')
        
        # Progress
        print(f"[{i+1}/{total}] {title[:40]}...", end=" ")
        
        try:
            trailer_data = collector.get_trailer_data(title, year)
            
            if trailer_data:
                # Update movie in database
                movies_collection.update_one(
                    {'_id': movie['_id']},
                    {
                        '$set': {
                            'youtube': {
                                'videoId': trailer_data.get('videoId'),
                                'viewCount': trailer_data.get('viewCount'),
                                'likeCount': trailer_data.get('likeCount'),
                                'commentCount': trailer_data.get('commentCount'),
                                'trailerTitle': trailer_data.get('title'),
                                'fetchedAt': datetime.now()
                            },
                            'updatedAt': datetime.now()
                        }
                    }
                )
                
                views = trailer_data.get('viewCount', 0)
                print(f"✓ {views:,} views")
                enriched_count += 1
            else:
                print("✗ Not found")
                failed_count += 1
                
            # Rate limiting - be nice to API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:30]}")
            failed_count += 1
            continue
    
    print("\n" + "=" * 60)
    print("✅ YOUTUBE ENRICHMENT COMPLETE!")
    print("=" * 60)
    print(f"   Movies enriched: {enriched_count}")
    print(f"   Not found: {failed_count}")
    
    # Stats
    total_with_yt = movies_collection.count_documents({'youtube': {'$exists': True}})
    remaining = movies_collection.count_documents({'youtube': {'$exists': False}})
    
    print(f"\n📊 Coverage:")
    print(f"   With YouTube data: {total_with_yt}")
    print(f"   Remaining: {remaining}")
    print(f"\n💡 Run again tomorrow to enrich more movies (free tier limit)")
    
    client.close()
    return enriched_count

if __name__ == '__main__':
    run_youtube_enrichment()
