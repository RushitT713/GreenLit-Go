"""Test TMDB API for Indian movies"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.tmdb_collector import TMDBCollector
from dotenv import load_dotenv

load_dotenv()

collector = TMDBCollector()

print("Testing TMDB API for Indian movies...")
print("=" * 50)

# Test Hindi (Bollywood)
print("\n📽️ Testing Hindi (Bollywood):")
result = collector.get_indian_movies(page=1, language='hi')
if result:
    print(f"   Total results: {result.get('total_results', 0)}")
    print(f"   Total pages: {result.get('total_pages', 0)}")
    movies = result.get('results', [])
    print(f"   Movies on page 1: {len(movies)}")
    for m in movies[:5]:
        print(f"      - {m.get('title')} ({m.get('release_date', 'N/A')[:4] if m.get('release_date') else 'N/A'})")
else:
    print("   ❌ No response")

# Test Telugu (Tollywood)
print("\n📽️ Testing Telugu (Tollywood):")
result = collector.get_indian_movies(page=1, language='te')
if result:
    print(f"   Total results: {result.get('total_results', 0)}")
    movies = result.get('results', [])
    print(f"   Movies on page 1: {len(movies)}")
    for m in movies[:5]:
        print(f"      - {m.get('title')} ({m.get('release_date', 'N/A')[:4] if m.get('release_date') else 'N/A'})")
else:
    print("   ❌ No response")

# Test Tamil (Kollywood) 
print("\n📽️ Testing Tamil (Kollywood):")
result = collector.get_indian_movies(page=1, language='ta')
if result:
    print(f"   Total results: {result.get('total_results', 0)}")
    movies = result.get('results', [])
    print(f"   Movies on page 1: {len(movies)}")
    for m in movies[:5]:
        print(f"      - {m.get('title')} ({m.get('release_date', 'N/A')[:4] if m.get('release_date') else 'N/A'})")
else:
    print("   ❌ No response")
