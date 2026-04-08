"""
Indian Cinema Expanded Collection — Unified Script
Collects Bollywood, Tollywood, Kollywood, and Mollywood movies from TMDB.
Targets 250-500 movies per industry using increased page limits.
Uses upsert logic to avoid duplicates.
"""

import os
import sys
import time
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load env from the data-collection directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
MONGODB_URI = os.getenv('MONGODB_URI')

# ─── Configuration ───────────────────────────────────────────
INDUSTRIES = [
    {'lang': 'hi', 'industry': 'bollywood', 'label': 'Bollywood (Hindi)',    'pages': 25},
    {'lang': 'te', 'industry': 'tollywood', 'label': 'Tollywood (Telugu)',   'pages': 15},
    {'lang': 'ta', 'industry': 'kollywood', 'label': 'Kollywood (Tamil)',    'pages': 15},
    {'lang': 'ml', 'industry': 'mollywood', 'label': 'Mollywood (Malayalam)','pages': 15},
]

# We collect movies using multiple sort strategies to maximize coverage
SORT_STRATEGIES = [
    'revenue.desc',
    'popularity.desc',
    'vote_count.desc',
    'primary_release_date.desc',
]


def fetch_discover(language, page, sort_by='revenue.desc', year_gte=None, year_lte=None):
    """Fetch movies from TMDB discover endpoint."""
    params = {
        'api_key': TMDB_API_KEY,
        'with_original_language': language,
        'sort_by': sort_by,
        'page': page,
        'vote_count.gte': 5,  # Filter out obscure entries with almost no ratings
    }
    if year_gte:
        params['primary_release_date.gte'] = f'{year_gte}-01-01'
    if year_lte:
        params['primary_release_date.lte'] = f'{year_lte}-12-31'

    try:
        resp = requests.get(f'{TMDB_BASE_URL}/discover/movie', params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print('      ⏳ Rate limited, waiting 10s...')
            time.sleep(10)
            return fetch_discover(language, page, sort_by, year_gte, year_lte)
        else:
            print(f'      ⚠ Discover API returned {resp.status_code}')
            return None
    except Exception as e:
        print(f'      ⚠ Request error: {e}')
        return None


def fetch_movie_details(movie_id):
    """Fetch full movie details including credits, videos, and keywords."""
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,videos,keywords',
    }
    try:
        resp = requests.get(f'{TMDB_BASE_URL}/movie/{movie_id}', params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print('      ⏳ Rate limited, waiting 10s...')
            time.sleep(10)
            return fetch_movie_details(movie_id)
        return None
    except Exception as e:
        print(f'      ⚠ Details error for {movie_id}: {e}')
        return None


def process_movie(movie, industry):
    """Process raw TMDB movie data into our database schema."""
    try:
        credits = movie.get('credits', {})
        videos = movie.get('videos', {}).get('results', [])

        # Director
        director = None
        for crew in credits.get('crew', []):
            if crew.get('job') == 'Director':
                director = {
                    'id': crew.get('id'),
                    'name': crew.get('name'),
                    'popularity': crew.get('popularity', 0),
                }
                break

        # Top 10 cast
        cast = []
        for actor in credits.get('cast', [])[:10]:
            cast.append({
                'id': actor.get('id'),
                'name': actor.get('name'),
                'character': actor.get('character'),
                'order': actor.get('order'),
                'popularity': actor.get('popularity', 0),
                'profilePath': actor.get('profile_path'),
                'gender': actor.get('gender'),
            })

        # Writers
        writers = []
        for crew in credits.get('crew', []):
            if crew.get('department') == 'Writing':
                writers.append({
                    'id': crew.get('id'),
                    'name': crew.get('name'),
                    'job': crew.get('job'),
                })

        # Trailer
        trailer = None
        for video in videos:
            if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
                trailer = {
                    'key': video.get('key'),
                    'url': f"https://www.youtube.com/watch?v={video.get('key')}",
                }
                break

        # Genres
        genres = [g.get('name') for g in movie.get('genres', [])]

        # Production companies
        production_companies = []
        for company in movie.get('production_companies', [])[:5]:
            production_companies.append({
                'id': company.get('id'),
                'name': company.get('name'),
                'logoPath': company.get('logo_path'),
                'originCountry': company.get('origin_country'),
            })

        # Sequel check
        is_sequel = movie.get('belongs_to_collection') is not None

        # Release date & year
        release_date = movie.get('release_date')
        year = None
        if release_date and len(release_date) >= 4:
            year = int(release_date[:4])

        return {
            'tmdbId': movie.get('id'),
            'imdbId': movie.get('imdb_id'),
            'title': movie.get('title'),
            'originalTitle': movie.get('original_title'),
            'overview': movie.get('overview'),
            'tagline': movie.get('tagline'),
            'industry': industry,
            'releaseDate': release_date,
            'year': year,
            'status': movie.get('status'),
            'budget': movie.get('budget'),
            'revenue': movie.get('revenue'),
            'runtime': movie.get('runtime'),
            'genres': genres,
            'originalLanguage': movie.get('original_language'),
            'spokenLanguages': [l.get('english_name') for l in movie.get('spoken_languages', [])],
            'countries': [c.get('name') for c in movie.get('production_countries', [])],
            'adult': movie.get('adult', False),
            'productionCompanies': production_companies,
            'isSequel': is_sequel,
            'partOfCollection': movie.get('belongs_to_collection'),
            'cast': cast,
            'director': director,
            'writers': writers,
            'voteAverage': movie.get('vote_average'),
            'voteCount': movie.get('vote_count'),
            'popularity': movie.get('popularity'),
            'posterPath': movie.get('poster_path'),
            'backdropPath': movie.get('backdrop_path'),
            'trailerUrl': trailer.get('url') if trailer else None,
            'trailerKey': trailer.get('key') if trailer else None,
            'keywords': [k.get('name') for k in movie.get('keywords', {}).get('keywords', [])],
            'updatedAt': datetime.now(),
        }
    except Exception as e:
        print(f'      ✗ Processing error: {e}')
        return None


def collect_industry(collection, lang, industry, label, max_pages):
    """Collect movies for a single industry using multiple sort strategies."""
    print(f'\n{"=" * 60}')
    print(f'🎬  {label}')
    print(f'{"=" * 60}')

    before_count = collection.count_documents({'industry': industry})
    print(f'   Current {label} count in DB: {before_count}')

    seen_ids = set()
    collected = 0
    skipped = 0

    for sort_by in SORT_STRATEGIES:
        print(f'\n   📊 Strategy: {sort_by}')

        for page in range(1, max_pages + 1):
            result = fetch_discover(lang, page, sort_by)
            if not result or not result.get('results'):
                break

            movies = result.get('results', [])
            total_pages = min(result.get('total_pages', 1), 500)  # TMDB caps at 500

            if page > total_pages:
                break

            for m in movies:
                tmdb_id = m.get('id')
                if tmdb_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(tmdb_id)

                details = fetch_movie_details(tmdb_id)
                if not details:
                    continue

                processed = process_movie(details, industry)
                if not processed:
                    continue

                try:
                    collection.update_one(
                        {'tmdbId': processed['tmdbId']},
                        {'$set': processed, '$setOnInsert': {'createdAt': datetime.now()}},
                        upsert=True,
                    )
                    collected += 1
                    title = processed['title'][:45]
                    yr = processed.get('year', '?')
                    print(f'      ✓ [{collected}] {title} ({yr})')
                except Exception as e:
                    print(f'      ✗ DB error: {e}')

                time.sleep(0.2)  # Rate limiting

            # Progress
            print(f'      Page {page}/{total_pages} done — {collected} new so far')

    after_count = collection.count_documents({'industry': industry})
    print(f'\n   ✅ {label} complete!')
    print(f'      New movies collected: {collected} | Duplicates skipped: {skipped}')
    print(f'      DB count: {before_count} → {after_count}')

    return collected


def main():
    print('\n' + '=' * 60)
    print('🇮🇳  INDIAN CINEMA EXPANDED COLLECTION')
    print('=' * 60)

    if not TMDB_API_KEY:
        print('❌ TMDB_API_KEY not found in .env')
        sys.exit(1)

    if not MONGODB_URI:
        print('❌ MONGODB_URI not found in .env')
        sys.exit(1)

    # Connect to MongoDB Atlas
    print(f'\n🔗 Connecting to MongoDB Atlas...')
    client = MongoClient(MONGODB_URI)
    db = client.get_database()
    movies_collection = db['movies']

    initial_total = movies_collection.count_documents({})
    print(f'   ✓ Connected! Total movies in DB: {initial_total}')

    # Print current breakdown
    print('\n📊 Current Industry Breakdown:')
    for cfg in INDUSTRIES:
        count = movies_collection.count_documents({'industry': cfg['industry']})
        print(f'   {cfg["label"]}: {count}')

    # Collect each industry
    grand_total = 0
    start_time = time.time()

    for cfg in INDUSTRIES:
        count = collect_industry(
            movies_collection,
            cfg['lang'],
            cfg['industry'],
            cfg['label'],
            cfg['pages'],
        )
        grand_total += count

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    # Final summary
    final_total = movies_collection.count_documents({})
    print('\n' + '=' * 60)
    print('🏁  COLLECTION COMPLETE!')
    print('=' * 60)
    print(f'   Total new movies collected: {grand_total}')
    print(f'   Database: {initial_total} → {final_total} movies')
    print(f'   Time elapsed: {elapsed_min:.1f} minutes')

    print('\n📊 Final Industry Breakdown:')
    for ind in ['hollywood', 'bollywood', 'tollywood', 'kollywood', 'mollywood']:
        c = movies_collection.count_documents({'industry': ind})
        print(f'   {ind.capitalize()}: {c}')

    client.close()
    print('\n✅ Done! Database connection closed.')


if __name__ == '__main__':
    main()
