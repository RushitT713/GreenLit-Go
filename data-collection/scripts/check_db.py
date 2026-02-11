"""Check current database status"""
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/greenlit_go')
db = client.get_database()
movies = db['movies']

print(f"Total movies in database: {movies.count_documents({})}")
print("\nBy Industry:")
for industry in ['hollywood', 'bollywood', 'tollywood', 'kollywood']:
    count = movies.count_documents({'industry': industry})
    if count > 0:
        print(f"  {industry.capitalize()}: {count}")

print("\nSample movies (first 15):")
for m in movies.find().limit(15):
    title = m.get('title', 'Unknown')[:40]
    year = m.get('year', 'N/A')
    industry = m.get('industry', 'N/A')
    print(f"  - {title} ({year}) [{industry}]")

client.close()
