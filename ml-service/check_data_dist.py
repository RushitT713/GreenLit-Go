import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['greenlit_go']
movies = db['movies']

# Count by industry
print("=== MOVIES BY INDUSTRY ===")
for industry in ["hollywood", "bollywood", "tollywood", "kollywood", "mollywood"]:
    count = movies.count_documents({"industry": industry})
    sample = movies.find_one({"industry": industry})
    budget = sample.get("budget", 0) if sample else 0
    revenue = sample.get("revenue", 0) if sample else 0
    print(f"  {industry}: {count} movies")

total = movies.count_documents({})
print(f"  TOTAL: {total}")

# Check field structure of a Hollywood movie
print("\n=== SAMPLE HOLLYWOOD MOVIE FIELDS ===")
hw = movies.find_one({"industry": "hollywood"})
if hw:
    for k, v in hw.items():
        if k == '_id':
            continue
        if isinstance(v, dict):
            print(f"  {k}: (dict with keys: {list(v.keys())})")
        elif isinstance(v, list):
            print(f"  {k}: (list, len={len(v)})")
        else:
            print(f"  {k}: {v}")

# Check field structure of a Bollywood movie
print("\n=== SAMPLE BOLLYWOOD MOVIE FIELDS ===")
bw = movies.find_one({"industry": "bollywood"})
if bw:
    for k, v in bw.items():
        if k == '_id':
            continue
        if isinstance(v, dict):
            print(f"  {k}: (dict with keys: {list(v.keys())})")
        elif isinstance(v, list):
            print(f"  {k}: (list, len={len(v)})")
        else:
            print(f"  {k}: {v}")

# Data completeness comparison
print("\n=== DATA COMPLETENESS ===")
for industry in ["hollywood", "bollywood", "tollywood", "kollywood"]:
    total_ind = movies.count_documents({"industry": industry})
    has_omdb = movies.count_documents({"industry": industry, "omdb": {"$exists": True, "$ne": None}})
    has_yt = movies.count_documents({"industry": industry, "youtube": {"$exists": True, "$ne": None}})
    has_bom = movies.count_documents({"industry": industry, "bom": {"$exists": True, "$ne": None}})
    has_budget = movies.count_documents({"industry": industry, "budget": {"$gt": 0}})
    has_revenue = movies.count_documents({"industry": industry, "revenue": {"$gt": 0}})
    has_votes = movies.count_documents({"industry": industry, "voteCount": {"$gt": 0}})
    
    print(f"\n  {industry.upper()} ({total_ind} movies):")
    print(f"    Has budget>0:    {has_budget}/{total_ind} ({has_budget/total_ind*100:.0f}%)")
    print(f"    Has revenue>0:   {has_revenue}/{total_ind} ({has_revenue/total_ind*100:.0f}%)")
    print(f"    Has voteCount>0: {has_votes}/{total_ind} ({has_votes/total_ind*100:.0f}%)")
    print(f"    Has OMDB data:   {has_omdb}/{total_ind} ({has_omdb/total_ind*100:.0f}%)")
    print(f"    Has YouTube data:{has_yt}/{total_ind} ({has_yt/total_ind*100:.0f}%)")
    print(f"    Has BOM data:    {has_bom}/{total_ind} ({has_bom/total_ind*100:.0f}%)")

client.close()
