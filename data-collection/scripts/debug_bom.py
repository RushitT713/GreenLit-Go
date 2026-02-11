"""Debug BOM scraper - check actual HTML structure"""

import requests
from bs4 import BeautifulSoup

url = "https://www.boxofficemojo.com/year/2023/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

response = requests.get(url, headers=headers, timeout=15)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table')
if table:
    # Get headers
    headers_row = table.find('tr')
    if headers_row:
        headers = [th.get_text(strip=True) for th in headers_row.find_all(['th', 'td'])]
        print("Table Headers:")
        for i, h in enumerate(headers):
            print(f"  [{i}] {h}")
    
    # Get first data row
    rows = table.find_all('tr')[1:3]
    print("\nFirst 2 rows data:")
    for row in rows:
        cells = row.find_all('td')
        print(f"\nRow with {len(cells)} cells:")
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)[:50]
            print(f"  [{i}] {text}")
