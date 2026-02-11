"""
Box Office Mojo Scraper
Scrapes detailed box office data (Opening Weekend, Theaters, Distributor)
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
import re
import time

class BoxOfficeMojoScraper:
    """Scrapes movie financial data from Box Office Mojo"""
    
    def __init__(self):
        self.base_url = "https://www.boxofficemojo.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        print("✓ Box Office Mojo Scraper initialized")
    
    def get_year_movies(self, year: int) -> List[Dict[str, Any]]:
        """Scrape all movies from a specific year"""
        url = f"{self.base_url}/year/{year}/"
        movies = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the main table
            table = soup.find('table')
            if not table:
                print(f"   No table found for year {year}")
                return movies
            
            # Get all rows (skip header)
            rows = table.find_all('tr')[1:]
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 7:
                    try:
                        movie = self._parse_row(cells, year)
                        if movie:
                            movies.append(movie)
                    except Exception as e:
                        continue
            
            return movies
            
        except Exception as e:
            print(f"   Error fetching year {year}: {str(e)[:50]}")
            return movies
    
    def _parse_row(self, cells, year: int) -> Optional[Dict[str, Any]]:
        """Parse a table row into movie data"""
        try:
            # BOM has 11 columns: Rank, Release, Genre, Budget, Running Time, 
            # Gross, Theaters, Total Gross, Release Date, Distributor, Estimated
            
            if len(cells) < 10:
                return None
            
            # Extract text from cells
            rank = self._clean_number(cells[0].get_text(strip=True))
            
            # Title (may have a link)
            title_cell = cells[1]
            title_link = title_cell.find('a')
            title = title_link.get_text(strip=True) if title_link else title_cell.get_text(strip=True)
            movie_url = title_link.get('href') if title_link else None
            
            # Financial data - correct indices
            genre = cells[2].get_text(strip=True)
            budget = self._parse_money(cells[3].get_text(strip=True))
            runtime = cells[4].get_text(strip=True)
            opening_gross = self._parse_money(cells[5].get_text(strip=True))
            theaters = self._clean_number(cells[6].get_text(strip=True))
            total_gross = self._parse_money(cells[7].get_text(strip=True))
            release_date = cells[8].get_text(strip=True)
            distributor = cells[9].get_text(strip=True)
            
            return {
                'bomRank': rank,
                'title': title,
                'bomUrl': movie_url,
                'bomGenre': genre if genre != '-' else None,
                'bomBudget': budget,
                'bomRuntime': runtime if runtime != '-' else None,
                'openingGross': opening_gross,
                'openingTheaters': theaters,
                'totalGross': total_gross,
                'bomReleaseDate': release_date,
                'distributor': distributor,
                'year': year
            }
            
        except Exception as e:
            return None
    
    def _parse_money(self, text: str) -> Optional[int]:
        """Parse money string like '$1,234,567' to integer"""
        if not text or text == '-':
            return None
        # Remove $ and commas
        cleaned = re.sub(r'[,$]', '', text)
        try:
            return int(cleaned)
        except:
            return None
    
    def _clean_number(self, text: str) -> Optional[int]:
        """Parse number string, removing commas"""
        if not text or text == '-':
            return None
        cleaned = re.sub(r'[,]', '', text)
        try:
            return int(cleaned)
        except:
            return None


if __name__ == '__main__':
    # Test
    scraper = BoxOfficeMojoScraper()
    movies = scraper.get_year_movies(2023)
    print(f"\nFound {len(movies)} movies for 2023")
    if movies:
        print("\nSample movies:")
        for m in movies[:5]:
            gross = m.get('totalGross') or 0
            print(f"  {m.get('bomRank', '?')}. {m['title']} - ${gross:,}")
