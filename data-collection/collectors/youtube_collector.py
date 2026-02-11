"""
YouTube API Collector
Fetches trailer view counts and engagement metrics
FREE TIER: 10,000 quota units/day
- Search costs 100 units, Video stats costs 1 unit
- We can do ~100 searches + stats per day
"""

import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

load_dotenv()

class YouTubeCollector:
    """Collects trailer data from YouTube API v3"""
    
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found in environment")
        
        print(f"✓ YouTube Collector initialized")
    
    def search_trailer(self, movie_title: str, year: int = None) -> Optional[str]:
        """
        Search for official movie trailer and return video ID
        Costs: 100 quota units per search
        """
        try:
            query = f"{movie_title} official trailer"
            if year:
                query += f" {year}"
            
            params = {
                'key': self.api_key,
                'q': query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 1,
                'order': 'relevance'
            }
            
            response = requests.get(f"{self.base_url}/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            items = data.get('items', [])
            if items:
                return items[0]['id']['videoId']
            return None
            
        except Exception as e:
            print(f"   Search error: {str(e)[:50]}")
            return None
    
    def get_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video statistics (views, likes, comments)
        Costs: 1 quota unit per call
        """
        try:
            params = {
                'key': self.api_key,
                'id': video_id,
                'part': 'statistics,snippet'
            }
            
            response = requests.get(f"{self.base_url}/videos", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            items = data.get('items', [])
            if items:
                stats = items[0].get('statistics', {})
                snippet = items[0].get('snippet', {})
                
                return {
                    'videoId': video_id,
                    'title': snippet.get('title'),
                    'channelTitle': snippet.get('channelTitle'),
                    'viewCount': self._parse_int(stats.get('viewCount')),
                    'likeCount': self._parse_int(stats.get('likeCount')),
                    'commentCount': self._parse_int(stats.get('commentCount')),
                    'publishedAt': snippet.get('publishedAt')
                }
            return None
            
        except Exception as e:
            print(f"   Stats error: {str(e)[:50]}")
            return None
    
    def get_trailer_data(self, movie_title: str, year: int = None) -> Optional[Dict[str, Any]]:
        """
        Search for trailer and get its stats in one call
        Total cost: ~101 quota units
        """
        video_id = self.search_trailer(movie_title, year)
        if video_id:
            return self.get_video_stats(video_id)
        return None
    
    def _parse_int(self, value: str) -> Optional[int]:
        """Parse string to int"""
        if not value:
            return None
        try:
            return int(value)
        except:
            return None


if __name__ == '__main__':
    # Test
    collector = YouTubeCollector()
    result = collector.get_trailer_data("Inception", 2010)
    if result:
        print(f"Found: {result.get('title')}")
        print(f"Views: {result.get('viewCount'):,}")
        print(f"Likes: {result.get('likeCount'):,}")
