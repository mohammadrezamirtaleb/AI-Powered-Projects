import urllib.request
import json
from pathlib import Path
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import re

def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

def read_file(filepath: str) -> str:
    try:
        path = Path(filepath).resolve()
        workspace = Path(".").resolve()
        if not str(path).startswith(str(workspace)):
            return "Error: Path traversal outside workspace restricted."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Read failed: {str(e)}"

def write_file(filepath: str, content: str) -> str:
    try:
        path = Path(filepath).resolve()
        workspace = Path(".").resolve()
        if not str(path).startswith(str(workspace)):
            return "Error: Path traversal outside workspace restricted."
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File {filepath} written successfully."
    except Exception as e:
        return f"Write failed: {str(e)}"

def get_weather(location: str) -> str:
    try:
        # Using a public unauthenticated API for weather (wttr.in)
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            current = data['current_condition'][0]
            return f"Weather in {location}: {current['temp_C']}°C, {current['weatherDesc'][0]['value']}, Humidity: {current['humidity']}%"
    except Exception as e:
        return f"Could not fetch weather: {str(e)}"

def summarize_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 JARVIS/Mark-86'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=' ', strip=True)
            # Basic cleanup
            text = re.sub(r'\s+', ' ', text)
            return text[:3000] + ("..." if len(text) > 3000 else "")
    except Exception as e:
        return f"Could not fetch URL: {str(e)}"
