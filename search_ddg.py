from duckduckgo_search import DDGS
import sys

query = sys.argv[1]
print(f"Searching for: {query}")
try:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        for r in results:
            print(f"- {r['title']}: {r['href']}")
            print(f"  {r['body']}")
except Exception as e:
    print(f"Error: {e}")
