from googlesearch import search
import sys

query = sys.argv[1]
print(f"Searching for: {query}")
for j in search(query, num_results=5):
    print(j)
