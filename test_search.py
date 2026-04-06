import sys
from src.retrieval.search import SearchEngine
engine = SearchEngine()
try:
    results = engine.search("two guys sitting and talking in rain")
    print(results)
except Exception as e:
    print("Error:", e)
