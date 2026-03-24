import urllib.request
import json

url = "http://127.0.0.1:8000/search"

def test_query(query):
    data = json.dumps({"query": query, "top_k": 5}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as f:
        response = f.read().decode('utf-8')
        print(f"Query: {query}")
        print(f"Response: {json.dumps(json.loads(response), indent=2)}")

test_query("red color")
test_query("green color")
