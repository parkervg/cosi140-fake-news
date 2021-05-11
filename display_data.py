import json

with open("data/test_api_data.json", "r") as f:
    j = json.load(f)

print(json.dumps(j, indent=2))
