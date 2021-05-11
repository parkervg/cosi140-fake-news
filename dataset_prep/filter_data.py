import pandas as pd
import requests
import json
from secrets import API_KEY

with open("data/test_api_data.json", "r") as f:
    test_api_data = json.load(f)
with open("data/train_api_data.json", "r") as f:
    train_api_data = json.load(f)


def valid_review(review_json):
    if review_json["publisher"]["name"] == "PolitiFact":
        if review_json["textualRating"] != "True":
            return True
    return False


with open("data/annotation_data.json", "r") as f:
    annotation_data = json.load(f)

new_annotation_data = {}
ix = 0
for k, v in annotation_data.items():
    if v["claim"] in train_api_data:
        if train_api_data[v["claim"]]["claims"][0]["claimReview"][0]["textualRating"] != "True":
            new_annotation_data[k] = v
            ix += 1
    if v["claim"] in test_api_data:
        if test_api_data[v["claim"]]["claims"][0]["claimReview"][0]["textualRating"] != "True":
            new_annotation_data[k] = v
            ix += 1

with open("data/annotation_data.json", "w") as f:
    json.dump(new_annotation_data, f)

len(new_annotation_data)

if __name__ == "__main__":
    api_data = dict(test_api_data, **train_api_data)
    valid_claims = {}
    for claim, claim_data in api_data.items():
        if "claims" in claim_data:
            chosen_claim = claim_data["claims"][0]
            for review in chosen_claim["claimReview"]:
                if valid_review(review):
                    valid_claims[claim] = claim_data
