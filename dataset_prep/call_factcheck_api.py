import pandas as pd
import requests
import json
from secrets import API_KEY

BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search?languageCode=en"
liar_test_df = pd.read_csv("LIAR/test.tsv", sep="\t", header=None)
liar_train_df = pd.read_csv("LIAR/train.tsv", sep="\t", header=None)

"""
API Reference:
    -https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search?apix=true&apix_params=%7B%22languageCode%22%3A%22en%22%2C%22maxAgeDays%22%3A12%2C%22pageSize%22%3A10%2C%22query%22%3A%22covid%20vaccine%22%7D
"""


def submit_fakenews_query(**kwargs):
    ENDPOINT = BASE_URL
    for kwarg in kwargs:
        ENDPOINT += f"&{kwarg}={kwargs.get(kwarg)}"
    r = requests.get(f"{ENDPOINT}&key={API_KEY}")
    return json.loads(r.content)


if __name__ == "__main__":
    r = submit_fakenews_query(query="The covid vaccine is toxic", page_size=10)
    print(json.dumps(r, indent=3))

    # claims = liar_train_df[2].tolist()
    # train_api_data = {}
    # for claim in claims:
    #     r = submit_fakenews_query(query=claim, page_size=20)
    #     if r:
    #         train_api_data[claim] = r
    #
    # with open('train_api_data.json', 'w') as f:
    #     json.dump(train_api_data, f)
