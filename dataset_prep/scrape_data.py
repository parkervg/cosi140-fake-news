import json
from bs4 import BeautifulSoup
import requests
import re

SUMMARY_CUES = ["Our rating", "Our ruling"]

with open("data/annotation_data.json", "r") as f:
    annotation_data = json.load(f)

with open("data/valid_claims.json", "r") as f:
    api_data = json.load(f)

claim_ix = 0
annotation_data = {}
for claim, data in api_data.items():
    url = data["claims"][0]["claimReview"][0]["url"]
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    strong_tags = [i.find("strong") for i in soup.findAll("p") if i.find("strong")]
    our_ruling = [i for i in strong_tags if i.text.strip() in SUMMARY_CUES]
    if not our_ruling:
        print("No summary for this PolitiFact post")
        print(url)
        print()
    else:
        our_ruling = our_ruling[0]
        summary_text = []
        current_p = our_ruling.findNext("p")
        if re.search(r"<a href=", current_p.decode()):  # It's a bad summary
            # The actual summary isn't in p tags
            valid_paragraphs = 0
            x = our_ruling.next
            while valid_paragraphs < 3:
                x = x.next
                if str(x).replace("\n", "").strip() and not str(x).startswith("<"):
                    summary_text.append(str(x))
                    valid_paragraphs += 1
        else:
            for _ in range(3):  # Assumption made that summary is 3 paragraphs
                summary_text.append(current_p.decode())
                current_p = current_p.findNext("p")
        annotation_data[claim_ix] = {}
        annotation_data[claim_ix]["summary"] = " ".join(summary_text)
        annotation_data[claim_ix]["claim"] = claim
        annotation_data[claim_ix]["verdict_url"] = url
        claim_ix += 1
