# FAKE NEWS SURVEY FLASK SIDE

from flask import Flask, render_template, request, send_from_directory, make_response
import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

creds = ServiceAccountCredentials.from_json_keyfile_name("REASONCreds.json", scope)
client = gspread.authorize(creds)
file = client.open_by_key("1JKuCr-mROfbXS14tR-oXbGuxDY7RL61J5kx2cX1OpjY")
sheet = file.sheet1


app = Flask(__name__)

DATA = None


def ready_data(path):
    global DATA
    with open(path) as j:
        DATA = json.load(j)


@app.route("/icon.jpg", methods=["POST", "GET"])
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "icon.jpg")


@app.route("/", methods=["POST", "GET"])
def home():
    global DATA
    ready_data(os.path.join(app.root_path, "static", "annotation_data2.json"))
    return render_template("home.html")


@app.route("/user", methods=["POST", "GET"])
def annotator():
    global DATA
    ready_data(os.path.join(app.root_path, "static", "annotation_data2.json"))
    if request.form["name"]:
        lastDAT = 0
        allcodes = sheet.col_values(12)
        name = request.form["name"].upper()
        strip = [int(c[len(name + ",") :]) for c in allcodes if name + "," in c]
        if strip:
            lastDAT = max(strip) + 1
        data = DATA[str(lastDAT)]  # fetch data
        return render_template("FAKENEWSSURVEY.html", n=name, data=data, p=lastDAT)
    else:
        return render_template("home.html")


@app.route("/user/<name>/<int:page_id>", methods=["POST", "GET"])
def newpage(page_id, name):
    global DATA
    ready_data(os.path.join(app.root_path, "static", "annotation_data2.json"))
    if request.method == "POST":
        results = request.form.getlist("annotate")
        # if an empty annotation, return last page again
        if results == [""]:
            return render_template(
                "FAKENEWSSURVEY.html",
                n=name,
                data=DATA[str(page_id - 1)],
                p=page_id - 1,
            )
        else:
            tmprow = [
                name,
                page_id - 1,
                "contradictory quote" in results,
                "exaggeration" in results,
                "quantitative data" in results,
                "evidence lacking" in results,
                "dubious reference" in results,
                "out of context" in results,
                "qualitative data" in results,
                "click" in results,
                "",
            ]
            if "other" in results:
                tmprow[10] = results[-1]

            sheet.append_row(tmprow)
            return render_template(
                "FAKENEWSSURVEY.html", n=name, data=DATA[str(page_id)], p=page_id
            )
    else:
        return render_template(
            "FAKENEWSSURVEY.html", n=name, data=DATA[str(page_id)], p=page_id
        )


if __name__ == "__main__":
    ready_data(os.path.join(app.root_path, "static", "annotation_data2.json"))
    app.run(threaded=True, port=5000)
