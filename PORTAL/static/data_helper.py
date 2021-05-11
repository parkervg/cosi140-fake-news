import json

with open("annotation_data.json") as j:
    jj = json.load(j)
print(len(jj))
newj = {}
oldj = jj.items()
print(len(oldj))
i = 0
for item in oldj:
    newj[i] = item[1]
    i += 1

with open("annotation_data2.json", "w") as w:
    json.dump(newj, w)
