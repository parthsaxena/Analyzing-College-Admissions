import json
import csv

with open('nyu.json') as ucsd:
  data = json.load(ucsd)

updated = []
scores = data["scores"]

for score in scores:
    score.pop('satConcorded', None)
    score.pop('sat2400', None)
    score.pop('legacyStudentId', None)
    score.pop('type', None)
    score.pop('act', None)
    score.pop('deferred', None)
    score.pop('waitlisted', None)
    score.pop('gpaCumulative', None)
    if score["resultCode"] == 1:
        score["accepted"] = 1
        score.pop("resultCode")
        if score["sat1600"] != 0:
            updated.append(score)
    elif score["resultCode"] == 2 or score["resultCode"] == 0:
          score["accepted"] = 0
          score.pop("resultCode")
          if score["sat1600"] != 0:
              updated.append(score)

with open('output_nyu.csv', 'w') as data_file:
    fieldnames = ['sat1600', 'gpaWeighted', 'accepted']
    writer = csv.DictWriter(data_file, fieldnames=fieldnames)
    writer.writeheader()

    for score in updated:
        writer.writerow(score)
