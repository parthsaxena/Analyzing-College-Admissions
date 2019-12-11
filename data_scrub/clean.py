import json

with open('uiuc.json') as ucsd:
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
    if score["resultCode"] == 1:
        score["accepted"] = 1
        #score.pop("resultCode")
        if score["sat1600"] != 0:
            updated.append(score)
    elif score["resultCode"] == 2 or score["resultCode"] == 0:
          score["accepted"] = 0
          #score.pop("resultCode")
          if score["sat1600"] != 0:
              updated.append(score)

with open('output_uiuc.json', 'w') as data_file:
    data = json.dump(updated, data_file)
