from os import TMP_MAX
import plotter
import json
import sys

json_path = sys.argv[1]
print(json_path)

# Opening JSON file
f = open(json_path)

# returns JSON object as a dictionary
json_content = json.load(f)

# Closing file
f.close()

metrics = list(json_content["genuine_acceptances"])
print(metrics)

models_temp_key = list(json_content["genuine_acceptances"])[0]
models = list(json_content["genuine_acceptances"][models_temp_key])
print(models)

thresholds = list(json_content["genuine_acceptances"][metrics[0]][models[0]])
print(thresholds)


# threshold_FAR_FRR
# for metric in metrics:

#   for model in models: