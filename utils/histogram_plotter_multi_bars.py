import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np

dict = {
    "VGG-Face": {
        "correct": 926,
        "wrong": 612
    },

    "OpenFace": {
        "correct": 923,
        "wrong": 615
    },

    "Facenet": {
        "correct": 924,
        "wrong": 614
    },

    "Facenet512": {
        "correct": 925,
        "wrong": 613
    },

    "DeepFace": {
        "correct": 929,
        "wrong": 609
    },

    "DeepID": {
        "correct": 925,
        "wrong": 613
    },

    "Dlib": {
        "correct": 922,
        "wrong": 616
    },

    "ArcFace": {
        "correct": 924,
        "wrong": 614
    }
}

labels = list(dict.keys())

tot_dict = {}
for model in labels:
    tot_dict[model] = dict[model]["correct"] + dict[model]["wrong"]

sample1 = [
    round(dict[model]["correct"] / tot_dict[model] * 100, 2) for model in labels
]
sample2 = [
    round(dict[model]["wrong"] / tot_dict[model] * 100, 2) for model in labels
]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
ax2 = ax.twinx()
rects1 = ax.bar(
    x - width/2, sample1, width, label = 'Correct recognition', color = "g"
)
rects2 = ax2.bar(
    x + width/2, sample2, width, label = 'Wrong recognition', color = "r"
)

ax.set_ylabel("Correct recognition (percentages, higher is better)")
ax2.set_ylabel("Wrong recognition (percentages, lower is better)")
ax.set_xlabel("Deep Learning models")
ax.set_title("Correct & wrong recognition percentages per model")
ax.set_xticks(x, labels)

ax.bar_label(rects1, padding=3)
ax2.bar_label(rects2, padding=3)

# to make y axes reach 100
ax.plot(0, 100)
ax2.plot(0, 100)

fig.tight_layout()

fig.legend(bbox_to_anchor=(0.87, 0.9))

plt.show()

time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')
plt.savefig("../plots/emotion/" + time_stamp + ".png", dpi = 300)