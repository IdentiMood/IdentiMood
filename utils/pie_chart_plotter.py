import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np

dict = {
    "DeepFace CNN for Emotion Verification": {
        "correct": 926,
        "wrong": 612
    }
}

labels = '% of correct\nemotion verifications\n(bigger is better)', \
    '% of wrong\nemotion verifications\n(smaller is better)'
sizes = [926, 612]

fig1, ax1 = plt.subplots()
ax1.pie(
    sizes, labels=labels, autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 16, "fontweight": "bold"},
    colors = ["g", "r"]
)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.suptitle(
    "Correct & wrong emotion recognition", fontsize = 16, fontweight = "bold"
)

plt.show()

time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')
plt.savefig("../plots/emotion/" + time_stamp + ".png", dpi = 300)