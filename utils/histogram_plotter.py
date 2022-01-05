import matplotlib.pyplot as plt
from datetime import datetime
import time

dict = {
    "DLib": 8917 / 20083 * 100,
    "OpenCV": 8026 / 20083 * 100,
    "SSD": 6566 / 20083 * 100,
    "MTCNN": 5372 / 20083 * 100,
    "RetinaFace": 2629 / 20083 * 100
}

x = list(dict.keys())
y = list(dict.values())

plt.bar(x, height=y)  # density=False would make counts
plt.ylabel('Percentage of undetected faces (lower is better)')
plt.xlabel('Detector backend');
plt.title("Face detection errors for each backend detector")
plt.plot()
# plt.show()
time_stamp = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')
plt.savefig("../plots/detection/" + time_stamp + ".png", dpi = 300)