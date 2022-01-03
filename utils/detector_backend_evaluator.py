import deepface
from deepface.commons import functions

from deepface import DeepFace
from datetime import datetime
import time
import json

detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

errors = dict()

models = {}
# models['age'] = DeepFace.build_model('Age')
# models['gender'] = DeepFace.build_model('Gender')
models['emotion'] = DeepFace.build_model('Emotion')
# models['race'] = DeepFace.build_model('race')

for detector in detector_backends:
    errors[detector] = 0

with open("file_list_COMPLETE_short.txt") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

tot_lines = len(lines)

for (line, index) in zip(lines, range(0, tot_lines)):
    print("loading image", index + 1, " of ", tot_lines, "...")
    img = functions.load_image(line)
    print("loading image", index + 1, " of ", tot_lines, "OK")

    for detector in detector_backends:
        print("  ", detector)
        
        try:
            functions.detect_face(img = img, detector_backend = detector)
            # DeepFace.analyze(
            #     line, actions = ['emotion'], models = models, 
            #     enforce_detection = True, detector_backend = detector, 
            #     prog_bar = True
            # )
            # DeepFace.verify(
            #     img1_path=line, 
            #     img2_path=line,
            #     detector_backend=detector
            # )
        except ValueError as e:
            print("     ", "detection error!")
            errors[detector] += 1

file_name = datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H-%M-%S')

with open("../logs/detector/" + file_name + ".json", "w") as output_log:
    output_log.write(json.dumps(errors, indent = 4))
    output_log.close()

