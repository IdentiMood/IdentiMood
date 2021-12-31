from deepface import DeepFace

INPUT_FILE_PATH = "file_list_VGG-Face2.txt"

with open(INPUT_FILE_PATH) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

f = open('file_list_VGG-Face2_accepted_by_deepface.txt', 'a') 

for (line, index) in zip(lines, range(0, len(lines))):
    print(index, "/", len(lines))
    
    try:
        result = DeepFace.verify(
            img1_path = line,
            img2_path = line,
        )
        print(line)
        f.write(line + "\n")
    except ValueError as e:
        # print(e)
        continue

