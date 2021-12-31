from deepface import DeepFace

INPUT_FILE_PATH = "file_list_ExtendedYaleB.txt"

with open(INPUT_FILE_PATH) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

f = open('file_list_ExtendedYaleB_accepted_by_deepface.txt', 'a') 

tot_files = len(lines)

output_str = ""

for (line, index) in zip(lines, range(0, len(lines))):
    print(index, "/", tot_files)
    
    try:
        result = DeepFace.verify(
            img1_path = line,
            img2_path = line,
        )
        print(line)
        output_str = output_str + line + "\n"
        
    except ValueError as e:
        # print(e)
        continue

f.write(output_str)

