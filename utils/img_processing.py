from PIL import Image

INPUT_FILE_PATH = "file_list_VGG-Face2.txt"
MIN_RES = 700

with open(INPUT_FILE_PATH) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

valid_imgs_count = 0

f = open('file_list_VGG-Face2_gt_700.txt', 'a') 

for line in lines:
  im = Image.open(line)
  w, h = im.size

  if (w > MIN_RES and h > MIN_RES):
    print(line, "-->", im.size)
    f.write(line + "\n")
    valid_imgs_count = valid_imgs_count + 1

print("valid_imgs_count: ",valid_imgs_count)