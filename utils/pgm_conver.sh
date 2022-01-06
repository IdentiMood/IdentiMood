files=$(find . -type f -iname "*.pgm")

for file in $files
do
  echo $file
  convert $file $file.png
done

