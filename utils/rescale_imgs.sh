find $1 -iname "*.jpg" | xargs -L1 -I{} convert -verbose -resize $2% "{}" "{}"