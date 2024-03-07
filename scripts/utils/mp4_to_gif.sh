#!/bin/bash
FILES="/home/ubuntu/Videos/SayFunc/*.mp4"
for filename in $FILES; do
    echo "Processing $filename file..."
    ffmpeg -i $filename -i palette.png -filter_complex "[0:v] fps=30,crop=ih-20:ih-20:iw/2-ih/2+10:10,scale=450:-1 [new];[new][1:v] paletteuse" ${filename%.mp4}.gif
done
