#!/bin/bash
FILES="/home/ubuntu/Github/Rofunc/doc/img/task_gif/*.gif"
for filename in $FILES; do
    echo "Processing $filename file..."
    ffmpeg -i $filename -vf crop=640*ih/480:ih:iw/2-320*ih/480:0 -r 10 -b:v 5000k ../task_gif2/$(basename -- "$filename")
done
