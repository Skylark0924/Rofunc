#!/bin/bash
FILES="/home/ubuntu/Github/Rofunc/doc/img/task_gif3/*.gif"
for filename in $FILES; do
    echo "Processing $filename file..."
    ffmpeg -hide_banner -i $filename -i /home/ubuntu/Github/Rofunc/doc/img/logo/logo2_nb.png -filter_complex "[1:v] scale=-1:ih/6 [logo];[0:v][logo]overlay=x=W-w-10:y=H-h-10" /home/ubuntu/Github/Rofunc/doc/img/task_gif/$(basename -- "$filename") -y
done
