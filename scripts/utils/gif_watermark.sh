#!/bin/bash
FILES="/home/ubuntu/Github/Rofunc/doc/img/task_gif/*.gif"
for filename in $FILES; do
    echo "Processing $filename file..."
    ffmpeg -hide_banner -i $filename -i /home/ubuntu/Github/Rofunc/doc/img/logo/logo2_nb.png -filter_complex "[1:v][0:v]scale2ref=oh*mdar:ih/5[logo-out][video-out];[video-out][logo-out]overlay=W-w-10:H-h-10" -c:a copy /home/ubuntu/Github/Rofunc/doc/img/task_gif/$(basename -- "$filename") -y
done
