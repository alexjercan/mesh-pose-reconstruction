#!/bin/bash
# blenderRenderer 2020
# Download command: bash data/scripts/get_bdataset.sh
# Train command: python train.py
# Default dataset location is next to /mesh-pose-reconstruction:
#   /parent_folder
#     /bdataset
#     /mesh-pose-reconstruction

start=$(date +%s)
mkdir -p ../tmp
cd ../tmp/

# Download/unzip images and labels
d='..' # unzip directory
filename="bdataset.zip"
fileid="1Xbyltr4ZhsLx1i8yKuOwWxak0heZkiew"
echo 'Downloading' $url$f '...' 
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename} && unzip -q ${filename} -d $d && rm ${filename} & # download, unzip, remove in background
wait # finish background tasks

end=$(date +%s)
runtime=$((end - start))
echo "Completed in" $runtime "seconds"