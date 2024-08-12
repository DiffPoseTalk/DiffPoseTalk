#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Fetch FLAME data
echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ./models/data

echo -e "\nDownloading FLAME..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './models/data/FLAME2020.zip'  --no-check-certificate --continue
# wget --no-check-certificate --http-user=$username --http-password=$password 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './models/data/FLAME2020.zip' 
# Check if download was successful before unzipping
if [ -f './models/data/FLAME2020.zip' ]; then
    unzip ./models/data/FLAME2020.zip -d ./models/data/FLAME2020
else
    echo "FLAME2020 download failed."
fi

# Download FLAME_masks.zip
echo -e "\nDownloading FLAME_masks..."
wget --no-check-certificate --http-user=$username --http-password=$password 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O './models/data/FLAME_masks.zip'
# Check if download was successful before unzipping
if [ -f './models/data/FLAME_masks.zip' ]; then
    unzip ./models/data/FLAME_masks.zip -d ./models/data/FLAME_masks
    mv ./models/data/FLAME_masks/FLAME_masks.pkl ./models/data
else
    echo "FLAME_masks download failed."
fi

# Download mediapipe_landmark_embedding.zip
echo -e "\nDownloading mediapipe_landmark_embedding..."
wget 'https://github.com/yfeng95/DECA/raw/master/data/landmark_embedding.npy' -O './models/data/landmark_embedding.npy'
echo -e "\nAll downloads and extractions completed."

