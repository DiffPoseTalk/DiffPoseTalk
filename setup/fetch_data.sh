#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Fetch FLAME data
echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ./data

echo -e "\nDownloading FLAME..."
wget --no-check-certificate --http-user=$username --http-password=$password 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './data/FLAME2020.zip' 
# Check if download was successful before unzipping
if [ -f './data/FLAME2020.zip' ]; then
    unzip ./data/FLAME2020.zip -d ./data/FLAME2020
else
    echo "FLAME2020 download failed."
fi

# Download FLAME_masks.zip
echo -e "\nDownloading FLAME_masks..."
wget --no-check-certificate --http-user=$username --http-password=$password 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O './data/FLAME_masks.zip'
# Check if download was successful before unzipping
if [ -f './data/FLAME_masks.zip' ]; then
    unzip ./data/FLAME_masks.zip -d ./data/FLAME_masks
    mv ./data/FLAME_masks/FLAME_masks.pkl ./data
else
    echo "FLAME_masks download failed."
fi

# Download mediapipe_landmark_embedding.zip
echo -e "\nDownloading mediapipe_landmark_embedding..."
wget --no-check-certificate --http-user=$username --http-password=$password 'https://files.is.tue.mpg.de/tbolkart/FLAME/mediapipe_landmark_embedding.zip' -O './data/landmark_embedding.zip'
# Check if download was successful before unzipping
if [ -f './data/landmark_embedding.zip' ]; then
    unzip ./data/landmark_embedding.zip -d ./data/landmark_embedding
    mv ./data/landmark_embedding/mediapipe_landmark_embedding.npz ./data
else
    echo "mediapipe_landmark_embedding download failed."
fi

echo -e "\nAll downloads and extractions completed."

