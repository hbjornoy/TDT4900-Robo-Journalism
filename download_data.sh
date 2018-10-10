#!/bin/bash

# make directories
mkdir -p data/cnn_clean/cnn_stories_tokenized
mkdir -p data/cnn_clean/dm_stories_tokenized

printf "Downloading tokenized CNN-data and unzip it in data/cnn_clean/cnn_stories_tokenized/\n"
printf "Downloaded from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail\n"
page="$(curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0BzQ6rtO2VN95cmNuc2xwUS1wdEE" > /tmp/intermezzo.html)"
echo "$page"

curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > cnn_stories_tokenized.zip
printf "\nUnzipping..."
unzip -o cnn_stories_tokenized.zip -d data/cnn_clean | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'

printf "\nDownloading tokenized DailyMail-data and unzip it in data/cnn_clean/dm_stories_tokenized/\n"
printf "Downloaded from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail\n"
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=0BzQ6rtO2VN95bndCZDdpdXJDV1U" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > dm_stories_tokenized.zip
printf "\nUnzipping...\n"
unzip -o dm_stories_tokenized.zip -d data/cnn_clean | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'

# remove zip-files
printf "\nRemoving zip-files\n"
rm cnn_stories_tokenized.zip
rm dm_stories_tokenized.zip
