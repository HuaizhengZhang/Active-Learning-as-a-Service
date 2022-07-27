#!/usr/bin/env bash

mkdir -p data
cd data
input="../test_images.txt"
while IFS= read -r line; do
  class_name=$(echo "$line" | cut -d'/' -f 9)
  img_name=$(echo "$line" | cut -d'/' -f 10)
  wget "$line" -O "${class_name}_${img_name}"
done <"$input"
