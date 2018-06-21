#!/bin/bash
mogrify -format png *.JPG
for filename in *.png; do
  ../extract_features/extract_features.ln -haraff -i $filename -sift
  ../extract_features/extract_features.ln -hesaff -i $filename -sift
done
