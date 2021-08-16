#!/bin/sh
while read p; do
    echo "$p"
    python bing_image_downloader.py -s "$p" -o "Dataset/$p" --limit 1000000000 
    python google_image_scraper.py -s "$p" -o "Dataset/$p" 
    python duckduckgo.py -s "$p" -o "Dataset/$p" 
done <$1
