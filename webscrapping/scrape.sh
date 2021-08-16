#!/bin/sh
while read p; do
    echo "$p"
    ~/.local/share/virtualenvs/venv/bin/python dataretrieval/bing_image_downloader.py -s "$p" -o "Dataset/$p" --limit 1000000000 
    ~/.local/share/virtualenvs/venv/bin/python dataretrieval/google_image_scraper.py -s "$p" -o "Dataset/$p" 
    ~/.local/share/virtualenvs/venv/bin/python dataretrieval/duckduckgo.py -s "$p" -o "Dataset/$p" 
done <$1
