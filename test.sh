#!/bin/bash

# Run Python scripts in sequence
python3 DocClassificationSimCSE_news.py newsgroup20
python3 DocClassificationSimCSE_rt.py rotten_tomatoes
python3 DocClassificationSimCSE_db.py dbpedia
