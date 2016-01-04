import tweepy
import argparse
import sys
import json
import time
import csv	

with open("Tweet_Data/bamman.csv","r") as f:
	row_count = sum(1 for row in f)
print row_count

csvfile = open('Tweet_Data/bamman.csv', 'r')
jsonfile = open('Tweet_Data/bamman.json', 'w')

fieldnames = ("tweet_id", "sarcasm_score", "author_full_name", "author_id", "tweet_text")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')