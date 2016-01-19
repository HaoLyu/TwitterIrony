# Extract the count of follwing accounts of the author
# Haolyu, University of Texas at Austin
import sys
import urllib
import re
import json
import string
import datetime
import csv 
from pymongo import MongoClient
from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

# remove punctuation in string
regex = re.compile('[%s]' % re.escape(string.punctuation))

# number convert '100,5K'->1005000
def num_convert(val):
	try:
		if (('K' not in val)&('M' not in val)&('B' not in val)):
			return float(regex.sub('', val))
		val_1 = regex.sub('.', val)
		lookup = {'K': 1000, 'M': 1000000, 'B': 1000000000}
		unit = val_1[-1]
		try:
			number = float(val_1[:-1])
		except ValueError:
			print 'number is wrong: ',val
			return -1
		if unit in lookup:
			return lookup[unit] * number
		return float(val_1)
	except ValueError:
		print 'number is wrong: ',val
		return -1

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


for i in range(dbtweets.find().count()):
	sid = dbtweets.find()[i]['tweet_id']
	test_author_name = dbtweets.find()[i]['author_full_name']


	try:
		f = urllib.urlopen("http://twitter.com/%s" % (test_author_name))
		html = f.read().replace("</html>", "") + "</html>"
		f.close()
		soup = BeautifulSoup(html,'lxml')
		
		jstt = soup.find("li", {"class": "ProfileNav-item ProfileNav-item--following"}).find("span", {"class": "ProfileNav-value"})
		following_count = str(jstt.get_text())
		following_count = num_convert(following_count)
		#print 'following count : %s'%(following_count)

		
		result = dbtweets.update_one({"tweet_id": sid},
			{
			    "$set": {
	                "following_count": following_count
	        	}
			}
		)

		print "No. %d tweet, " %(i+1)
		
	except Exception:
		continue

