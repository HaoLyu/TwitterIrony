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

# time convert 'Joined October 2009' -> datetime.datetime(2009, 10, 1, 0, 0)
def date_convert(s):
	date_string = s.split('-')[1].strip()
	date = datetime.datetime.strptime(date_string, '%d %b %Y')

	return date

def time_zone_check(s):
	try:
		g = geocoders.GoogleV3()
		place, (lat, lng) = g.geocode(s)
		return place.split(',')[-1].encode('utf-8')
	except:
		return ''
# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

cache = {}

for i in range(dbtweets.find().count()):
	sid = dbtweets.find()[i]['tweet_id']
	test_author_name = dbtweets.find()[i]['author_full_name']

	if cache.has_key(sid):
		continue
	else:
		try:
			f = urllib.urlopen("http://twitter.com/%s" % (test_author_name))
			html = f.read().replace("</html>", "") + "</html>"
			f.close()
			soup = BeautifulSoup(html,'lxml')
			jstt = soup.find("li", {"class": "ProfileNav-item ProfileNav-item--tweets is-active"}).find("span", {"class": "ProfileNav-value"})
			tweets_count = str(jstt.get_text())
			tweets_count = num_convert(tweets_count)
			#print 'test_author_name : %s'%(test_author_name)
			#print 'tweets count : %s'%(tweets_count)

			jstt = soup.find("li", {"class": "ProfileNav-item ProfileNav-item--following"}).find("span", {"class": "ProfileNav-value"})
			following_count = str(jstt.get_text())
			following_count = num_convert(following_count)
			#print 'following count : %s'%(following_count)

			jstt = soup.find("li", {"class": "ProfileNav-item ProfileNav-item--followers"}).find("span", {"class": "ProfileNav-value"})
			followers_count = str(jstt.get_text())
			followers_count = num_convert(followers_count)
			#print 'followers count : %s'%(followers_count)

			jstt = soup.find("p", {"class": "ProfileHeaderCard-bio u-dir"})
			#print jstt.get_text()
			profile = jstt.get_text().encode('ascii','ignore')
			#print 'profile : %s'%(profile)

			jstt = soup.find("div", {"class": "ProfileHeaderCard-joinDate"}).find("span", {"class": "ProfileHeaderCard-joinDateText js-tooltip u-dir"})["title"]
			join_time = str(jstt)
			#print "join time",join_time
			join_time_convert = date_convert(join_time)
			#print "join time",join_time_convert
			current_time = datetime.datetime.now()
			duration = (current_time - join_time_convert).days
			#print 'duration: %s'%(duration)
			
			jstt = soup.find("h1", {"class": "ProfileHeaderCard-name"}).find("span", {"class": "ProfileHeaderCard-badges ProfileHeaderCard-badges--1"})
			verified = str(jstt)
			if verified is not None:
				verified = 'yes'
			else:
				verified = 'no'
			#print 'verified: %s'%verified

			avg = tweets_count/float(duration)
			# print "avg tweet number is: %f"%avg

			jstt = soup.find("div", {"class": "ProfileHeaderCard-location"}).find("span", {"class": "ProfileHeaderCard-locationText u-dir"})
			time_zone = time_zone_check(str(jstt.text).strip()).strip()
			#print time_zone

			result = dbtweets.update_one({"tweet_id": sid},
				{
				    "$set": {
		                "tweets_count": tweets_count,
		                "following_count": following_count,
		                "followers_count": followers_count,
		                "profile": profile,
		                "duration": duration,
		                "verified": verified,
		                "avg_tweet": avg,
		                "time_zone": time_zone
		        	}
				}
			)

  			cache[sid] = tweets_count
  			print "No. %d tweet, " %(i+1)
		except Exception:
			continue

