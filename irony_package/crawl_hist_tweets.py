# run this after running author_basic.py
from pymongo import MongoClient
import sys
import json
import argparse
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# select different pagedown based on the count of tweets
def select_pagedown(n):
	number = int(n)
	if number < 1:
		return 0
	elif number <= 500:
		return (number/3)
	else:
		return 200

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets


for i in range(dbtweets.find().count()):
	hist_tweets_list = []
	tweets_count = dbtweets.find()[i]['tweets_count']
	sid = dbtweets.find()[i]['tweet_id']
	test_author_name = dbtweets.find()[i]['author_full_name']

	browser = webdriver.Firefox()
	browser.get("https://twitter.com/%s" % (test_author_name))
	time.sleep(1)

	elem = browser.find_element_by_tag_name("body")
	# no_of_pagedowns should be decided by the status count of the author.
	# 5:20, 10:40, 25: 80, 50:140, 75:200
	no_of_pagedowns = select_pagedown(tweets_count)

	while no_of_pagedowns:
		elem.send_keys(Keys.PAGE_DOWN)
		time.sleep(0.2)
		no_of_pagedowns-=1

	post_elems = browser.find_elements_by_tag_name('li')

	for post in post_elems:
		data_id = post.get_attribute("data-item-id")
		try:
			if ((data_id  != None) & (len(str(data_id))==18)):
				hist_tweets_list.append(post.get_attribute("data-item-id")) 
			else:
				continue		
		except:
			continue

	result = dbtweets.update_one({"tweet_id": sid},
				{
				    "$set": {
		                "hist_tweets": hist_tweets_list
		        	}
				}
			)

	print "No. %d author %s has historical tweets: %d" %(i+1, test_author_name,len(hist_tweets_list))
	browser.close()
#########
# crawl all content in a page

"""
post_elems2 = browser.find_elements_by_tag_name('p')
for post in post_elems2:
	print post.text
print 'text count is:', len(post_elems2)
"""
