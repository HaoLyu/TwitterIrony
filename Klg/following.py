# Extract all following names of an author 
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
	if n < 19:
		return 0
	else:
		a = round(n/18.0)
		b = int((a+1)*7)
		return b

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

for i in range(dbtweets.find().count()):
	try:
		following_count = dbtweets.find()[i]['following_count']
		following_list = []
		sid = dbtweets.find()[i]['tweet_id']
		author_name = dbtweets.find()[i]['author_full_name']

		if following_count:	
			browser = webdriver.Firefox()
			browser.get("https://twitter.com/DataScienceCtrl/following")

			time.sleep(1)
			# Login to Twitter
			username = browser.find_element_by_xpath("//div[@class='clearfix field']/input[@class='js-username-field email-input js-initial-focus']")
			password = browser.find_element_by_class_name("js-password-field")
			username.send_keys("irony_research")
			password.send_keys("research_irony")
			login_attempt = browser.find_element_by_xpath("//div[@class='clearfix']/button[@class='submit btn primary-btn']")
			login_attempt.click()
			time.sleep(1)
			# Scroll down in the browser
			elem = browser.find_element_by_tag_name("body")
			no_of_pagedowns = select_pagedown(int(following_count))

			while no_of_pagedowns:
				elem.send_keys(Keys.PAGE_DOWN)
				time.sleep(0.2)
				no_of_pagedowns-=1

			#browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
			post_elems = browser.find_elements_by_xpath("//a[@class='ProfileCard-screennameLink u-linkComplex js-nav']/span[@class='u-linkComplex-target']")

			if len(post_elems) > 0:
				for post in post_elems:
					following_list.append(post.text)
					print type(post.text)
					#following_list.append(post.text.encode('ascii','ignore'))
				print ' total %d' % (len(post_elems))
			else:
				following_list = "Not Available"

			result = dbtweets.update_one({"tweet_id": sid},
					{
					    "$set": {
			                "following_list": following_list
			        	}
					}
				)

			browser.close()

		else:
			result = dbtweets.update_one({"tweet_id": sid},
					{
					    "$set": {
			                "following_list": "Not Available"
			        	}
					}
				)

		print "No. %d following accounts, " %(i+1)

	except Exception:
			continue


