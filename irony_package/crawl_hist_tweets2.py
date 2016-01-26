# process
# requests model
import requests
import sys
import operator
from bs4 import BeautifulSoup
from pymongo import MongoClient
import datetime
from multiprocessing import Process, Queue
from multiprocessing import Pool

# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

count = Queue()
for i in range(dbtweets.find({'hist_list' : {'$exists': False}}).count()):
	count.put(i)

queue = Queue()
total_number = dbtweets.find({'hist_list' : {'$exists': False}}).count()

for i in range(dbtweets.find({'hist_list' : {'$exists': False}}).count()):
	try:
		one_doc = [dbtweets.find({'hist_list' : {'$exists': False}})[i]['tweet_id'], 
					dbtweets.find({'hist_list' : {'$exists': False}})[i]['author_full_name']
					]
		#print "this is No.%d" %i + " Doc: ", one_doc
		queue.put_nowait(one_doc)
		break
	except Exception:
		continue



url = "https://twitter.com/login"
payload = { 'session[username_or_email]': 'irony_research', 
			'session[password]': 'research_irony'
			}

# two more accounts

#payload1 = { 'session[username_or_email]': 'irony_research1', 
#			'session[password]': 'research1_irony'
#			}

#payload2 = { 'session[username_or_email]': 'irony_research2', 
#			'session[password]': 'research2_irony'
#			}

r = requests.post(url, data=payload)

proxies = {
	"http": "http://96.5.28.23:8008",
	"https": "https://45.32.43.100:3128",
	"https": "https://137.135.166.225:8120",
	"http": "http://134.249.168.16:80", 
	"http": "http://108.165.33.7:3128"
	
}


headers = [{
			'Host': 'twitter.com',
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:43.0) Gecko/20100101 Firefox/43.0',
			'Accept': 'application/json, text/javascript, */*; q=0.01',
			'Accept-Language': 'en-US,en;q=0.5',
			'Accept-Encoding': 'gzip, deflate',
			'X-Requested-With': 'XMLHttpRequest',
			"Cookie": "guest_id=v1%3A142982010899019442; _ga=GA1.2.513563793.1452280674; dnt=1; remember_checked_on=1; webn=4729798346; _twitter_sess=BAh7CiIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCMYhRDlSAToMY3NyZl9p%250AZCIlNjAwZTgyMDc3MThmNmZhODg1OTA5Zjg5ZGZhMDYyMWM6B2lkIiU0YjY0%250AN2RlMDFlYWM2ZGU4ZjgwY2VjMjY0ZWU5OWZiOToJdXNlcmwrCMr%252B6hkBAA%253D%253D--2df9ca19a7a75ca949a2f50b20fc000b21c61cef; lang=en; kdt=CraZVP9NO2TVAdEcyUYkqUPTKgcLfQ03IilYsiVG; ua=\'f5,m2,m5,msw\'; _gat=1; twid=\'u=4729798346\';auth_token=545FAB363CF7AAA2F49492111432B01C5613DE48; _gat=1",
			'Connection': 'keep-alive'},
			{
			'Host': 'twitter.com',
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:43.0) Gecko/20100101 Firefox/43.0',
			'Accept': 'application/json, text/javascript, */*; q=0.01',
			'Accept-Language': 'en-US,en;q=0.5',
			'Accept-Encoding': 'gzip, deflate',
			'X-Requested-With': 'XMLHttpRequest',
			"Cookie": "guest_id=v1%3A142982010899019442; _ga=GA1.2.513563793.1452280674; dnt=1; remember_checked_on=1; webn=4729798346; _twitter_sess=BAh7CiIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCMYhRDlSAToMY3NyZl9p%250AZCIlNjAwZTgyMDc3MThmNmZhODg1OTA5Zjg5ZGZhMDYyMWM6B2lkIiU0YjY0%250AN2RlMDFlYWM2ZGU4ZjgwY2VjMjY0ZWU5OWZiOToJdXNlcmwrCMr%252B6hkBAA%253D%253D--2df9ca19a7a75ca949a2f50b20fc000b21c61cef; lang=en; kdt=CraZVP9NO2TVAdEcyUYkqUPTKgcLfQ03IilYsiVG; ua=\'f5,m2,m5,msw\'; _gat=1; twid=\'u=4809014914\';auth_token=6A2CD516E37E2D66CC8B6B79D5A35307E5916FA2; _gat=1",
			'Connection': 'keep-alive'},
			{
			'Host': 'twitter.com',
			'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:43.0) Gecko/20100101 Firefox/43.0',
			'Accept': 'application/json, text/javascript, */*; q=0.01',
			'Accept-Language': 'en-US,en;q=0.5',
			'Accept-Encoding': 'gzip, deflate',
			'X-Requested-With': 'XMLHttpRequest',
			"Cookie": "guest_id=v1%3A142982010899019442; _ga=GA1.2.513563793.1452280674; dnt=1; remember_checked_on=1; webn=4729798346; _twitter_sess=BAh7CiIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCMYhRDlSAToMY3NyZl9p%250AZCIlNjAwZTgyMDc3MThmNmZhODg1OTA5Zjg5ZGZhMDYyMWM6B2lkIiU0YjY0%250AN2RlMDFlYWM2ZGU4ZjgwY2VjMjY0ZWU5OWZiOToJdXNlcmwrCMr%252B6hkBAA%253D%253D--2df9ca19a7a75ca949a2f50b20fc000b21c61cef; lang=en; kdt=CraZVP9NO2TVAdEcyUYkqUPTKgcLfQ03IilYsiVG; ua=\'f5,m2,m5,msw\'; _gat=1; twid=\'u=4808934921\';auth_token=0DA3357AC77D39E2BC73FD0F481E61750A6DED36; _gat=1",
			'Connection': 'keep-alive'}]
# set start time
start_time = datetime.datetime.now()

 
test_count = 0

def foo(key):
	print '***************'
	global test_count
	#print 'test_count %d' % test_count

	client_no = test_count%3
	tweet_id = key[0]
	user_screen_name = key[1]
	hist_list = []
	lc = 0
	#print "user is", user_screen_name, 'has so many following', f_count
	
	try:
		get_url = "https://twitter.com/" + user_screen_name 
		p = requests.get(get_url, headers=headers[client_no], proxies=proxies)
			
		soup = BeautifulSoup(p.content,  "lxml")

		jstt = soup.find("div", {"id": "timeline"}).find("div", {"class": "stream-container  "})
		#print jstt['data-min-position']

		start_position = str(jstt['data-min-position'])
		print start_position
		#print "No %d user %s is under inputing %s :" %(test_count+1, user_screen_name, str(f_count))

		while(start_position != None):
			one_url = 'https://twitter.com/i/profiles/show/' + user_screen_name + '/timeline?include_available_features=1&include_entities=1&last_note_ts=123&max_position=' + start_position + '&reset_error_state=false'
			params = {
					'include_available_features': '1',
					'include_entities': '1',
					'max_position': start_position, 
					'reset_error_state': 'false',
					'last_note_ts': 123
					}

			response = requests.get(one_url, params=params, headers=headers[client_no], proxies=proxies)

			fixtures = (response.json())['inner']
			start_position = fixtures['min_position']
			latent_count = fixtures['new_latent_count']
			lc += int(latent_count)
			print 'lc is:', lc
			print start_position
			soup2 = BeautifulSoup(fixtures['items_html'], "lxml")
			jstt2 = soup2.find_all("div", {"lang": "en"})
			for one_fol in jstt2:
				hist_list.append(one_fol.text)

			if len(hist_list)>999:
				break
				
		test_count += 1
		print "No %d user %s inputed:" %(count.get_nowait(), user_screen_name)
				
		result = dbtweets.update_one({"tweet_id": tweet_id},
					{
					    "$set": {
			                "following_list": f_list
			        	}
					}
				)
	
	
		
	except KeyError:
		return
	
	return



def many_foos():
	try:
		while(not(queue.empty())):
			foo(queue.get_nowait())
	except Exception, e:
		print "exception %s", e
		break

process_num = 4

p = Pool(process_num)
for i in range(process_num):
	p.apply_async(many_foos, args=())
p.close()
p.join()

end_time = datetime.datetime.now()
duration = end_time - start_time

print " total time is ", duration
