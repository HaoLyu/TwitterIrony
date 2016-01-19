# thread
# requests model
import requests
import sys
import operator
from bs4 import BeautifulSoup
from pymongo import MongoClient
import datetime
import threading
# Load the data into mongodb
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

queue = {}

#for i in range(dbtweets.find().count()):
for i in range(dbtweets.find({'following_list' : {'$exists': False}}).count()):
#	try:
#		queue[dbtweets.find()[i]['tweet_id']] = [dbtweets.find()[i]['following_count'], dbtweets.find()[i]['author_full_name'], []]

	try:
		queue[dbtweets.find({'following_list' : {'$exists': False}})[i]['tweet_id']] = [dbtweets.find({'following_list' : {'$exists': False}})[i]['following_count'], 
																						dbtweets.find({'following_list' : {'$exists': False}})[i]['author_full_name'],
																						[]]
						
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
	"http": "http://107.167.24.130:80",
	"http": "http://104.155.30.220:3128",
	"http": "http://104.155.218.45:80", 
	"http": "http://104.155.12.205:80",
	"http": "http://104.155.209.138:80"
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
	global queue
	global test_count
	#print 'test_count %d' % test_count

	client_no = test_count%3
	
	user_screen_name = queue[key][1]
	f_count = queue[key][0]
	#print "user is", user_screen_name, 'has so many following', f_count
	
	if f_count > 0:
		try:
			get_url = "https://twitter.com/" + user_screen_name + "/following"
			p = requests.get(get_url, headers=headers[client_no], proxies=proxies)
				
			soup = BeautifulSoup(p.content,  "lxml")

			jstt = soup.find("div", {"class": "GridTimeline"}).find("div", {"class": "GridTimeline-items"})
			#print jstt['data-min-position']

			start_position = str(jstt['data-min-position'])
			try: 
				print "No %d user %s is under inputing %s :" %(test_count+1, user_screen_name, str(f_count))

				while(start_position != '0'):
					one_url = 'https://twitter.com/' + user_screen_name + '/following/users?include_available_features=1&include_entities=1&max_position=' + start_position + '&reset_error_state=false'
					params = {
							'include_available_features': '1',
							'include_entities': '1',
							'max_position': start_position, 
							'reset_error_state': 'false'
							}

					response = requests.get(one_url, params=params, headers=headers[client_no], proxies=proxies)

					fixtures = response.json()
					start_position = fixtures['min_position']
					soup2 = BeautifulSoup(fixtures['items_html'], "lxml")
					jstt2 = soup2.find_all("div", {"class": "ProfileCard  js-actionable-user"})
					for one_fol in jstt2:
						queue[key][2].append(one_fol['data-screen-name'])
				
				result = dbtweets.update_one({"tweet_id": key},
							{
							    "$set": {
					                "following_list": queue[key][2]
					        	}
							}
						)				

		
			except Exception, e:
				print 'error %s' %e
				return
			test_count += 1
			print "No %d user %s inputed %s :" %(test_count, user_screen_name, str(f_count))
		except AttributeError:
			return
	else: 
		return
	#print queue[key][2]
	pass

def many_foos(threadName, keys):
	try:
		for key in keys:
			foo(key)
		return '%s finished' % threadName
	except Exception, e:
		print "exception %s", e
def partition(alist, indices):
	output = []
	prev = 0
	for index in indices:
		output.append(alist[prev:index])
		prev = index
	return output


def ave_dist(sorted_dict):
	return_list = []
	sorted_list = []
	big_number_list = []
	tem_sum = 0
	for i in range(len(sorted_dict)):
		tem_sum += sorted_dict[i][1]
		if sorted_dict[i][1] >= ave_count:
			big_number_list.append(i)
			continue
		elif (tem_sum >= ave_count): 			
			sorted_list.append(i)
			tem_sum = 0
		else:
			continue
	sorted_dict_keys = [y[0] for y in sorted_dict]
	return_list = return_list + partition([m[0] for m in sorted_dict], sorted_list)

	for j in big_number_list:
		return_list.append([sorted_dict[j][0]])

	return return_list

thread_up = tuple(queue.keys())
thread_up_list = []
thread_num = 10
following_count_sum = sum(float(queue[key][0]) for key in thread_up)
ave_count = following_count_sum/thread_num

thread_up_dic = {}
for key in thread_up:
	thread_up_dic[key] = float(queue[key][0])

sort_thread_up_dic = sorted(thread_up_dic.items(), key=operator.itemgetter(1))
thread_up_list = tuple(ave_dist(sort_thread_up_dic))

threads = []

try:
	for i in range(len(thread_up_list)):
		thread_name = 'thread'+str(i)
		t = threading.Thread(target=many_foos, name=thread_name, args=(thread_name, thread_up_list[i],))
		threads.append(t)
		t.start()
		
except:
	print "Error: unable to start thread"

for t in threads:
	t.join()

end_time = datetime.datetime.now()
duration = end_time - start_time

print " total time is ", duration
