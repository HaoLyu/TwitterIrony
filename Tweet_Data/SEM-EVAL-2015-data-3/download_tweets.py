#!/usr/bin/python

import sys
import urllib
import re
import json
import csv 

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

cache = {}

fout = open("irony-tweets.txt",'w')
writer = csv.writer(fout)

for line in open(sys.argv[1]):
	fields = line.rstrip('\n').split('\t')
	sid = fields[0]
	#uid = fields[1]

	#url = 'http://twitter.com/intent/retweet?tweet_id=%s' % (sid)
	#print url

        tweet = None
	text = "Not Available"
	if cache.has_key(sid):
		text = cache[sid]
	else:
                try:
                        f = urllib.urlopen("http://twitter.com/intent/retweet?tweet_id=%s" % (sid))
                        #Thanks to Arturo
                        #Modified by Aniruddha!
                        html = f.read().replace("</html>", "") + "</html>"
                        soup = BeautifulSoup(html)

			jstt   = soup.find_all("div", "tweet-text")
			tweets = list(set([x.get_text() for x in jstt]))
			#print len(tweets)
			#print tweets
			if(len(tweets)) > 1:
				continue

			text = tweets[0]
			cache[sid] = tweets[0]

                        for j in soup.find_all("input", "json-data", id="init-data"):
                                js = json.loads(j['value'])
                                if(js.has_key("embedData")):
                                        tweet = js["embedData"]["status"]
                                        text  = js["embedData"]["status"]["text"]
                                        cache[sid] = text
                                        break
                except Exception:
                        continue

        if(tweet != None and tweet["id_str"] != sid):
                text = "Not Available"
                cache[sid] = "Not Available"
        text = text.replace('\n', ' ',)
        text = re.sub(r'\s+', ' ', text)
        #print json.dumps(tweet, indent=2)
        cur_tweet = "\t".join(fields + [text]).encode('utf-8')
        print cur_tweet
        writer.writerow([sid, cur_tweet])

fout.close()

