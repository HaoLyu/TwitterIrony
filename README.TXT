Twitter Irony Feature Model

1. you need MongoDB and Firefox Browser installed. In addition, you also need these Python packages installed:pymongo, bs4, selenium, sklearn, tweepy  
2. Create new db 'IronyHQ' in the localhost and create a new collection 'tweets' in it. Then run: ./runme.sh 
Then you have tweets documents in your collection 'tweets'.
Now I've added these fields:
tweet_id, tweet_text, author_full_name, author_id, sarcasm_score  
tweets_count, following_count, followers_count, profile, duration
intensifier, word_unigrams, word_bigrams. 


