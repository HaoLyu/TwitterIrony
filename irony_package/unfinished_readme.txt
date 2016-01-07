Files in this directory:

tweet_features.py is the main file, which could generate all features of an author of the ironic tweet and the features of one tweet.

tweet_stream.py is used to crawl tweets with #irony and store them into local MongoDB 'IronyHQ'

LDA_topic.py could generate top 100 topic from all tweets in the MongoDB. This file could get the author’s historical proportions of theses topics. 

Tweet_Transfer_BOW.py is used to process tweets and get Bag of Words and tokenizer counts matrix. It also gets the unigram word list and bigram word list.

test_sarcasm_stream.py crawls all the tweets indexed in the test_sarcasm.tv and stores them into MongoDB

AllTweets.py collects all the tweets’ context and authors in the MongoDB

intensifiers.txt contains 51 intensifier words in wikipedia(https://en.wikipedia.org/wiki/Intensifier)

test_sarcasm.tsv contains the test corpus given by Riloff 2013, which consists of 2278 tweets, out of which 506 are annotated as sarcastic.(http://www.cs.utah.edu/~riloff/publications.html#socialmediapubs) 

StanfordNLP directory contains the StanfordNLP Core Sentiment Analysis
There are detailed comments in each file


