# Collect all profiles of users. Get the bag of words (unigrams) of each profile.
# update each doucment in MongoDB with field "profile_unigrams_names" and "profile_unigrams"
from AllTweets import collect_profiles
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import Tweet_Transfer_BOW as BOW

# Connect to MongoDB
client = MongoClient('127.0.0.1', 27017)
db = client['IronyHQ']
dbtweets = db.tweets

all_unigrams = BOW.Get_unigrams(collect_profiles())
vect1 = CountVectorizer(vocabulary=all_unigrams)

for i in range(dbtweets.find({'profile':{"$exists": True}}).count()):
	tweet_id = dbtweets.find({'profile':{"$exists": True}})[i]['tweet_id']
	profile = dbtweets.find({'profile':{"$exists": True}})[i]['profile']
	profile_unigrams = vect1.transform(profile).toarray()
	print profile_unigrams
	print profile_unigrams[0]
	print profile_unigrams.shape
	print profile_unigrams[0].shape
	break
	for uu in xrange(profile_unigrams[0].shape[0]):
		if profile_unigrams[0][uu]>1:
			profile_unigrams[0][uu] = 1
	S_uni = sparse.csr_matrix(profile_unigrams)
	serialized_uni = pickle.dumps(S_uni, protocol=0)
	result = dbtweets.update_one({"tweet_id": tweet_id},
		{
		    "$set": {
                "profile_unigrams": serialized_uni
        	}
		}
	)
	print 'No %d'%i


