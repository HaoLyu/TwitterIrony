import org.bson.Document;

import java.util.ArrayList;
import com.mongodb.BasicDBList;
import com.mongodb.Block;
import com.mongodb.MongoClient;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoDatabase;


public class Author_historical_sentiment {
	public static void main(String[] args) {
		// Connect to local MongoDB "IronyHQ"
		MongoClient mongoClient = new MongoClient();
		MongoDatabase db = mongoClient.getDatabase("IronyHQ");
		// Iterate all files in collection "test"
		FindIterable<Document> iterable = db.getCollection("test").find();
		
		iterable.forEach(new Block<Document>() {
		    @Override
		    public void apply(final Document document) {
		    	BasicDBList historical_tweets = (BasicDBList) document.get("historical_tweets");
		    	ArrayList<String> alltweets = new ArrayList<String>();
		    	// get all the historical tweets by author id and store them in alltweets
		    	for (Object one : historical_tweets) {
		    	    alltweets.add((String) one);
		    	}
		    	int PositiveReturn = 0;
		    	int NegativeReturn = 0;
		    	NLP.init();
		    	
		    	for(int i = 0, n = alltweets.size(); i<n; i++) {
		    		String onetweet = alltweets.get(i);
				    String tweetsentiment = NLP.findSentiment(onetweet).get(0);
				    if (tweetsentiment == "very negative" || tweetsentiment == "negative"){
				    	NegativeReturn += 1;
				    }
				    if (tweetsentiment == "very positive" || tweetsentiment == "positive"){
				    	PositiveReturn += 1;
				    }
		    	}
		    	
		    	float positive_proportion = (float) PositiveReturn/(NegativeReturn + PositiveReturn);
		    	float negative_proportion = (float) NegativeReturn/(NegativeReturn + PositiveReturn);
		    	
		    	db.getCollection("test").updateOne(new Document("author", document.get("author")),
		                new Document("$set", new Document("positive_proportion", positive_proportion)
		                					.append("negative_proportion", negative_proportion)
		                					));
		    }
		});
	}
}
