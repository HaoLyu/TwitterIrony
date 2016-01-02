// This is the main class
// It could connect MongoDB and read each tweet
// Then it will update each document(tweet file) with new features
// Added features: predict, positivenode, negativenode, sentiment_distance, max_word_senti, 
// min_word_senti, effect_distance, max_word_effect, min_word_effect

import java.util.ArrayList;
import java.util.Collections;

import org.bson.Document;

import com.mongodb.Block;
import com.mongodb.client.FindIterable;
import com.mongodb.MongoClient;
import com.mongodb.client.MongoDatabase;
public class SentiMain {

	public static void main(String[] args) {
		// Connect to local MongoDB "IronyHQ"
		MongoClient mongoClient = new MongoClient();
		MongoDatabase db = mongoClient.getDatabase("IronyHQ");
		// Iterate all files in collection "test"
		FindIterable<Document> iterable = db.getCollection("test").find();		
		
		iterable.forEach(new Block<Document>() {
		    @Override	
		    public void apply(final Document document) {
				// Get the tweet and all words in it
			    String tweet = (String) document.get("text");
			    String[] words = tweet.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");
			     
			    ArrayList<Double> wordsentiment= new ArrayList<Double>();
			    ArrayList<Double> wordeffect = new ArrayList<Double>();
			    
			    // Get the whole sentiment of the tweet from NLP.class
			    NLP.init();
			    String tweetsentiment = NLP.findSentiment(tweet).get(0);
			    String posinode = NLP.findSentiment(tweet).get(1);
			    String neganode = NLP.findSentiment(tweet).get(2);
			
				for(String word:words){
					// Get the sentiment of each word
					Double SentiStrengthScore = Double.parseDouble(WordSentiStrength.getSentiStrength(word));
					wordsentiment.add(SentiStrengthScore);
					
					// Get the effect score of each word
					Double effectScore = Double.parseDouble(WordEffectScore.geteffectscore(word));
					wordeffect.add(effectScore);
				}
		 
				Double max_word_senti = Collections.max(wordsentiment);
				Double min_word_senti = Collections.min(wordsentiment);
				Double sentiment_distance = max_word_senti- min_word_senti;
		 
				Double max_word_effect = Collections.max(wordeffect);
				Double min_word_effect = Collections.min(wordeffect);
				Double effect_distance = Math.round((max_word_effect - min_word_effect)*100)/100.0;
		    	
				// Update this tweet
		        db.getCollection("test").updateOne(new Document("author", document.get("author")),
		                new Document("$set", new Document("predict", tweetsentiment)
		                					.append("positivenode", posinode)
		                					.append("negativenode", neganode)
		                					.append("sentiment_distance", sentiment_distance)
		                					.append("max_word_senti", max_word_senti)
		                					.append("min_word_senti", min_word_senti)
		                					.append("effect_distance", effect_distance)
		                					.append("max_word_effect", max_word_effect)
		                					.append("min_word_effect", min_word_effect)));
		    }
		});
	}
}