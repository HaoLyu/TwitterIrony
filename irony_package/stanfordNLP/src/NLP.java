// Get the whole sentence sentiment by Stanford NLP Sentiment Analysis
// We also get the proportion of positive and negative nodes in the sentence 
// This class onyly get the sentiment analysis of one sentence by default
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;

public class NLP {
    static StanfordCoreNLP pipeline;
    
    public static void init() {
    	Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
        pipeline = new StanfordCoreNLP(props);
    }

    public static ArrayList<String> findSentiment(String tweet) {
    	ArrayList<String> SentiReturn = new ArrayList<String>();
    	String[] SentiClass ={"very negative", "negative", "neutral", "positive", "very positive"};

    	//Sentiment is an integer, ranging from 0 to 4. 
    	//0 is very negative, 1 negative, 2 neutral, 3 positive and 4 very positive.
    	int sentiment = 2;
    	
        if (tweet != null && tweet.length() > 0) {
            Annotation annotation = pipeline.process(tweet);
            
            List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
            if (sentences != null && sentences.size() > 0) {
   
	            ArrayCoreMap sentence = (ArrayCoreMap) sentences.get(0);            	
	            Tree tree = sentence.get(SentimentAnnotatedTree.class);
	            
	            Iterator<Tree> it = tree.iterator();
		    	double nodescount = 0.0;
		    	double positivenode = 0.0;
		    	double negativenode = 0.0;
	    	    
		    	// Loop each node in the tree
		    	while(it.hasNext()){
		    		nodescount++;
		    		Tree t = it.next();
		    		if(RNNCoreAnnotations.getPredictedClass(t)>2){
		    			positivenode++;
		    		} else if(RNNCoreAnnotations.getPredictedClass(t)<2){
		    			negativenode++;
		    		}
		    	}
		    	
		    	Double posinode = Math.round((positivenode/nodescount)*100)/100.0;
		    	Double neganode = Math.round((negativenode/nodescount)*100)/100.0;
		    	sentiment = RNNCoreAnnotations.getPredictedClass(tree);
	            
	            SentiReturn.add(SentiClass[sentiment]);
	            SentiReturn.add(Double.toString(posinode));
	            SentiReturn.add(Double.toString(neganode));
	        }
        }
        return SentiReturn;
    }
}
