// Get the word sentiment score by SentiStrength
// Each word has two value: negative score and positive score
// Negative score: -1 (not negative) to -5 (extremely negative)
// Positive score: 1 (not positive) to 5 (extremely positive)
// I add them to get a whole sentiment score, which is from -4(extremely negative) to 4(extremely positive), 0 means neutral.
import uk.ac.wlv.sentistrength.*;

public class WordSentiStrength {

	public static void main(String[] args) {
		// Change the Exampleword to see its sentiment score
		String outcome = WordSentiStrength.getSentiStrength("Exampleword");
		System.out.println("Sentiment score is: "+outcome);

	}
	
	public static String getSentiStrength(String word) {
		// This function need the sentidata and I define its path
		String Datapath = "/Users/haolyu/IronyHQpub/myfile/stanfordNLP/lib/SentStrength_Data/";
		
		// One initialization and repeated classifications
		SentiStrength sentiStrength = new SentiStrength(); 
		
		// Create an array of command line parameters to send 
		String ssthInitialisation[] = {"sentidata", Datapath, "explain"};
		sentiStrength.initialise(ssthInitialisation); //Initialize
		
		//can now calculate sentiment scores quickly without having to initialize again

		String[] scorelist = (sentiStrength.computeSentimentScores(word)).split(" ");
		int positive_score = Integer.parseInt(scorelist[0]);
		int negative_score = Integer.parseInt(scorelist[1]);
		String score = Integer.toString(positive_score + negative_score);
		return score; 
		}
}

