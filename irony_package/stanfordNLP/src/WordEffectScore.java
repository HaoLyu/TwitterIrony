// Get the effect score of the given word by retriving the Warriner's corpus.
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class WordEffectScore {
	// Create new HashMap.
	static HashMap<String, String> hash = new HashMap<>();
	
	public static void main(String[] args) throws FileNotFoundException{
		// We can replace Exampleword to any word and see its effect score
		WordEffectScore.geteffectscore("Exampleword");
	}
    
	public static String geteffectscore(String word) {
		// Read the corpus from path csvFile
		String csvFile = "/Users/haolyu/IronyHQpub/myfile/stanfordNLP/lib/Warriner_word_effect/Ratings.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		
		// Use Hashmap to map the corpus
		try {

			br = new BufferedReader(new FileReader(csvFile));
			
			while ((line = br.readLine()) != null) {
				// use comma as separator
				String[] linesplit = line.split(cvsSplitBy);
				String key = linesplit[0].replace("\"", "");
				String value = linesplit[1].replace("\"", "");

				hash.put(key, value);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
  
     
		// Look up the effect score value of this word.
		// This score is from 0 to 9. 0 is the least pleasant, 9 is most pleasant.
		String wordscore = hash.get(word);
		
		if(wordscore!=null){
			//System.out.println("word score is:" + wordscore);
			return wordscore;
		} else {
			//System.out.println("This word is not in the corpus, so we give 5.0");
			return "5.0";
		}
	}
}
