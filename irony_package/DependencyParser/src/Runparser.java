import java.util.ArrayList;
import java.util.HashMap;


public class Runparser {
	public static void main(String[] args) {
		HashMap<String, Integer> hmap = new HashMap<String, Integer>();
		String[] Stanford_dependencies = new String[] {"acomp", "advcl", "advmod", "agent", "amod", "appos", "aux", "auxpass", "cc",
		"ccomp", "conj", "cop", "csubj", "csubjpass", "dep", "det", "discourse", "dobj", "expl", "goeswith", "iobj", "mark", "mwe", 
		"neg", "nn", "npadvmod", "nsubj", "nsubjpass", "num", "number", "parataxis", "pcomp", "pobj", "poss", "possessive", "preconj", 
		"predet", "prep", "prepc", "prt", "punct", "quantmod", "rcmod", "ref", "root", "tmod", "vmod", "xcomp", "xsubj"};		
		for (String one_type: Stanford_dependencies){
			hmap.put(one_type, 0);
		}
		System.out.println("final : " + hmap);
		
		String tweet = "My dog also likes sausage. I like more burgers.";
		ArrayList<String> ParserArray = new ArrayList<String>();
		for (String s:tweet.split("(?<=[.])\\s+")){
		    System.out.println(s);
			ArrayList<String> Return = Parser.main(s);
			ParserArray.addAll(Return);
			System.out.println(Return);
			System.out.println("\n");
		}
		
		for (String str: ParserArray){
			if(str.contains("nmod:")){
				str = str.replace("nmod:","");
			}
			if(hmap.get(str) != null){
				hmap.put(str, hmap.get(str)+1);
			}
			
		}
		System.out.println("final : " + hmap);
	}
}
