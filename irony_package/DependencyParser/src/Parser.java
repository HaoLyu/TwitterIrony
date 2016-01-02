import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;


public class Parser {

    private final static String PCG_MODEL = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";        

    private final TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "invertible=true");

    private final LexicalizedParser parser = LexicalizedParser.loadModel(PCG_MODEL);

    public Tree parse(String str) {                
        List<CoreLabel> tokens = tokenize(str);
        Tree tree = parser.apply(tokens);
        return tree;
    }

    private List<CoreLabel> tokenize(String str) {
        Tokenizer<CoreLabel> tokenizer =
            tokenizerFactory.getTokenizer(
                new StringReader(str));    
        return tokenizer.tokenize();
    }

    public static ArrayList<String> main(String tweet) { 
    	ArrayList<String> DependencyReturn = new ArrayList<String>();
        //String str = "My dog also likes eating sausage. I don't like it. #Irony";
        Parser parser = new Parser(); 
        Tree tree = parser.parse(tweet);  

        List<Tree> leaves = tree.getLeaves();
        // Print words and Pos Tags
        //for (Tree leaf : leaves) { 
            //Tree parent = leaf.parent(tree);
            //System.out.print(leaf.label().value() + "-" + parent.label().value() + " ");
        //}
        
	     // Get dependency tree
	     TreebankLanguagePack tlp = new PennTreebankLanguagePack();
	     GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
	     GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
	     Collection<TypedDependency> td = gs.typedDependenciesCollapsed();
	     //System.out.println(td);
	     
	     Object[] list = td.toArray();
	     TypedDependency typedDependency;
	     for (Object object : list) {
	     typedDependency = (TypedDependency) object;
	     DependencyReturn.add(typedDependency.reln().toString());
	     //System.out.println("Depdency Name:  "+typedDependency.dep().toString()+ " :: "+ "Node:  "+typedDependency.reln());
	     }
	     return DependencyReturn;
	}
}



