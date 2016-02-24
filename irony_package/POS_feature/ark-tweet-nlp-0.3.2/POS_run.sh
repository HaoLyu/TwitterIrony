cur_dir=$(pwd)
example_dir=$cur_dir/examples
python ../import_DB_text_into_exampletxt.py
./runTagger.sh $example_dir/example_tweets.txt
python ../POS_count.py
python ../Cap_feature.py