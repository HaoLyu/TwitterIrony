use IronyHQ
db.createCollection("tweets")
$mongoimport -d IronyHQ -c tweets --file bamman.json --drop
