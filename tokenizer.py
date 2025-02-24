import sentencepiece as spm



######################
## Defining Constants 

dataset_path = "text/processed_brainrot_trimmed.txt"
tokenizer_prefix = "brainrot_tokenizer"

 #####################





######################
## Tokenizer training 


spm.SentencePieceTrainer.train(
    input=dataset_path, 
    model_prefix="brainrot_tokenizer", 
    vocab_size=1000, 
    model_type="unigram",  
    character_coverage=1
)

#####################
