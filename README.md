# CCNN project

Weekly Log

Week 2.
a. Paraphrasing value generation..? :
  1. Calculating similarity between sentences can be done by using SpaCy. The algorithm which SpaCy uses is to compare the n-gram language model. They claim
  the context similarity is still reflected.
  2. Spacy is trained based on the word2vec. (GloVe corpus can also be used)
  - In terms of CC, will paraphrasing include negation often?
  - What about multiple sentences merged into a single sentence?

b. Grammar check is not perfect.. 
  - using nltk language model
  - using Google's ngram corpus

c. reading srt file:
  - matching start time with the caption words
  - how to detect a sentence? -> PERIOD AT THE END OF EVERY SENTENCE.
    - multiple caption blocks can be used to create a sentence.

Done: 
  - Spell check can be done.

Todo:
  - 


Week 1.
a. Sentence similarity
  - Spacy: the sentence similarity seems to be accurate, but the word similarities are questionable
      - The sentence similarities between antonyms are fairly high.
      - This is inaccurate because the sentence semantic similarity score should be completely opposite when antonym was used.

  - wordnet(synset): lexical database for the english language
      - The wordnet with Wu and Palmer similarity, which gives (~25%) for similarity between antonyms.
      - The synset also provides synonyms, antonyms, hypernyms/hyponym, meronyms
      - Lemma, verb frame, 
      - Hyponymy? red is hyponym of color (opposite is hypernym)
      - Meronymy? finger is meronym of hand (opposite is holonym)

b. What's done?
  - Transcript file and caption file (srt files) for the city news video that was used in the study, are generated.
  - Tested different methods to find semantic similarities for paraphrasing
  
c. What to do
  - Read the srt files and get two text caption blocks to be compared.
  - Implement the methods of calculate following:
    - Grammar/Spelling mistake
    - No missing words (does 'missing words'/ different number of words infer to paraphrasing?
      - Verbatim accuracy..?
    - Speaker ID? (not in weather forecasting though)
    - Speed of Captions (delay..?)

Week 0.
- Keras: high level framework on tensorflow
      - open sourced, support multiple backend (i.e. theano, CNTR, tensorflow)
- Word2Vec: These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.          - Word2vec takes as its input a large corpus of text and 
        produces a vector space with each unique word in the corpus being assigned a corresponding vector in the space.
    - Word vectors are positioned in the vector space such that        words that share common contexts in the corpus are 
        located in close proximity to one another in the space.
