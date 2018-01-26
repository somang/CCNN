# CCNN project

Weekly log

Week 2.

Paraphrasing value generation..? :
  1. 





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
