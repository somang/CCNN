# CCNN project




What are my contributions?

- I have designed and developed a “subjective” Closed Captioning quality assessment method.
- I have devised an “automated” assessment system using the subjective quality assessment.
- It is possible to predict subjective assessment of D/HoH viewers using the automated system.

Lit review:
- Why do you need the subjective CCQ method?
    - > current models are primarily empirical, even though some sections allow the subjectivity of assessments to be included.
    - The NER model subjectivity could be more accurate, but it doesn’t include D/HoH viewer perspective
    - Costly to include human
        - > Why automation? 
        - > to reduce HR
How does an AI predict ‘subjective assessment’? isn’t that self-contradictory? -> 

CCQNN
Design constraints
	-> lack of data
		-> user model for the probabilities in decision
		-> generated for ANN training / statistical regression fitting.

Study 1:
	User model generation
	
	Research question: What is the probabilities in assessment decisions when D/HoH viewers find errors in CC?
		
	
Study 2:
	Training ANN by using active learning













Weekly Development Log

Week 14. Update on April 11
- Will it be possible to feed the 'features' to be natural?
  - i.e. such that, instead of feeding generated values,
  natural values will be 
    1. vectorized_caption_sentence
    2. vectorized_transcript_sentence
    3. caption_start_time
    4. caption_end_time
    5. transcript_start_time
    6. transcript_end_time
    7. genre (maybe?)
    8. viewer (0=Deaf, 1=HOH, 2=ESL, 3=Others)
  - How to vectorize the sentences?
    1) Skip gram method: paper here and the tool that uses it, google word2vec
    2) Using LSTM-RNN to form semantic representations of 
    sentences.

- DATA prep: new data is here. from BAF. create caption and transcript.




Week 12. Update on April 2
- Generating data based on median and quartiles can only be done using maximum likelihood function: and it wouldn't give the proper quartiles...
- Another method is to fit the existing data points to a curve fit. (i.e. polynomial)

Week 11. On March 29th: todo by April 3rd.

- Next week, presentation for the BAF group.
  - talk about the motivation (why this framework?)
  - talk about the progress, methods, performance rate
  - future work?

- Finish writing literature review
  - Quality factors?
  - ANN?
- Finish writing data, architecture part.
- What to write in the discussion??

Week 9. March 14th
1. Paraphrasing Factor:
  a. Check the sentence similarity (cosine similarity)
  b. if it's 100% then it is a verbatim
  c. else, then chek whether there are missing words
    compare two sentences, and find the missing words
    -> generate number of missing words
  d. if there are no grammar mistakes,
    -> then paraphrasing is 1
    else: 
      give it a 0

2. make ccnn to have
  a. number of spelling mistakes
  b. number of grammar mistakes
  c. number of missing words

3. generated data -> fit the model

4. Write ASSETS paper:
  A. lit review from qual report
  
  B. data generation
    a. how did you handle paraphrasing?
    
  C. main neural net architecture
    number of neurons, layers,
    keras, activation functions
  
  D. discussion:
    performance -> compare with other models
  
  E. future work




Week 6-7. March 7th
Todo:
1. Generate data that can train the NN to be 'reasonable' and 'rational'
 - Should be modified to fit the sentence by sentence wise.
2. Use numpy/other library to find the statistical regression model.


Week 5. Feb 21st
TODo:
1. Create another caption file, and transcript for CTV video.
2. Organize, and gather up methods in class, 
   using sentence by sentence for data generation.
3. Filter the data, to create a numpy matrix along with the result from 2.
4. Train the NN with the numpy matrix and generated values.

Week 4. Feb 14th
0. CCQNN
a. CC Neural Net
  - Turns out what we have can be enough to represent the population.
  - Given 85 (with 12 D) D/HOH data * 2 videos each
  - The input will be values generated from caption file which is compared against the transcript file
  - Then, the ratings of the participants preferences or
      - Q: how did each 'factor' affected the pleasure?
      - factors include: 
      1. fast appearing/disappearing (speed)
      2. slow appearing/disappearing (delay)
      3. missing words (how many words are missing)
      4. spelling/grammar errors (how many errors)
      5. speaker identification (are there any?)
      6. verbatim accuracy (paraphrasing)
b. value generation from srt file.
  - should work on the delay calculation
  - Should work on how many characters are in a single caption 'line'
c. srt file parser is done.

Week 3.
a. CC Neural Net
  - Data: Insufficient data points- but can be generated from normal distribution..?
    - seeding from what's already existing data
b. Value generation from srt file
  Todo
  - Speed of Caption generation
  - Delay calculation
  - Paraphrasing
  Done
  - Spelling check
  - Antonym detection
  - Apostrophe use-Lemmatization
c. srt file parser
  Todo
  - Sentence generation for multiple sentence endings in a single caption object
  Done
  - reading
  - parsing the srt file into caption objects

Week 2. Feb 7nd
a. Paraphrasing value generation..? :
  1. Calculating similarity between sentences can be done by using SpaCy. The algorithm which SpaCy uses is to compare the n-gram language model. They claim
  the context similarity is still reflected.
  2. Spacy is trained based on the word2vec. (GloVe corpus can also be used)
  - In terms of CC, will paraphrasing include negation often?
  - What about multiple sentences merged into a single sentence?
b. CCNN:
  - First layer input: values from the CC files
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
  - srt reader script
  - neural values generation script

Week 1. Jan 31
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
Week 0. Jan 24
- Keras: high level framework on tensorflow
      - open sourced, support multiple backend (i.e. theano, CNTR, tensorflow)
- Word2Vec: These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.          
- Word2vec takes as its input a large corpus of text and 
        produces a vector space with each unique word in the corpus being assigned a corresponding vector in the space.
    - Word vectors are positioned in the vector space such that        words that share common contexts in the corpus are 
        located in close proximity to one another in the space.
