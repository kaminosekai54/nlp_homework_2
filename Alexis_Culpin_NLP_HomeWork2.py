# Import
import pandas as pd
import nltk
from pprint import pprint # for printing objects nicely
from gensim import corpora, models
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from random import choice
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

################################################################################
#Answering the question :
# Q: Why is it called bag-of-words? 
# It's called bag of words because all the information about the order, meaning, length or any other information on the word is removed.
# the only information we have is the presence or absence of the word, and with multiple lines, it's frequency.
# just like the bag of balls, we only have it or not, and the frequency of it.




################################################################################
# code
np.random.seed(1234)

# I choose the dataset of papers published in NIPS conference
# found it here : https://www.kaggle.com/datasets/rowhitswami/nips-papers-1987-2019-updated/code?resource=download

# importing the articles
df = pd.read_csv("papers.csv",
sep=',', 
                 on_bad_lines='skip' 
)
print(df.columns)

# now we clean the data
df = df.drop(columns=["source_id","year","title","abstract"], axis=1)
df = df.dropna()
# we can lower all the token because it's english text and capital letter have less meaning than other langage
df.full_text= df.full_text.apply(lambda x: x.lower())


print(df.head(5))





# stemming preparation
stemmer = SnowballStemmer('english')
english_stop_words = set(stopwords.words('english'))

# function from the class notbook for steming and preprocessing
def lemmatize_stemming(text):
  return stemmer.stem(text)

def preprocess(text):
  try :
    result = [lemmatize_stemming(token) for token in simple_preprocess(text) if token not in english_stop_words and len(token) > 3]
    return  result
  except:
    print(text)
    # for w in text: 
      # if type(w) != str: print(w)

def comparVocLength():
    print('original document: ')
    article = choice(all_articles)
#   print(article)

    # This time, we don't care about punctuations as tokens (Can you think why?):
    print('original document, broken into words: ')
    words = [word for word in article.split(' ')]
#   print(words)
    print("Vocabulary size of the original article:", len(set(words)))

    print('\n\n tokenized and lemmatized document: ')
    preprocessed_article = preprocess(article)
    # print(preprocessed_article)
    print("Vocabulary size after preprocessing:", len(set(preprocessed_article)))

# now we convert each article into one element of a list
all_articles = df.full_text.to_list()
# now, we preprocess all our article
print("preprocessing of all the article")
processed_docs = list(map(preprocess, all_articles))
# print(processed_docs[:10])

# now we prepare our vocabulary dictionnary 
print("vocab dict preparation")
dictionary = corpora.Dictionary(processed_docs)
# dictionary.save("dict.txt")
## Model hyper parameters:

## These are the dictionary preparation parameters:
filter_tokens_if_container_documents_are_less_than = 30
filter_tokens_if_appeared_percentage_more_than = 0.2
# I change it to increase the bow word, speciaally when I have more topics, it help to decrease the perplexity
keep_the_first_n_tokens=100000

## and the LDA Parameters: 
num_of_topics = 15
# I choose this number because it still a low number, and the match score for the lowest still acceptable,
# I choose to increase it following the graph in the original article, where we can see that the lower perplexity is, the better it is, and topics amount seams to decrease it.

dictionary.filter_extremes(
    no_below=filter_tokens_if_container_documents_are_less_than, 
    no_above=filter_tokens_if_appeared_percentage_more_than, 
    keep_n=keep_the_first_n_tokens)

# bag of word creation
print("creation of the bag of word")
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# LDA model :
print("creation of lda model")
lda_model = models.LdaMulticore(bow_corpus, 
                                num_topics=num_of_topics, 
                                id2word=dictionary, 
                                passes=5, 
                                workers=4)
print("lda model finish")

# printing the topics and the words that compose them
for idx, topic in lda_model.print_topics(num_of_topics):
    print(f'Topic: {idx} \t Words: {topic}')

#tf/idf 
tfidf = models.TfidfModel(bow_corpus)
# apply it on our corpus
tfidf_corpus = tfidf[bow_corpus]

pprint(tfidf_corpus[0][:10])
# new lda with the tf/idf
print("creating tf/idf lda model")
lda_model_tfidf = models.LdaMulticore(tfidf_corpus, 
                                      num_topics=num_of_topics, 
                                      id2word=dictionary, 
                                      passes=5, 
                                      workers=4)

print("topics and words for the tf/idf lda model")
for idx, topic in lda_model_tfidf.print_topics(num_of_topics):
  print(f'Topic: {idx} \t Word: {topic}')


# Applying my model on randomly picked article
test_doc = choice(range(len(processed_docs)))

# Using the original BOW model:

for index, score in sorted(lda_model[bow_corpus[test_doc]], key=lambda tup: -1*tup[1]):
    print(f"Topic match score: {score} \nTopic: {lda_model.print_topic(index, num_of_topics)}")

# And with the TF/IDF model:
for index, score in sorted(lda_model_tfidf[bow_corpus[test_doc]], key=lambda tup: -1*tup[1]):
    print("Topic match score: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, num_of_topics)))

    # Calculating the [perplexity score](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94) (lower is better):"""

print('Perplexity: ', lda_model.log_perplexity(bow_corpus)) 
print('Perplexity TFIDF: ', lda_model_tfidf.log_perplexity(bow_corpus))


# visualisation
bow_lda_data = gensimvis.prepare(lda_model, bow_corpus, dictionary)
bow_lda_tfidf_data = gensimvis.prepare(lda_model_tfidf, bow_corpus, dictionary)
pyLDAvis.display(bow_lda_data)
pyLDAvis.display(bow_lda_tfidf_data )

