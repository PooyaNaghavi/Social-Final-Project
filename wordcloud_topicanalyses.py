#Reference - http://stackoverflow.com/questions/8548030/why-does-pip-install-inside-python-raise-a-syntaxerror

#import pip
#pip.main(['install','gensim'])
#pip.main(['install','nltk']) 


#Question C
#Reference - Dr. Gene Moo Lee notes for Data Science

import matplotlib.pyplot as matp
import json
import string
import nltk
#nltk.download()
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer


norm_list=[] 
with open('trump_10000_tweets.json','r') as tweetfile:
     jsonread=json.load(tweetfile)     
     for i in range(0,len(jsonread)):
              unicodedeachtweet = jsonread[i]['text']
              unicodedeachtweetwd=unicodedeachtweet.split()
              unicodedeachtweetline=[]
              'To remove hyperlinks and usernames from tweets'
              for word in unicodedeachtweetwd:
                   if word.startswith('http') or word.startswith('https') or word.startswith('@'):
                       continue
                   else:
                       unicodedeachtweetline.append(word)
              unicodedfulltweet= " ".join(unicodedeachtweetline)
              punct=string.punctuation
              table_punct=string.maketrans(punct, len(punct) * " ")
              norm=filter(lambda x: x in string.printable, unicodedfulltweet)
              doc=str(norm).translate(table_punct).lower()
              norm_list.append(doc)
     

'Snowball stemming'
snst = SnowballStemmer("english")
stemlist=[]
for tweet in norm_list:
    'To remove hyperlinks and usernames from tweets'
    for word in tweet.split():
        #print word
        if word.startswith('http') or word.startswith('https') or word.startswith('@'):
             continue
        else:
             stemlist.append(snst.stem(word))
        
        
        
#'Removing stop words from stemming applied words and collecting unique words'

stopwords = nltk.corpus.stopwords.words('english')

extrastop = ['trump','donald','RT','rt','http','https','lt','gt','realdonaldtrump','co','amp','today','via','wh','day']
uniextrastop=[]
for j in extrastop:
    uniextrastop.append(j.decode('utf-8'))
 
nostop=''
for k in stemlist:
    if k.decode('utf-8') not in stopwords and k not in uniextrastop and len(k)>1:
        nostop += ' ' + k        
        
#'Word cloud generation'
wordcld = WordCloud(max_font_size=50).generate(nostop)
matp.figure()
matp.imshow(wordcld)
matp.axis("off")
matp.show()



# Question D
#Reference - Dr. Gene Moo Lee notes for Data Science

#'Creating a stemmed and stop words removed corpus for topic modelling'
#referencec - http://stackoverflow.com/questions/3627270/python-how-exactly-can-you-take-a-string-split-it-reverse-it-and-join-it-back

stemmedtweetswd=[]
stemmedtweets=[]
for tweet in norm_list:
    eachtweet=[]      
    for word in tweet.split():
        #print word
        if word.startswith('http') or word.startswith('https') or word.startswith('@') :
            continue
        elif word.decode('utf-8') not in stopwords and word.decode('utf-8') not in uniextrastop and len(word)>1:               
                stemming = snst.stem(word)
                eachtweet.append(stemming)
    tw= ' '.join(eachtweet)
    stemmedtweets.append(tw)
    stemmedtweetswd.append(eachtweet)        
    

#'To vectorize the text and check for unique words' 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
doc_term_matrix = vectorizer.fit_transform(stemmedtweets)
print doc_term_matrix.shape
vocab = vectorizer.get_feature_names()


#'Non-negative matrix factorization from Scikit-Learn'

from sklearn import decomposition
#To print seven topics
clf = decomposition.NMF(n_components=7, random_state=1) #total 7 topics
doctopic = clf.fit_transform(doc_term_matrix)
print clf.reconstruction_err_


import numpy as np
for topic in clf.components_:
    #print topic.shape, topic[:10]
    word_idx = np.argsort(topic)[::-1][:7] #7 words in a topic
    print word_idx
    for idx in word_idx:
        if vocab[idx]:
            print vocab[idx]
            continue
        

#'Latent Dirichlet Allocation (LDA) from GENSIM'
#Reference - http://stackoverflow.com/questions/33229360/gensim-typeerror-doc2bow-expects-an-array-of-unicode-tokens-on-input-not-a-si
   
from gensim import corpora
#'Creating a dictionary from the corpus of stemmed words from each tweet'
dic = corpora.Dictionary(stemmedtweetswd)
#print dic

#'Converting the stemmed words in a tweet to a bag of words using doc2bow'
#'The input is list of lists in which each list corresponds to words in each tweet'
corpus = [dic.doc2bow(i) for i in stemmedtweetswd]
#print(type(corpus), len(corpus))

#'Creating a model and printing six topics'

from gensim import models
tfidf = models.TfidfModel(corpus)
#print(type(tfidf))


corpus_tfidf = tfidf[corpus]

model = models.ldamodel.LdaModel(corpus_tfidf, num_topics=6, id2word=dic, passes=4)
#model.print_topics()

topics_found = model.print_topics(6)#Printing 6 topics
counter = 1
for t in topics_found:
    print("Topic #{} {}".format(counter, t))
    counter += 1

