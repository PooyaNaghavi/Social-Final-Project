import matplotlib.pyplot as plt
import json
import string
import nltk
import numpy as np
import utils
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from gensim import corpora, models

# nltk.download()   # use this once

norm_list = []

for person in ['Donald_Trump', 'Barack_Obama']:
    tweets = utils.get_sentences(f'{person}_tweets.json')
    for tweet in tweets:
        punctuation = string.punctuation
        table_punct = str.maketrans(punctuation, len(punctuation) * " ")
        norm = list(filter(lambda x: x in string.printable, tweet))
        doc = "".join(norm).translate(table_punct).lower()
        norm_list.append(doc)

    snst = SnowballStemmer("english")
    stemlist = []
    for tweet in norm_list:
        for word in tweet.split():
            if word.startswith('http') or word.startswith('https') or word.startswith('@'):
                continue
            else:
                stemlist.append(snst.stem(word))

    # 'Removing stop words from stemming applied words and collecting unique words'

    stopwords = nltk.corpus.stopwords.words('english')

    extrastop = ['trump', 'donald', 'RT', 'rt', 'http', 'https', 'lt', 'gt', 'realdonaldtrump', 'co', 'amp', 'today',
                 'via', 'wh', 'day', 'obama', 'barack', 'barackobama']
    uniextrastop = []

    for j in extrastop:
        uniextrastop.append(j)

    nostop = ''
    print(stemlist)
    for k in stemlist:
        if k not in stopwords and k not in uniextrastop and len(k) > 1:
            nostop += ' ' + k

    wordcld = WordCloud(max_font_size=50).generate(nostop)
    plt.figure()
    plt.imshow(wordcld)
    plt.axis("off")
    plt.savefig(f'img/{person}_wordcloud.png')

    # Question D
    # Reference - Dr. Gene Moo Lee notes for Data Science

    # 'Creating a stemmed and stop words removed corpus for topic modelling'
    # referencec - http://stackoverflow.com/questions/3627270/python-how-exactly-can-you-take-a-string-split-it-reverse-it-and-join-it-back

    stemmed_tweets_wd = []
    stemmed_tweets = []
    for tweet in norm_list:
        eachtweet = []
        for word in tweet.split():
            # print word
            if word.startswith('http') or word.startswith('https') or word.startswith('@'):
                continue
            elif word not in stopwords and word not in uniextrastop and len(word) > 1:
                stemming = snst.stem(word)
                eachtweet.append(stemming)
        tw = ' '.join(eachtweet)
        stemmed_tweets.append(tw)
        stemmed_tweets_wd.append(eachtweet)

    # 'To vectorize the text and check for unique words'

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    doc_term_matrix = vectorizer.fit_transform(stemmed_tweets)
    print(doc_term_matrix.shape)

    vocab = vectorizer.get_feature_names()

    # 'Non-negative matrix factorization from Scikit-Learn'

    # To print seven topics
    clf = decomposition.NMF(n_components=10, random_state=1)  # total 7 topics
    doctopic = clf.fit_transform(doc_term_matrix)
    print(clf.reconstruction_err_)

    for topic in clf.components_:
        # print topic.shape, topic[:10]
        word_idx = np.argsort(topic)[::-1][:10]  # 7 words in a topic
        print(word_idx)
        for idx in word_idx:
            if vocab[idx]:
                print(vocab[idx])
                continue

    # 'Latent Dirichlet Allocation (LDA) from GENSIM'
    # Reference - http://stackoverflow.com/questions/33229360/gensim-typeerror-doc2bow-expects-an-array-of-unicode-tokens-on-input-not-a-si

    # 'Creating a dictionary from the corpus of stemmed words from each tweet'
    dic = corpora.Dictionary(stemmed_tweets_wd)
    # print dic

    # 'Converting the stemmed words in a tweet to a bag of words using doc2bow'
    # 'The input is list of lists in which each list corresponds to words in each tweet'
    corpus = [dic.doc2bow(i) for i in stemmed_tweets_wd]
    # print(type(corpus), len(corpus))

    # 'Creating a model and printing six topics'

    tfidf = models.TfidfModel(corpus)
    # print(type(tfidf))

    corpus_tfidf = tfidf[corpus]

    model = models.ldamodel.LdaModel(corpus_tfidf, num_topics=10, id2word=dic, passes=4)
    # model.print_topics()

    topics_found = model.print_topics(10)
    counter = 1
    for t in topics_found:
        print(f'Topic #{counter} {t}')
        counter += 1
