import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import utils

for person in ['Donald_Trump', 'Barack_Obama']:
    polarity = []
    subjectivity = []
    tweets = utils.get_sentences(f'{person}_tweets.json')
    for tweet in tweets:
        sent_an = TextBlob(tweet)
        polarity.append(float(sent_an.polarity))
        subjectivity.append(float(sent_an.subjectivity))

    plt.figure()
    (n, bins, patches) = plt.hist(polarity, bins=10, color='pink')
    plt.xlabel('polarity')
    plt.ylabel('tweet count')
    plt.axvline(np.mean(polarity), color='b', linestyle='dashed', linewidth=1.5)
    plt.savefig(f'img/{person}_tweets_polarity.png')

    plt.figure()
    plt.hist(subjectivity, bins=10, color='green')
    plt.xlabel('subjectivity')
    plt.ylabel('tweet count')
    plt.axvline(np.mean(subjectivity), color='b', linestyle='dashed', linewidth=1.5)
    plt.savefig(f'img/{person}_tweets_subjectivity.png')

    avg_polarity = sum(polarity) / len(polarity)
    avg_subjectivity = sum(subjectivity) / len(subjectivity)

    print('average polarity is {}'.format(avg_polarity))
    print('average subjectivity is {}'.format(avg_subjectivity))
