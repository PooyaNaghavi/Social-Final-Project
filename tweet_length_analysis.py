import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import utils

for person in ['Donald_Trump', 'Barack_Obama']:
    data_positive = []
    data_negative = []
    tweets = utils.get_sentences(f'{person}_tweets.json')
    for tweet in tweets:
        sentan = TextBlob(tweet)
        polarity = float(sentan.polarity)
        if polarity < 0:
            data_negative.append(len(tweet))
        else:
            data_positive.append(len(tweet))

    for data in [(data_positive, 'positive'), (data_negative, 'negative')]:
        plt.figure()
        sns.distplot(data[0])
        plt.xlabel('Tweet Length')
        plt.savefig(f'img/{person}_tweet_lengths_{data[1]}.png')
