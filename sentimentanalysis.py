import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np


polarity = []
subjectivity = []

with open('Donald_Trump_Tweets.json', 'r') as tweetfile:
    jsonread = json.load(tweetfile)

for j in range(0, len(jsonread)):
    eachtweet = jsonread[j]['text']
    eachtweetwd = eachtweet.split()
    tweetnohyp = []
    'To remove hyperlinks and usernames from tweets and perform sentiment analysis'
    for word in eachtweetwd:
        if word.startswith('http') or word.startswith('https') or word.startswith('@'):
            continue
        else:
            tweetnohyp.append(word)
    tweetnohypstr = " ".join(tweetnohyp)
    sentan = TextBlob(tweetnohypstr)
    polarity.append(float(sentan.polarity))
    subjectivity.append(float(sentan.subjectivity))


(n, bins, patches) = plt.hist(polarity, bins=10, color='pink')
plt.xlabel('polarity')
plt.ylabel('tweetcount')
plt.axvline(np.mean(polarity), color='b', linestyle='dashed', linewidth=1.5)
plt.show()
print('number of tweets in each bin for polarity histogram')
print(n)


plt.hist(subjectivity, bins=10, color='green')
plt.xlabel('subjectivity')
plt.ylabel('tweetcount')
plt.axvline(np.mean(subjectivity), color='b', linestyle='dashed', linewidth=1.5)
plt.show()

avgpolarity = sum(polarity) / len(polarity)
avgsubjectivity = sum(subjectivity) / len(subjectivity)

print('average polarity is {}'.format(avgpolarity))
print('average subjectivity is {}'.format(avgsubjectivity))
