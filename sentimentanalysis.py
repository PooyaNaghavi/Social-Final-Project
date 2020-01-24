#Reference - http://stackoverflow.com/questions/8548030/why-does-pip-install-inside-python-raise-a-syntaxerror
#import pip
#pip.main(['install','textblob'])
#pip.main(['install','matplotlib'])

#To do sentiment Analysis on the collected 10k tweets, plot histogram and calculate average subjectivity and polarity
#Reference - http://stackoverflow.com/questions/493819/python-join-why-is-it-string-joinlist-instead-of-list-joinstring
#Reference - Dr. Gene Moo Lee notes for Data Science


import json
from textblob import TextBlob
import matplotlib.pyplot as matp
import numpy as np

#'To check polarity and subjectivity of each tweet and append them to respective list'

polarity=[]
subjectivity=[]

with open('trump_10000_tweets.json','r') as tweetfile:
     jsonread=json.load(tweetfile)

for j in range(0,len(jsonread)):
    eachtweet = jsonread[j]['text']
    eachtweetwd = eachtweet.split()
    tweetnohyp=[]
    'To remove hyperlinks and usernames from tweets and perform sentiment analysis'
    for word in eachtweetwd:
        if word.startswith('http') or word.startswith('https') or word.startswith('@'):
            continue
        else:
            tweetnohyp.append(word)
    tweetnohypstr = " ".join(tweetnohyp)
    sentan=TextBlob(tweetnohypstr)
    polarity.append(float(sentan.polarity))
    subjectivity.append(float(sentan.subjectivity))      
   

#'Creating histogram for polarity. To count the number of positive and negative polarity tweets'
#Reference - http://stackoverflow.com/questions/16180946/drawing-average-line-in-histogram-matplotlib
#Reference - http://stackoverflow.com/questions/9039961/finding-the-average-of-a-list
#Reference - http://stackoverflow.com/questions/19442224/getting-information-for-bins-in-matplotlib-histogram-function

(n, bins, patches) = matp.hist(polarity, bins=10, color='pink')
matp.xlabel('polarity')
matp.ylabel('tweetcount')
matp.axvline(np.mean(polarity), color='b', linestyle='dashed', linewidth=1.5)
matp.show()
print 'number of tweets in each bin for polarity histogram'
print n

#'To get the number of positive and negative polarity tweets'
#(n, bins, patches) = matp.hist(polarity, bins=10, label='polarity')
#matp.show()
#print n

#'Creating histogram for subjectivity'
#Reference - http://stackoverflow.com/questions/16180946/drawing-average-line-in-histogram-matplotlib
#Reference - http://stackoverflow.com/questions/9039961/finding-the-average-of-a-list
matp.hist(subjectivity, bins=10, color='green')
matp.xlabel('subjectivity')
matp.ylabel('tweetcount')
matp.axvline(np.mean(subjectivity), color='b', linestyle='dashed', linewidth=1.5)
matp.show()

#'Average polarity and average subjectivity calculation'
avgpolarity=sum(polarity)/len(polarity)
avgsubjectivity=sum(subjectivity)/len(subjectivity)

print 'average polarity is {}'.format(avgpolarity)
print 'average subjectivity is {}'.format(avgsubjectivity)
