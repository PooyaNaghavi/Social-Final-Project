#import pip
#pip.main(['install','twython'])

#Question E
#Reference - Dr. Gene Moo Lee notes for Data Science

from twython import TwythonStreamer
import json
#import time
 
tweets = []
 
class MyStreamer(TwythonStreamer):
#'Class to collect tweets and store in json'
    def on_success(self, data):
        
        if 'lang' in data and data['lang'] == 'en':
            if 'Trump' in data['text'] or 'POTUS' in data['text'] or 'Donald Trump' in data['text'] or 'donaldjtrumpjr' in data['text'] or 'TRUMP' in data['text'] or 'realDonaldTrump' in data['text'] or 'Donald Trump Jr.' in data['text'] or 'trump' in data['text'] or 'trumprussia' in data['text']:
                tweets.append(data)
                print 'received tweet #', len(tweets), data['text'][:500]

        if len(tweets) >= 1000:
            self.store_json()
            self.disconnect()
                         
    def on_error(self, status_code, data):
        print status_code, data
        self.disconnect()
 
    def store_json(self):
        with open('tweet_stream_trump_CA_1000.json'.format(len(tweets)), 'w') as f:
            json.dump(tweets, f, indent=4)
 
 

CONSUMER_KEY = open('ConsumerKey.txt','r').read()
CONSUMER_SECRET = open('ConsumerSecret.txt','r').read()
ACCESS_TOKEN = open('AccessToken.txt','r').read()
ACCESS_TOKEN_SECRET = open('AccessTokenSecret.txt','r').read()

#'''
#stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)   
#stream.statuses.filter(track='trump')
#'''

#'To run this code without interruptions'
#Reference - https://dev.twitter.com/streaming/overview/request-parameters#track
#locations=[-103.381348,29.333101,-94.240723,33.237539] - Texas
#locations=[-79.387207,42.023793,-73.674316,44.737954] - New York
#locations=[-82.287598,27.321755,-81.496582,30.475899] - Florida
#locations=[-122.761230,37.986422,-120.080566,41.794864] - California
#locations=[-99.558105,34.405777,-81.979980,36.375962] - Oklahoma, Louisiana, Alabama, Tennessee

while True:
      if len(tweets) < 1000:
        try:
          stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)   
          stream.statuses.filter(locations=[-122.761230,37.986422,-120.080566,41.794864])
        except:
          #time.sleep(10)
          continue
      else:
          break



