#Reference - http://stackoverflow.com/questions/8548030/why-does-pip-install-inside-python-raise-a-syntaxerror
#import pip
#pip.main(['install','twython'])

#To Collect 10,000 tweets from twiiter using the keyword Trump Globally
#Reference - Dr. Gene Moo Lee notes for Data Science
from twython import TwythonStreamer
import json
 
tweets = []
 
class MyStreamer(TwythonStreamer):
    'Class to collect tweets and store in json'
    def on_success(self, data):
        
        if 'lang' in data and data['lang'] == 'en':
            tweets.append(data)
            print 'received tweet #', len(tweets), data['text'][:500]

        if len(tweets) >= 10000:
            self.store_json()
            self.disconnect()
                         
    def on_error(self, status_code, data):
        print status_code, data
        self.disconnect()
 
    def store_json(self):
        with open('trump_10000_tweets.json', 'w') as f:
            json.dump(tweets, f, indent=4)
 
 

CONSUMER_KEY = open('ConsumerKey.txt','r').read()
CONSUMER_SECRET = open('ConsumerSecret.txt','r').read()
ACCESS_TOKEN = open('AccessToken.txt','r').read()
ACCESS_TOKEN_SECRET = open('AccessTokenSecret.txt','r').read()

#stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)   
#stream.statuses.filter(track='trump')

#'To run this code without interruptions'


while True:
     if len(tweets) < 10000:
       try:
          stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)   
          stream.statuses.filter(track='trump')
       except:
          continue
     else:
          break



