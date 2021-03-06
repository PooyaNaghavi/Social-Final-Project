from twython import TwythonStreamer
import json


class MyStreamer(TwythonStreamer):
    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret, person):
        self.person = person
        super(MyStreamer, self).__init__(app_key, app_secret, oauth_token, oauth_token_secret)

    def on_success(self, data):
        if 'lang' in data and data['lang'] == 'en':
            tweets.append(data)
            print('received tweet #', len(tweets), data['text'][:500])

        if len(tweets) >= 1000:
            self.store_json()
            self.disconnect()

    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()

    def store_json(self):
        with open(f'{self.person}_tweets.json', 'w') as f:
            json.dump(tweets, f, indent=4)


CONSUMER_KEY = open('ConsumerKey.txt', 'r').read()
CONSUMER_SECRET = open('ConsumerSecret.txt', 'r').read()
ACCESS_TOKEN = open('AccessToken.txt', 'r').read()
ACCESS_TOKEN_SECRET = open('AccessTokenSecret.txt', 'r').read()

for person in [('Donald_Trump', '25073877'), ('Barack_Obama', '813286')]:
    tweets = []
    while True:
        if len(tweets) < 100000:
            try:
                stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, person=person[0])
                userid = person[1]
                stream.statuses.filter(follow=userid)
            except:
                continue
        else:
            break
