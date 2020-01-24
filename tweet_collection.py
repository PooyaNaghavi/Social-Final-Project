from twython import TwythonStreamer
import json

tweets = []


class MyStreamer(TwythonStreamer):
    def on_success(self, data):

        if 'lang' in data and data['lang'] == 'en':
            if 'Trump' in data['text'] or 'POTUS' in data['text'] or 'Donald Trump' in data[
                'text'] or 'donaldjtrumpjr' in data['text'] or 'TRUMP' in data['text'] or 'realDonaldTrump' in data[
                'text'] or 'Donald Trump Jr.' in data['text'] or 'trump' in data['text'] or 'trumprussia' in data[
                'text']:
                tweets.append(data)
                print('received tweet #', len(tweets), data['text'][:500])

        if len(tweets) >= 1000:
            self.store_json()
            self.disconnect()

    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()

    def store_json(self):
        with open('Donald_Trump_Tweets.json'.format(len(tweets)), 'w') as f:
            json.dump(tweets, f, indent=4)


CONSUMER_KEY = open('ConsumerKey.txt', 'r').read()
CONSUMER_SECRET = open('ConsumerSecret.txt', 'r').read()
ACCESS_TOKEN = open('AccessToken.txt', 'r').read()
ACCESS_TOKEN_SECRET = open('AccessTokenSecret.txt', 'r').read()

while True:
    if len(tweets) < 100000:
        try:
            stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
            userids = ["25073877", "813286"]  # Trump and Obama's userids
            stream.statuses.filter(follow=userids)
        except:
            continue
    else:
        break
