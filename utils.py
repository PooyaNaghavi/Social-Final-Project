import json


def get_sentences(file):
    with open(file, 'r') as tweetfile:
        jsonread = json.load(tweetfile)
    result = []
    for j in range(0, len(jsonread)):
        eachtweet = jsonread[j]['text']
        eachtweetwd = eachtweet.split()
        tweetnohyp = []
        for word in eachtweetwd:
            if word.startswith('http') or word.startswith('https') or word.startswith('@'):
                continue
            else:
                tweetnohyp.append(word)

        tweetnohypstr = " ".join(tweetnohyp)
        result.append(tweetnohypstr)
    return result