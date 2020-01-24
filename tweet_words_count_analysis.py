import operator
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

from utils import get_sentences


for person in ['Donald_Trump']:
	sentences = get_sentences('{}_tweets.json'.format(person))
	sentences_text_blob = TextBlob('')
	
	for sentence in sentences:
		sentences_text_blob += TextBlob(sentence)

	sentences_word_counts = sentences_text_blob.word_counts
	invalid_keys = ['the', 'to', 'a', 'of', 's', 'obama', 'in', 'for', 'and', 'you', u'\u2019', 'is', 'on',
	 'we', u'\u2014president', 'president', 'that', 'it', 'this', 'our', 'their', 'out',
	 'your', 'i', 'http', u'\u201c', u'\u201d', 'have', 'are', 'more', 'at', 'with', 'if', 'be', 'can', 'will', 'not',
	 "n't", 'from', 'who', 'as', 'do', 're', 'all', 'should', 'has', 'has', 'get', 'what', 
	  'time', 'by', 'up', 'today', 'about', 'than', 'make', 'america', 't', 'now', 'they',
	  've', 'an', 'american', 'his', 'one', 'just', 'us', 'or', 'day', 'new', 'romney', 
	  'he', 'here', 'my', 'was', 'so', 'would', 'when', 'how', 'run']

	sentences_word_counts = {x: sentences_word_counts[x] for x in sentences_word_counts.keys() if x not in invalid_keys}

	sorted_sentences_word_counts = sorted(sentences_word_counts.items(), key=operator.itemgetter(1), reverse=True)[:20]

	words_count = [word[1] for word in sorted_sentences_word_counts]
	words_name = [word[0] for word in sorted_sentences_word_counts]

	print(words_name)
	print(words_count)

	y_positions = np.arange(len(words_name))
 	
 	plt.figure(figsize=(14,10))
 	plt.bar(y_positions, words_count, align='center', alpha=0.5, color='g')
	plt.xticks(y_positions, words_name, rotation=90)
	plt.ylabel('counts')
	plt.title('words')
	plt.savefig('img/{}_tweet_lengths_words_count.png'.format(person))
	plt.show()
