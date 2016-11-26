import nltk
from collections import defaultdict

class Act_Tag:
	def __init__(self):
		self.tagged_data = nltk.corpus.nps_chat.xml_posts()[:10000]
		self.output = []
		self.act_set = [(self.__tokenize_sentence(post.text), post.get('class')) for post in self.tagged_data]
		self.classifier = nltk.NaiveBayesClassifier.train(self.act_set)

		
	def __tokenize_sentence(self, utterance):
		word_act = {}

		for word in nltk.word_tokenize(utterance):
			word_act['contains({})'.format(word.lower())] = True
		return word_act

	def get_act_tag(self, utterance):
		inp_utt = []
		for utt in utterance:
			inp_utt.append(self.__tokenize_sentence(utt))
		
		return self.classifier.classify_many(inp_utt)

# Remove the below comment for example on how to run this class
"""
act = Act_Tag()
inp = ["I am bored","What are your plans for tomorrow?"]
print(act.get_act_tag(inp))
"""
