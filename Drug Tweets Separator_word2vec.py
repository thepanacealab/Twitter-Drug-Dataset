import itertools 
import time
import os
import gensim
import re
import math
import sys
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy import spatial
import sent2vec
#from nltk.tokenize import word_tokenize

#####################
stanard_dataset = {}
tweets = {}
stanard_dataset_path ='adrmine_combine_tweet_tokenized.txt'
tweets_path ='tweettext2018_tokenized.txt'
size0fvector=300
#####################

# for each run, please load one of word2vec.bin or sent2vec.bin due to memory problem and so comment the other's loading

#load word2vec
word2vec_model = Word2Vec.load('/mnt/local/hdd/Javadr/trained_models/Word2Vec/300/wiki/word2vec.bin')

'''
#load sent2vec
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('/mnt/local/hdd/Javadr/trained_models/Sen2Vec/700/sent2vec_twitter_bigrams.bin')
'''

def wordTovec(sentence):
	global word2vec_model
	vocab = list(word2vec_model.wv.vocab.keys())
	word_tokens = sentence.split(' ') # or  word_tokens = word_tokenize(sentence)
	i =0
	v = [0]*size0fvector
	for word in word_tokens:
		if word in vocab:
			if i ==0:
				v = word2vec_model[word]
			else:
				v = np.add(v,word2vec_model[word])
			i = i+1
	return v
	

def sentTovec(sentence):
	global sent2vec_model
	return sent2vec_model.embed_sentence(sentence)

def similarity(v1,v2):
	return (1 - spatial.distance.cosine(v1, v2))

#[avg,min,max]
def get_score(tweet):
	global stanard_dataset
	v = wordTovec(tweet)
	min= 2.0
	max=-2.0
	sum=0.0
	for item in stanard_dataset:
		score = similarity(v, stanard_dataset[item])
		if(min>score):
			min = score
		if(max<score):
			max = score
		sum = sum + score
	avg = sum/len(stanard_dataset)
	return [avg,min,max]		

def main():
	global stanard_dataset, tweets
	print('00000')
	fi = open(stanard_dataset_path,'r')
	for line in fi:
		stanard_dataset[line.replace('\n','')] = wordTovec(line.replace('\n',''))
		#print(line)
	fi.close()
	print('11111')
	fi = open(tweets_path,'r')
	for line in fi:
		if(len(line)<3):
			continue
		tweets[line.replace('\n','')]= -1
	fi.close()
	print('2222')
	for key in tweets:
		tweets[key]=get_score(key)
		#print(key,tweets[key])
		#x= int(input('Enter a number:'))

	tweets = {k:v for k, v in sorted(tweets.items(), key=lambda t: t[1][0],reverse=True)}
	fo=open('word2vec_avg.txt','w')
	for key in tweets:
		fo.write(key+'\n')
	fo.close()
	
	fo=open('word2vec_min.txt','w')
	tweets = {k:v for k, v in sorted(tweets.items(), key=lambda t: t[1][1],reverse=True)}
	for key in tweets:
		fo.write(key+'\n')
	fo.close()

	fo=open('word2vec_max.txt','w')
	tweets = {k:v for k, v in sorted(tweets.items(), key=lambda t: t[1][2],reverse=True)}
	for key in tweets:
		fo.write(key+'\n')
	fo.close()


if __name__ == "__main__":
	main()



