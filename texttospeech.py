from gtts import gTTS
import os
import cv2

def load_text(file):
    with open(file,encoding='utf-8') as f:
        text = f.read()
    return text.lower()

text = load_text("out.txt")

from textblob import TextBlob
from translate import Translator

t = TextBlob('out.txt')
t = t.detect_language()
translator= Translator(from_lang=t,to_lang="english")
translation = translator.translate('out.txt')


## Summarization
import spacy
import textwrap
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
punctuation += '\n' 
stopwords = list(STOP_WORDS)

reduction_rate = 0.1  #defines how small the output summary should be compared with the input

nlp_pl = spacy.load('en_core_web_sm')     #process original text according with the Spacy nlp pipeline for english
document = nlp_pl(text)                   #doc object

tokens = [token.text for token in document] #tokenized text

word_frequencies = {}
for word in document:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

max_frequency = max(word_frequencies.values())
#print(max_frequency)

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

#print(word_frequencies)
sentence_tokens = [sent for sent in document.sents]

def get_sentence_scores(sentence_tok, len_norm=True):
  sentence_scores = {}
  for sent in sentence_tok:
      word_count = 0
      for word in sent:
          if word.text.lower() in word_frequencies.keys():
              word_count += 1
              if sent not in sentence_scores.keys():
                  sentence_scores[sent] = word_frequencies[word.text.lower()]
              else:
                  sentence_scores[sent] += word_frequencies[word.text.lower()]
      if len_norm:
        sentence_scores[sent] = sentence_scores[sent]/word_count
  return sentence_scores
                
sentence_scores = get_sentence_scores(sentence_tokens,len_norm=False)        #sentence scoring without lenght normalization
#sentence_scores_rel = get_sentence_scores(sentence_tokens,len_norm=True)     #sentence scoring with length normalization

def get_summary(sentence_sc, rate):
  summary_length = int(len(sentence_sc)*rate)
  summary = nlargest(summary_length, sentence_sc, key = sentence_sc.get)
  final_summary = [word.text for word in summary]
  summary = ' '.join(final_summary)
  return summary


sumer = open("summary.txt", "w+")
sumer.write("Summarized Report "+ get_summary(sentence_scores, reduction_rate))
sumer.close()