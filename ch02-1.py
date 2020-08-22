# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:01:03 2020

@author: Young Hun Park
"""
#%% 1
from nltk.tokenize import word_tokenize

print(word_tokenize("Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#%% 2
from nltk.tokenize import WordPunctTokenizer

print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#%% 3
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

#%% 4
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own"
print(tokenizer.tokenize(text))

#%% 5
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print(sent_tokenize(text))
#%% 6
from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D.student"
print(sent_tokenize(text))

#%% 7
import kss

text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니예요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))
#%% 8
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
text="I am actively looking for Ph.D. students and you are a Ph.D. student."
print(word_tokenize(text))
x=word_tokenize(text)
pos_tag(x)

#%% 9
from konlpy.tag import Okt
okt=Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐"))

#%% 10
from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))











