#%% 1 
# 길이가 1~2인 단어들을 삭제
import re
text="I was wondering if anyone out there could enlighten me on this car"
shortword=re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('',text))

#%% 2
# 표제어 추출
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy','doing','organization','have','going','love','lives','fly','dies','watched','has','starting']
print([n.lemmatize(w) for w in words])

#%% 3
#품사를 정보를 포함한 표제어 추출
print(n.lemmatize('dies','v'))
print(n.lemmatize('watched','v'))
print(n.lemmatize('has','v'))

#%% 4 
#어간 추출(proter 알고리즘)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s=PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate\
 copy, complete in all things--names and heights and soundings--with the single\
 exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)
print([s.stem(w) for w in words])

words=['formalize','allowance','electrical']
print([s.stem(w) for w in words])

#%% 5
# 어간 추출(Lancaster stemmer 알고리즘)
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
s=PorterStemmer()
l=LancasterStemmer()

words=['policy','doing','organization','have','going','love','lives','fly','dies','watched','has','starting']
print([s.stem(w) for w in words])
print([l.stem(w) for w in words])
# %% 6
from nltk.corpus import stopwords
stopwords.words('english')[:10]
