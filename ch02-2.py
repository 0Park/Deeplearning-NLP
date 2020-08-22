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
# %%7
#불용어 제거(영어)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="Family is not an important thing. It's everything."
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
print(word_tokens)
print(result)

#%% 8
#불용어 제거(한글)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든.\
 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words="아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"

stop_words=stop_words.split(' ')
word_tokens=word_tokenize(example)

result=[word for word in word_tokens if not word in stop_words]

print(word_tokens)
print(result)
#%% 9
#정규 표현식
# . : 임의의 문자
import re
r=re.compile("a.c")
r.search("kkk")
r.search("abc")

# ?: 앞의 문자가 존재할 수도 있고 존재 안할수도 있음
r=re.compile("ab?c")
r.search("abbc")
r.search("abc")
r.search("ac")

# *: 앞의문자가 0개 또는 여러개 올경우
r=re.compile("ab*c")
r.search("a")
r.search("ac")
r.search("abc")
r.search("abbbbbc")

# +: *와 비슷한데 앞의문자가 최소 1개이상
r=re.compile("ab+c")
r.search("ac")
r.search("abc")
r.search("abbbbc")

# ^:시작되는 글자 지정
r=re.compile("^a")
r.search("bbc")
r.search("ab")

#{}: 해당문자를 숫자만큼 반복
r=re.compile("ab{2}c")
r.search("ac")
r.search("abc")
r.search("abbc")
r.search("abbbc")

# {}:숫자 2개 붙이면 이상 이하
r=re.compile("ab{2,8}c")
r.search("ac")
r.search("abbc")
r.search("abbbbbbc")


#{숫자,}
r=re.compile("a{2,}bc")
r.search("bc")
r.search("aabc")
r.search("aaaaabc")

# []: 문자들 중에 한개 매치
r=re.compile("[abc]")#a-c와 같음
r.search("zzz")
r.search("a")
r.search("aaaaaa")
r.search("baac")

r=re.compile("[a-z]")
r.search("AAA")
r.search("aBC")

#[^문자] : 뒤에 붙은 문자들을 제외한 모든 문자 매치
r=re.compile("[^abc]")
r.search("a")
r.search("ab")
r.search("d")
r.search("1b")
        
# match와 search의 차이
r=re.compile("ab.")
r.search("kkkkkabc")

r.match("kkkabc")
r.match("abckk")

# split
text="사과 딸기 수박 메론 바나나"
re.split(" ",text)

text="사과+딸기+수박+메론+바나나"
re.split("\+",text)

# findall()
text="이름: 김철수\
 전화번호: 010-1234-1234\
 나이: 30\
 성별: 남"""
re.findall("\d+",text)
        