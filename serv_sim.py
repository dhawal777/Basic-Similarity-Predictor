from nltk import *
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
import re, math
import collections
from flask import Flask, request, redirect

#def get_cosine(vec1, vec2):Calculating cosine Similarity
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     print (vec1)
     print (vec2)
     print (intersection)
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
#def text_to_vector(text):text to vector conversion
def text_to_vector(text):
     WORD = re.compile(r'\w+')
     #print (WORD)
     words = WORD.findall(text)
     return Counter(words)
#similar_text(text1, text2):Calculating the similarity following the flow as described:
#Tokenizing---->Removing Stopwords---->POS(NN,JJ,VB,CD)---->Lemmatizing---->Calculating sementic similarity----->calculating cosine similarity(textual)----->averaging and getting final    
def similar_text(text1, text2):
    max_sim = 0.0
    

    words1=word_tokenize(text1)
    words2=word_tokenize(text2)
    lmtzr = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    #print (stop_words)
    words1=set(words1).difference(stop_words)   #Removing stop words
    words2=set(words2).difference(stop_words)
    #part of speech wordnet deal with noun,adjective,verb,adverb
    nouns1 = [token for token, pos in pos_tag(words1) if pos.startswith('NN')]
    nouns2 = [token for token, pos in pos_tag(words2) if pos.startswith('NN')]
    verbs1 = [token for token, pos in pos_tag(words1) if pos.startswith('VB')]
    verbs2 = [token for token, pos in pos_tag(words2) if pos.startswith('VB')]
    adjectives1 = [token for token, pos in pos_tag(words1) if pos.startswith('JJ')]
    adjectives2 = [token for token, pos in pos_tag(words2) if pos.startswith('JJ')]
    adverbs1 = [token for token, pos in pos_tag(words1) if pos.startswith('RB')]
    adverbs2 = [token for token, pos in pos_tag(words2) if pos.startswith('RB')]
    cardinals1 = [token for token, pos in pos_tag(words1) if pos.startswith('CD')]
    cardinals2 = [token for token, pos in pos_tag(words2) if pos.startswith('CD')]
    print(nouns1)
    print(verbs1)
    for w in nouns1:#lemmatize noun
        nouns1.remove(w)
        nouns1.append(lmtzr.lemmatize(w,pos="n"))
    for w in nouns2:
        nouns2.remove(w)
        nouns2.append(lmtzr.lemmatize(w,pos="n"))
    for w in verbs1:#lemmatize verb
        verbs1.remove(w)
        verbs1.append(lmtzr.lemmatize(w,pos="v"))
    for w in verbs2:
        verbs2.remove(w)
        verbs2.append(lmtzr.lemmatize(w,pos="v"))
    #print(nouns1)
    #print(verbs1)

#print (nouns1 + nouns2)
#print (verbs1 + verbs2)

    sum_sim=0.0
    count_words=0
    ws_noun={}
    #best possible related synonym noun-noun followed by verb-verb
    for noun1 in nouns1:        #Word set of nouns (Similar  words)..  (Semantic Similarity)
      max_sim=0.0
      for noun2 in nouns2:
        synsets_1 = wn.synsets(noun1)
        synsets_2 = wn.synsets(noun2)
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.wup_similarity(synset_1, synset_2)
                if sim is not None:
                    if sim > max_sim:
                        max_sim = sim
      if(max_sim>0.25):           #only include word if sim is greater than 0.25 (threshold)
        ws_noun[noun1]=max_sim
        count_words+=1
        sum_sim+=max_sim
        
    ws_verb={}
    for verb1 in verbs1:        #Word set of similar verbs (Semantic Similarity)
      max_sim=0.0
      for verb2 in verbs2:
        synsets_1 = wn.synsets(verb1)
        synsets_2 = wn.synsets(verb2)
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.wup_similarity(synset_1, synset_2)
                if sim is not None:
                    if sim > max_sim:
                        max_sim = sim
      if(max_sim>0.25):
        ws_verb[verb1]=max_sim
        count_words+=1
        sum_sim+=max_sim

    #print (ws_noun)
    #print (ws_verb)
    vector1 = text_to_vector(text1)#text to vector
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)

    print(cosine)



    similarity=cosine+sum_sim/count_words
    print(similarity/2)
    return similarity/2*100

app = Flask(__name__)

# global variable to hold ouptut
op = None

@app.route('/Input', methods = ["POST"])#it allows Input method to take data from the form.
def Input():
    F1 = request.form['F1']
    F2 = request.form['F2']

    # do something with F1, F2 that effect op, e.g.
    
    
    
    op = "Similarity Percent between this two statements is: "+"{}".format(similar_text(F1, F2))
    
    # now you can either return the string type op that will get displayed
    return op

    # or you may return a redirect to home()
    # return redirect('/')

@app.route('/')
def home():
    return op
    
    
if __name__=='__main__':
    app.debug=True
    app.run()


