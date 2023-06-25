import sys

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xml.etree.cElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os

def gettext(xmltext) -> str:
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    root = ET.fromstring(xmltext)

    string = ""
    for child in root:
        if(child.tag=='title'):
            string += "" + (child.text)
        elif(child.tag=='text'):
            for p in child:
                string += " " + p.text
    return string
            
def tokenize(text) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]  # ignore a, an, to, at, be, ...
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return tokens


def stemwords(words) -> list:
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    new_list = []
    ps = PorterStemmer()
    for w in words:
        rootWord=ps.stem(w)
        new_list.append(rootWord)
    return new_list


def tokenizer(text) -> list:
    return stemwords(tokenize(text))


def compute_tfidf(corpus:dict) -> TfidfVectorizer:
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. Meaning,
    call fit() on the list of document strings, which figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function. The corpus argument is a dictionary
    mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                        analyzer='word',
                        preprocessor=gettext,
                        tokenizer=tokenizer,
                        stop_words='english', # even more stop words
                        decode_error = 'ignore')
    return tfidf.fit(corpus.values())
    
    

def summarize(tfidf:TfidfVectorizer, text:str, n:int):
    
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.09. Sort the (word,score) pairs by TFIDF score in reverse order.
    """
    words = tfidf.get_feature_names_out()
    score = tfidf.transform([text])
    nonzero = score.nonzero()
    lst = []
    for index in nonzero[1]:
        if(score[0, index] > 0.09):
            lst.append((words[index], score[0, index]))
    lst.sort(key=lambda x: x[1], reverse = True)
    lst = lst[:n]
    return lst
    


def load_corpus(zipfilename:str) -> dict:
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (filename,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    my_dict = dict()
    with zipfile.ZipFile(zipfilename) as zp:
        for f in zp.namelist():
            if(f.endswith('.xml')):
                txt = zp.read(f)
                filename = os.path.basename(f)
                my_dict[filename] = txt
    return my_dict
