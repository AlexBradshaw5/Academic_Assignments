from tfidf import *
import sys

f = open(sys.argv[1])
txt = f.read()
stems = stemwords(tokenize(gettext(txt)))
tf = Counter(stems)
print(tf.most_common(10))
