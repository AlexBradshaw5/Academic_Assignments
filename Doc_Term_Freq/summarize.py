from tfidf import *

zipfilename = sys.argv[1]
summarizefile = sys.argv[2]

dictt = load_corpus(zipfilename)
txt = dictt[summarizefile]
print(summarize(compute_tfidf(dictt), txt, 20))