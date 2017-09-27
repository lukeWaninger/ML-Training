import numpy as np

# transform words into feature vectors
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs  = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

# assessing word relevancy (tf-idf)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())