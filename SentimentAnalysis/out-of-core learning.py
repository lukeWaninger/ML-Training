import numpy as np
import re
from nltk.corpus import stopwords

# separate the text and remove html mark simultaneously returning
# the tokenized text
stop = stopwords.words('english')
def tokenizer(text):
    text      = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text      = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# define a generator function that streams and returns one doc at a time
def stream_docs(path):
    with open(path, 'r', encoding='ISO-8859-1') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# define a function to process a specific number of documents in 'batches'
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# the document set is too large to store the vocabulary in memory
# so we can use a hash function to map words to their counts
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error = 'ignore',
                         n_features   = 2**21,
                         preprocessor = None,
                         tokenizer    = tokenizer)
clf = SGDClassifier(loss = 'log',
                    random_state = 1,
                    n_iter = 1)
doc_stream = stream_docs(path = './movie_data.csv')

# now start the out-of-core learning stream
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size = 1000)

    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes = classes)
    pbar.update()

# evaluate performance with remaining 5000 docs
X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' 
      % clf.score(X_test, y_test))
