import pandas   as pd
import numpy    as np
import pyprind
import os

# parse through the aclImdb documents appending each comment to a data frame
pbar   = pyprind.ProgBar(50000)
labels = { 'pos':1, 'neg':0 }
df     = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos','neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding = 'utf8') as infile:
                txt = infile.read()
                df  = df.append([[txt, labels[l]]], ignore_index = True)
                pbar.update()

df.columns = [ 'review', 'sentiment' ]

# shuffle and store to new csv file
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index = False)

# clean text data to remove html markup
import re
def preprocessor(text):
    text      = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text      = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

# setup a tokenizer to split at whitespace
def tokenizer(text):
    return text.split()

# getup with a porter stemmer tokenizer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# hit that data frame with the preprocessor (pull out html markup)
df['review'] = df['review'].apply(preprocessor)

# split the document int training/test
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test  = df.loc[25000:, 'review'].values
y_test  = df.loc[25000:, 'sentiment'].values

# use a GridSearchCV to find optimal set of parameters for
# logistic regression using 5-fold stratified cross-validation
from sklearn.grid_search    import GridSearchCV
from sklearn.pipeline       import Pipeline
from sklearn.linear_model   import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus            import stopwords
import nltk

nltk.download('stopwords')
stop = stopwords.words('english')

tfidf = TfidfVectorizer(strip_accents = None,
                        lowercase     = False,
                        preprocessor  = None)
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words' : [stop, None],
               'vect__tokenizer'  : [tokenizer, tokenizer_porter],
               'clf_penalty'      : ['l1', 'l2'],
               'clf__C'           : [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words' : [stop, None],
               'vect__tokenizer'  : [tokenizer, tokenizer_porter],
               'vect__use_idf'    : [False],
               'vect__smooth_idf'  : [False],
               'vect__norm'       : [None],
               'clf__penalty'     : ['l1', 'l2'],
               'clf__C'           : [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf' , LogisticRegression(random_state = 0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, 
                           param_grid,
                           scoring = 'accuracy',
                           cv      = 5,
                           verbose = 1,
                           n_jobs  = -1)
gs_lr_tfidf.fit(X_train, y_train)

# print the best parameter set
print('Best parameter set: %s ' 
      % gs_lr_tfidf.best_params_)
# best is shown to be: not using stopwords, the whitespace tokenizer,
# n-gram 1, logistic regression with l2 penalty where c=10.0

# print cv accuracy with best params
print('CV Accuracy: %.3f'
      % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'
      % clf.score(X_test, y_test))
# this model can therefor predict whether or not the review 
# is positive or negative with ~90% accuracy