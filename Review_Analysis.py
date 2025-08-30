import pandas as pd
import numpy as np
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
from nltk.corpus import stopwords
import string, re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
def clean_text(msg):
    sw = stopwords.words('english')
    sw.remove('not')
    sw.remove("don't")
    sw.remove("doesn't")
    sw.remove("hasn't")
    sw.remove("haven't")
    sw.remove("wasn't")
    sw.remove("weren't")
    def rem_punc(msg):
        return re.sub(f'[{string.punctuation}]','',msg)
    def rem_stop_wrds(msg):
        words = word_tokenize(msg)
        new_words = [i for i in words if i not in sw]
        return " ".join(new_words)
    def stemming(msg):
        ps = PorterStemmer()
        word = word_tokenize(msg)
        new_words = [ps.stem(w) for w in word]
        return " ".join(new_words)
    X1 = rem_punc(msg)
    X2 = X1.lower()
    X3 = rem_stop_wrds(X2)
    X4 = stemming(X3)
    return X4
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer(binary=False,ngram_range=(1,2))

X = cv.fit_transform(df['Review']).toarray()
y = df['Liked']
clf = MultinomialNB()
clf.fit(X,y)
def predict_single():
    inp = input("Enter Review : ")
    ct = clean_text(inp)
    x_test = cv.transform([ct]).toarray()
    pred = clf.predict(x_test)
    if(pred[0]==0):
        return "Not Liked"
    else:
        return "Liked"
def predict_bulk():
    srcpath = input("Enter File Path : ")
    savepath = input("Enter Folder Path Where You Want to Save The Predicted Output : ")
    df = pd.read_csv(srcpath,names=['Review'],sep='\t')
    X = df['Review'].map(clean_text)
    X_test = cv.transform(X).toarray()
    pred = clf.predict(X_test)
    result_df = pd.DataFrame()
    result_df['Review'] = df['Review']
    result_df['Sentiment'] = pred
    result_df['Sentiment'] = result_df['Sentiment'].map({0:"Not Liked",1:"Liked"})
    result_df.to_csv(f"{savepath}/result.csv",index=False,sep=',')
    print("File Saved")
while True:
    print("Review Analysis\n")
    inp = input("Choose : \n1. Single Prediction\n2. Bulk Prediction\nAny Other Key to Exit\n\n")
    if(inp == '1'):
        prediction = predict_single()
        print(prediction)
    elif(inp == '2'):
        predict_bulk()
    else:
        break
