#!/usr/bin/env python
# coding: utf-8

# In[635]:


import pandas as pd
data=pd.read_csv('/home/ftehrani/Desktop/SMSSpamCollection.csv', sep='\t', header=None)
data=data.rename(columns={0:'category',1:'title'})
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def encode(row):
    encoding=LabelEncoder()
    try:
        return pd.DataFrame(encoding.fit_transform(row['category'])).rename(columns={0:'category'})
    except:
        return('error')
    
def regular(row):
    
    row=row.replace('..','')
    
    row=row.replace('.','')
    
    row=row.replace('!','')
    
    row=row.replace("'s",'')
    
    row=row.replace('(','')
    
    row=row.replace(')','')
    
    row=row.replace('-','')
    
    row=row.replace('#','')
    
    row=row.replace('&','')
    
    row = row.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
    row = row.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    row = row.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    row = row.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
    row = row.replace(r'\d+(\.\d+)?', 'numbr')
    
    
    return row

def stop_words(row):
    stop_words = stopwords.words('english')
    word_tokens = word_tokenize(row.lower()) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    
    
    return filtered_sentence

def untokenize(row):
    try:
        return TreebankWordDetokenizer().detokenize(row)
    
    except:
        return ('error')
def clean_text(row):
    text = "".join([word.lower() for word in row if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopword])
    return text

def length(row):
    try:
        return row['title'].apply(lambda x: len(x)- x.count(" "))
    except:
        return ('error')

def count_punct(row):
    count = sum([1 for char in row if char in string.punctuation])
    return round(count/(len(row) - row.count(" ")), 3)*100    
    
def transform(row):
    cleaned_data=pd.DataFrame()
    
    cleaned_data['title']=row['title']
    
    cleaned_data['length_string']=length(cleaned_data)
    
    cleaned_data['title']= cleaned_data['title'].apply(regular)
    cleaned_data['title']=cleaned_data['title'].apply(stop_words)
    cleaned_data['title']=cleaned_data['title'].apply(untokenize)
    cleaned_data['clean_text']=cleaned_data['title'].apply(clean_text)
    cleaned_data['puctuation']= cleaned_data['title'].apply(count_punct)
    cleaned_data['category']=encode(row) 
    
    return cleaned_data

def X_items(row):
    tfidf_vect = TfidfVectorizer(analyzer=clean_text)
    X_tfidf = tfidf_vect.fit_transform(transform(row)['clean_text'])
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = tfidf_vect.get_feature_names()
    return pd.concat([transform(row)[['length_string','puctuation']],X_tfidf_df], axis=1).drop(columns= ' ')

# Logitic Regression Classification 
def predicting(row):
    final=pd.DataFrame()
    X_train, X_test, y_train, y_test = train_test_split(X_items(data),transform(data)['category'], test_size=0.2)
    final[['data','label']]= transform(row)[['title','category']]
    lin=LogisticRegression()
    lin.fit(X_train,y_train)
    final['pred']=pd.DataFrame(lin.predict(X_test))
   
    return final

# Random Forest Classification

def random_forest(row):
    final=pd.DataFrame()

    rf = RandomForestClassifier(n_jobs=-1)

    k_fold = KFold(n_splits=5)
    final[['title','label']]=transform(row)[['title','category']]

    

    X_train, X_test, y_train, y_test = train_test_split(X_items(data),transform(data)['category'], test_size=0.2)

    rf = RandomForestClassifier(n_estimators=50, max_depth=25, n_jobs=-1)

    rf.fit(X_train, y_train)
    final['pred']=pd.DataFrame(rf.predict(X_test))
    final['label1']=y_test

    print(confusion_matrix(y_test,rf.predict(X_test)))
    return final

