import pandas as pd
data= pd.read_csv('Tweets.csv')
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import string
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from category_encoders import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical

def clean_text(row):
    
    
    row= re.sub('@[^\s]+','',row)
    
    row = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',row)
    
    row= re.sub(r'\d+(\.\d+)?', '',row)
    
    row = re.sub(r'£|\$', '',row)
    
    row=row.replace('#','')
    
    row=row.replace('?','')
    
    row=row.replace('&','')
    
    row=row.replace('!','')
    
    row=row.replace('.','')
    
    row=row.replace(',','')
    
    row=row.replace(';','')
    
    row=row.replace('...','')
    
    row=row.replace('"','')
    
    row=row.replace("'",' ')
    
    row=row.replace("+",'')
    
    row=row.replace("-",'')
    
    row=row.replace(":",'')

    
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  
                u"\U0001F300-\U0001F5FF"  
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF"  
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    row=emoji_pattern.sub(r'', row) 
    
    
    return row.lower()



def lemmantizing(row):
    
    lemmatizering = WordNetLemmatizer()

    stop=set(stopwords.words('english')) 
    
    tokenize=word_tokenize(clean_text(row))
    
    stop=[words for words in tokenize if words not in stop]
    
    
    
    return [lemmatizering.lemmatize(w) for w in stop ]

def length(row):
    
    try:
        return row['text'].apply(lambda x: len(x)-x.count(' '))
    except:
        return('error')
    
def count_punct(row):
    count = sum([1 for char in row if char in string.punctuation])
    return round(count/(len(row) - row.count(" ")), 3)*100

def clean(row):
    ps = PorterStemmer()
    stopword=set(stopwords.words('english'))
    text = "".join([word.lower() for word in row if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopword])
    return text

def transform(row):
    
    clean_df=pd.DataFrame()
    
    clean_df['name']=row['name']
    
    
    
    clean_df[['airline_sentiment_confidence','negativereason','negativereason_confidence',
              'airline','retweet_count']]=row[['airline_sentiment_confidence','negativereason','negativereason_confidence',
                                'airline','retweet_count']]
    
    clean_df['text']=row['text'].apply(clean_text)
    
    clean_df['cleaned_text']=row['text'].apply(lemmantizing)

    clean_df['punctuation']=row['text'].apply(count_punct)
    
    clean_df['length']=length(row)
    
    clean_df['label']=row['airline_sentiment']

    
    
   
    
    return clean_df


def X_items(row):
    
    tfidf_vect = TfidfVectorizer(analyzer=clean)
    
    X_tfidf = tfidf_vect.fit_transform(transform(row)['cleaned_text'])
    
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    
    X_tfidf_df.columns = tfidf_vect.get_feature_names()
    
    columns=list(set(transform(row).columns)-set(['text','cleaned_text']))
    
    return pd.concat([transform(row)[columns],X_tfidf_df], axis=1)


def fill_missing(row):
        
    mean_columns=['negativereason_confidence','airline_sentiment_confidence']
    
    other_columns= list(set(row.columns)-set(mean_columns))
    
    mean_impute=SimpleImputer(missing_values=pd.np.nan, strategy='mean')
    
    mean_imputing=pd.DataFrame(mean_impute.fit_transform(row[mean_columns]))
    
    mean_imputing.columns=mean_columns
    
    return pd.concat([mean_imputing,row[other_columns]],axis=1)


def encode(row):
    
    labelencoder = LabelEncoder()
    
    label_col=labelencoder.fit_transform(row['label'])
    
    label_col=pd.DataFrame(label_col)
    
    label_col.columns=['label']
    
    other_columns=list(set(row.columns)-set(['label','name']))
    
    columns =['airline','negativereason']
    
    enc=TargetEncoder(cols=columns, min_samples_leaf=20,smoothing=1.0).fit(row[other_columns],label_col)
    
    encoded_train=pd.DataFrame(enc.transform(row[other_columns],label_col))
    
    return pd.concat([row['name'],encoded_train,row['label']],axis=1)

def normalize(row):
    
    min_max= MinMaxScaler()
    
    training_cols=list(set(row.columns)-set(['label','name']))
    
    normalized_df=pd.DataFrame(min_max.fit_transform(row[training_cols],row['label']))
    
    normalized_df.columns=training_cols
    
    normalized_label=pd.concat([normalized_df,row[['label','name']]],axis=1)
    
    # Dummies function will encode the label data of three classes into three binary columns so it can be 
    #used in our neural network
    normalized_label_df= pd.get_dummies(normalized_label, prefix_sep="__",
                              columns=['label'])
    
    return shuffle(normalized_label_df)

def model_pred(row):
    
    prediction=pd.DataFrame() 
    
    x_features=list(set(row.columns)-set(['name','label__negative',
       'label__neutral', 'label__positive']))
    
    X_train, X_test, y_train, y_test = train_test_split(row[x_features], row[['label__negative',
       'label__neutral', 'label__positive']], test_size=0.2, random_state=42)
    
    
    prediction[['label__negative',
       'label__neutral', 'label__positive']]= y_test
    model= Sequential()
    
    model.add(Dense(23, activation='relu', input_dim =43))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(23, activation='relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
              

    sgd= SGD(lr=0.01 , decay=1e-6, momentum= 0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    
    

    model.fit(X_train, y_train, epochs=2000, batch_size=128)
    
    
    prediction[['negative','neutural','positive']]= pd.DataFrame(model.predict_proba(X_test))
    
    return prediction
    
    
def main(row):
     
    imputed= fill_missing(X_items(row))
    
    encoded= encode(imputed)
    
    normalized= normalize(encoded)
    
    training=model_pred(normalized)
    
    return training
    

    
    
    
    
    
    
