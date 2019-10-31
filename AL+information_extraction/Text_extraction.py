import re 
import string 
import nltk 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from nltk.stem import WordNetLemmatizer
import itertools
from tqdm import tqdm 
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy
from nltk.tokenize import word_tokenize 
from spacy import displacy
from country_list import *

!pip install spacy && python -m spacy download en
!python spacy_ner_custom_entities.py 
from nltk.corpus import stopwords
lemmatizering = WordNetLemmatizer()

nlp = spacy.load("en")

def sex(text):
    doc= nlp(text)
    female_list=['female','Female','Girl','girl','Woman','woman','her','Her','She','she']
    for i in doc :
        if i.text in ['male','Male','boy','Boy','man','his','His','He','he'] and i.pos_ in ['NOUN','ADJ','PRON']:
            return 'Male'
        if i.text in female_list and i.pos_ in ['NOUN','ADJ','PRON'] :
            return 'Female'

        
def age(text):
    text=text.replace('.','')
    text=text.replace('�','')
    list1=[]
    list2=[]
    
    for i in range(100000):
        list2.append(str(i))
        
   
    for words in word_tokenize(text):
        for i in words:
            if i in [':']:
                words=words.replace(words,'')
        list1.append(words)
        list1.append(' ')
    text_time=' '.join([lemmatizering.lemmatize(w) for w in list1 ])
 
    doc= nlp(text_time)
    for i in range(len(doc)):
        if doc[i].pos_=='NUM':
            if str(doc[i+2].text) not in ['am','pm','miles','times']:
                if str(doc[i].text) in list2:
                    if int(doc[i].text)<=120:
                        return doc[i].text
        
        
                else:
                    return doc[i].text
        if doc[i].text in ['teen','Teen','infant','Infant','senior','Senior']:
            return doc[i].text
        
      
        
def name(text):
    text=text.replace('�','')
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
    if 'PERSON' in list1:
        
        return [list1[i-1] for i in range(len(list1)) if list1[i]=='PERSON' if list1[i-1] not in countries]
        
        #for e in range(len(list1)):
           # if list1[e]=='PERSON':
                #if list1[e-1] not in countries:
                   # return list1[e-1]
    #else:
      #  return None
def town_adress(text):
    text=text.replace('�','')
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
        
    for e in range(len(list1)):
            return [list1[e-1] for e in range(len(list1)) if list1[e] in ['GPE','LOC','FAC']]
            if list1[e] in countries:
                return list1[e]+[list1[i-1] for i in range(len(list1)) if list1[i] in ['GPE','LOC','FAC']]
            
    else:
        return None
def date(text):
    text=text.replace('.','')
    text=text.replace('�','')
    countries = list(dict(countries_for_language('en')).values())
    doc=nlp(text)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    list1=[]
    for i in entities:
        for e in i:
            list1.append(str(e))
    
    return [list1[i-1] for i in range(len(list1)) if list1[i] in ['DATE','TIME']]
           
                
def main(data):
    cleaned_data=pd.DataFrame()
    cleaned_data['Narrative']= data['Narrative']
    cleaned_data['sex']= data['Narrative'].apply(sex)
    cleaned_data['age']=data['Narrative'].apply(age)
    cleaned_data['name']=data['Narrative'].apply(name)
    cleaned_data['town']=data['Narrative'].apply(town_adress)
    cleaned_data['date']=data['Narrative'].apply(date)
    return cleaned_data
