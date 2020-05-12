import os
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import pickle
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import csv

def count_capital(message):
    count = 0
    for i in message:
         if i.isupper():
             count += 1
    return count/(len(message)-message.count(" "))*100

def count_punct(message):
    punctuations = string.punctuation
    count = sum([1 for word in message if word in punctuations])
    return count/(len(message)-message.count(" "))*100

def count_stopwords(message):
    stopwords= nltk.corpus.stopwords.words('english')
    count = sum([1 for word in message if word not in stopwords])
    return count/(len(message)-message.count(" "))*100

def punctuation(message):
    punctuations = string.punctuation
    sms = [word for word in message if word not in punctuations]
    sms = "".join(sms).split()
    return sms

def stopwords(message):
    stopwords= nltk.corpus.stopwords.words('english')
    sms = [word for word in message if word not in stopwords]
    return sms

def Lemmatization(message):
    sms = [WordNetLemmatizer().lemmatize(word) for word in message]
    return sms


def tf_idf(message):
    tfidfVectorizer = TfidfVectorizer()
    term_matrix = tfidfVectorizer.fit_transform(message)
    pd.set_option('display.max_columns', None)
    features = pd.DataFrame(term_matrix.toarray(),columns=tfidfVectorizer.get_feature_names())
    return features, tfidfVectorizer

def normalization(message):
    min_max_scaler = preprocessing.MinMaxScaler()
    sms = min_max_scaler.fit_transform(message.values.reshape(-1,1))
    return sms, min_max_scaler

def to_string(list):
    string = ''
    for i in list:
        string += i + ' '
    return string

def randomforest(X_train,y_train):
    rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    return rf_model

if __name__ == '__main__':
    wd = os.getcwd()
    file = os.path.join(wd, 'spam.csv')
    df = pd.read_csv(file,encoding='latin-1')
    df.drop(df.columns[[2,3,4]],axis=1, inplace=True)
    df.columns = ['label','content']

    '''
    df['label'].value_counts())
    ham 4825
    spam 747
    inbalanced data
    '''
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']

    ham = ham.sample(spam.shape[0]) # take same size with spam from ham
    message = ham.append(spam, ignore_index=True) # create new balance dataset with sampled ham and whole spam

    # tokenize
    message['remove_punctuation'] = message['content'].apply(punctuation)
    message['remove_stopwords'] = message['remove_punctuation'].apply(stopwords)
    message['lemmatization'] = message['remove_stopwords'].apply(Lemmatization)
    message['to_string'] = message['lemmatization'].apply(to_string)

    # check the difference of length in spam/ham
    message['length'] = message['content'].apply(len) # create column  for content length
    '''
    plt.title("sms length in spam/ham")
    plt.hist(message[message['label']=='ham']['length'],bins = 100)
    plt.hist(message[message['label']=='spam']['length'], bins = 100)
    plt.legend(['ham','spam'])
    plt.show()
    most spam are longer than ham
     '''
    # check the difference of punctuation rate in spam/ham
    '''
    message['punct%'] = message['content'].apply(count_punct)
    plt.title("percentage of punctuation in spam/ham")
    plt.hist(message[message['label']=='ham']['punct%'],bins = 250)
    plt.hist(message[message['label']=='spam']['punct%'],bins = 250)
    plt.legend(['ham','spam'])
    plt.show()
    # distribution of punctuation rate in spam/ham are similar
    '''

    # check the difference of stopwords rate in spam/ham
    message['stopwords%'] = message['content'].apply(count_stopwords)
    '''
    plt.title("percentage of stopwords in spam/ham")
    plt.hist(message[message['label']=='ham']['stopwords%'],bins = 100)
    plt.hist(message[message['label']=='spam']['stopwords%'],bins = 100)
    plt.legend(['ham','spam'])
    plt.show()
    # spam has more stopwords
    '''

    # check the difference of capital words rate in spam/ham
    message['capital%'] = message['content'].apply(count_capital)
    '''
    plt.title("percentage of capital words rate in spam/ham")
    plt.hist(message[message['label'] == 'ham']['capital%'], bins=100)
    plt.hist(message[message['label'] == 'spam']['capital%'], bins=100)
    plt.legend(['ham', 'spam'])
    plt.show()
    # spam tend to have more capital words
    '''

    # split to training and testing
    content_train, content_test, label_train, label_test = train_test_split(message,message['label'],test_size=0.33,random_state=0)

    # create tfidf features
    features,tfidf = tf_idf(content_train['to_string'])
    with open('tfidf.pkl', 'wb') as fw:
        pickle.dump(tfidf, fw)

    # consider sms length as an influence feature
    leng, length_nor = normalization(content_train['length'])
    features.insert(0, 'leng', leng)
    with open('length.pkl', 'wb') as fw:
        pickle.dump(length_nor, fw)

    # consider sms stopwords% as an influence feature
    stopword, stopwords_nor = normalization(content_train['stopwords%'])
    features.insert(0, 'stopwords%', stopword)
    with open('stopwords.pkl', 'wb') as fw:
        pickle.dump(stopwords_nor, fw)

    # consider sms capital% as an influence feature
    capital, capital_nor = normalization(content_train['capital%'])
    features.insert(0, 'capital%',capital)
    with open('capitalwords.pkl','wb') as fw:
        pickle.dump(capital_nor, fw)

    # build randomforest model
    model = randomforest(features, label_train)
    with open('model.pkl', 'wb') as fw:
        pickle.dump(model, fw)

    '''

    # create tfidf features
    features,tfidf = tf_idf(message['to_string'])
    with open('tfidf.pkl', 'wb') as fw:
        pickle.dump(tfidf, fw)
    # print(tfidf.transform([x]).toarray())

    # consider sms length as an influence feature
    length, length_nor = normalization(message['length'])
    features.insert(0, 'length', length)
    with open('length.pkl', 'wb') as fw:
        pickle.dump(length_nor, fw)

    # consider sms stopwords% as an influence feature
    stopwords, stopwords_nor = normalization(message['stopwords%'])
    features.insert(0, 'stopwords%', stopwords)
    with open('stopwords.pkl', 'wb') as fw:
        pickle.dump(stopwords_nor, fw)

    model = randomforest(features, message['label'])
    with open('model.pkl', 'wb') as fw:
        pickle.dump(model, fw)
    '''
    feature_length = length_nor.transform(content_test['length'].values.reshape(-1,1))
    feature_stopwords = stopwords_nor.transform(content_test['stopwords%'].values.reshape(-1,1))
    feature_capital = capital_nor.transform(content_test['capital%'].values.reshape(-1,1))
    feature_tfidf = tfidf.transform(content_test['to_string']).toarray()
    vect = pd.DataFrame(feature_tfidf)
    vect.insert(0, 'length', feature_length)
    vect.insert(0, 'stopwords%', feature_stopwords)
    vect.insert(0, 'capital%', feature_capital)

    my_prediction = model.predict(vect)
    accuracy = sum(my_prediction==label_test)/len(label_test)
    print('accuracy:',accuracy)
    f1= f1_score(label_test, my_prediction,average='weighted')
    print('f1 score:',f1)
    recall = recall_score(label_test, my_prediction,average='weighted')
    print('recall:', recall)
    precision = precision_score(label_test, my_prediction,average='weighted')
    print('precision:', precision)