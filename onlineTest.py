import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import csv

# train = list(csv.reader(open("D:/R/DBS/products1.csv")))
train = pd.read_csv("D:/R/DBS/products.csv", header=0)
print train.columns.values
print train['_id'][0]
# print train['_id'][1]
train_cat = list(train['cat'])
# print train_cat[5]
# print train_cat[5][2]

print train['cat']
ids = []
# create a bag of model
for id in range(len(train['cat'])):
    if train['cat'][id][2]=='u':
        ids.append(id)

# print ids
train_name_descp = []
category = []
for i in range(len(train['_id'])):
    if i in ids:
        continue
    train_name_descp.append(train['name'][i]+' '+train['description'][i])
    # +' '+train['description'][i]
    category.append(train['cat'][i])
    # train_name_descp.append(train['description'][i])
print "category="+train['cat'][0]
# print  train_name_descp[0]
# print  train_name_descp[1]

def review_to_wordlist(review):

    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words

train_name_descp_clean = []
for i in range(len(train_name_descp)):
    train_name_descp_clean.append(" ".join(review_to_wordlist(train_name_descp[i])))

# print train_name_descp_clean[0]
vectorizer = CountVectorizer(analyzer="word",\
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=386)
# fit data transform
train_data_features = vectorizer.fit_transform(train_name_descp_clean)
train_data_features = train_data_features.toarray()
# print train_data_features

# train a random forest model
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features, category)

# print category
# print forest

test_name_descp = []
for i in ids:
    test_name_descp.append(train['name'][i]+' '+train['description'][i])
    # +' '+train['description'][i]

test_name_descp_clean = []
for i in range(len(test_name_descp)):
    test_name_descp_clean.append(" ".join(review_to_wordlist(test_name_descp[i])))

test_data_features = vectorizer.fit_transform(test_name_descp_clean)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
# output = pd.DataFrame( data={"id":ids["id"], "sentiment":result} )
print result
counter=0
for id in ids:
    train['cat'][id] = result[counter]
    counter+=1
print "updated cetegory"
print train['cat']

