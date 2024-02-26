#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd


# In[ ]:





# In[12]:


import pandas as pd

# List of possible encodings to try
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

file_path = 'spam.csv'  # change this to the path of your CSV file

# Attempt to read the CSV file with different encodings
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)  # Fix the typo here
        print(f"File successfully read with encoding: {encoding}")  # Fix the f-string here
        break  # Stop the loop if successful
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")  # Fix the f-string here
        continue  # Try the next encoding

# If the loop completes without success, df will not be defined
if 'df' in locals():
    print("CSV file has been successfully loaded.")
else:
    print("All encoding attempts failed. Unable to read the CSV file.")


# In[14]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[16]:


df.sample(5)


# In[18]:


df.shape


# In[20]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# In[22]:


df.info()


# In[24]:


#drop last 3 cols
df.drop(columns=['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4'], inplace=True)


# In[26]:


df.sample(5)


# In[28]:


#renaming the cols
df.rename(columns={'v1' : 'target' , 'v2' : 'text'}, inplace=True)
df.sample(5)


# In[30]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])


# In[32]:


df.head()


# In[34]:


#missinng values
df.isnull().sum()


# In[36]:


df= df.drop_duplicates(keep='first')
df.duplicated().sum()


# In[38]:


df.shape


# In[40]:


df.head()


# In[42]:


df['target'].value_counts()


# In[44]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()


# In[45]:


#Big Chunk of ham and very less spam so out data is not balanced


# In[48]:


import nltk


# In[49]:


get_ipython().system('pip install nltk')


# In[51]:


nltk.download('punkt')


# In[52]:


df['num_characters'] = df['text'].apply(len)# numbers of char


# In[53]:


df.head()


# In[54]:


#no.of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x))) #words count


# In[55]:


df.head()


# In[60]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))#sentence


# In[63]:


df.head()


# In[66]:


print(df.columns)


# In[68]:


df[['num_characters', 'num_words', 'num_sentences']].describe()


# In[70]:


#targetting ham
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()


# In[72]:


#targetting spam
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()


# In[74]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[76]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[78]:


sns.pairplot(df,hue='target')


# In[79]:


numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True)


# In[82]:


#Data Preprocessing


# In[84]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords') #You may need to download the stopwords dataset

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        return " ".join(y)
transformed_text = transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

print(transformed_text)
        


# In[86]:


df['text'][10]


# In[87]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[88]:


df['transformed_text']=df['text'].apply(transform_text)


# In[89]:


df.head()


# In[92]:


pip install wordcloud


# In[93]:


from wordcloud import WordCloud

# Instantiate WordCloud object
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[94]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[95]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[96]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[97]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[98]:


df.head()


# In[99]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[100]:


len(spam_corpus)


# In[139]:


from collections import Counter
import pandas as pd
# Assuming spam_corpus is your list of words
counter_data = Counter(spam_corpus).most_common(30)
df_counter = pd.DataFrame(counter_data, columns=['Word', 'Frequency'])

# Plotting
sns.barplot(x='Word', y='Frequency', data=df_counter)
plt.xticks(rotation='vertical')
plt.show()


# In[141]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    if isinstance(msg, str):  # Check if msg is a string
        for word in msg.split():
            ham_corpus.append(word)


# In[143]:


len(ham_corpus)


# In[145]:


from collections import Counter

# Assuming ham_corpus is your list of words
counter_data = Counter(ham_corpus).most_common(30)
df_counter = pd.DataFrame(counter_data, columns=['Word', 'Frequency'])

# Plotting
sns.barplot(x='Word', y='Frequency', data=df_counter)
plt.xticks(rotation='vertical')
plt.show()


# In[147]:


#Text Vectorization
#using Bag of Words
df.head()


# In[149]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)


# In[151]:


# Filter out None values and convert to empty strings
df['transformed_text'] = df['transformed_text'].apply(lambda x: '' if x is None else x)

# Now you can proceed with vectorization
X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[152]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[155]:


# appending The num_character col to X
#X = scaler.fit_transform(X)


# In[157]:


# appending the num_characters col to X
#X = np.hstack((X, df['num_characters'].values.reshape(-1,1)))


# In[159]:


X.shape


# In[161]:


y = df['target'].values


# In[163]:


from sklearn.model_selection import train_test_split


# In[165]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[167]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[169]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_test))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[171]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_test))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[173]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_test))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# In[175]:


#tfid  ---> MNb


# In[177]:


get_ipython().system('pip install xgboost')


# In[178]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[181]:


svc = SVC(kernel = 'sigmod', gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver = 'liblinear', penalty ='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50,random_state=2)
bc = BaggingClassifier(n_estimators=50,random_state=2)
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[217]:


clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBCT' : gbdt,
    'XGB' : xgb
}


# In[219]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score ,precision_score


# In[221]:


def train_classifier(clf,X_train,y_train,X_test, y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)

    return accuracy,precision


# In[223]:


# Assuming X_train, X_test, y_train, y_test are defined
svc = SVC(kernel='sigmoid')  # Correct the typo here
accuracy = train_classifier(svc, X_train, y_train, X_test, y_test)
print("Accuracy:", accuracy)


# In[225]:


# Store classifiers in a dictionary
 

# Now you can iterate over clfs
accuracy_scores = []
precision_scores= []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
 
    print("For", name)
    print("Accuracy:", current_accuracy)
    print("Precision:", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[242]:


print(len(clfs.keys()))
print(len(accuracy_scores))
print(len(precision_scores))


# In[238]:


performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Precision',ascending=False)


# In[240]:


performance_df


# In[244]:


performance_df1 = pd.melt(performance_df , id_vars = "Algorithm")


# In[246]:


performance_df1


# In[248]:


sns.catplot(x = 'Algorithm', y='value' ,
            hue = 'variable' ,data = performance_df1, kind = 'bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation = 'vertical')
plt.show()


# In[250]:


#model improve
#1. change the max_features parameter of TfIdf


# In[254]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[256]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[258]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[260]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores, 'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending =False)


# In[262]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[266]:


#Voting Classifier
svc = SVC(kernel = 'sigmoid' , gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
from sklearn.ensemble import VotingClassifier


# In[268]:


voting = VotingClassifier(estimators=[('svm' ,svc), ('nb' , mnb), ('et' ,etc)],voting ='soft')


# In[272]:


voting.fit(X_train,y_train)


# In[275]:


y_pred = voting.predict(X_test)
print("Accuracy", accuracy_score(y_test,y_pred))
print("Precision", precision_score(y_test,y_pred))


# In[277]:


#Applying stacking
estimators = [('svm', svc), ('nb' ,mnb), ('et', etc)]
final_estimator = RandomForestClassifier()


# In[281]:


from sklearn.ensemble import StackingClassifier


# In[285]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[287]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[295]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#Sample text data and correspoding Lables(replace with your actual data)
X_train= ["Sample text 1", "Sample text 2", "Sample text 3"]
y_train = [0,1,0] #Example labels ( 0 for negative, 1 for positive )

#Create and train the TF- IDF vectorizer 
tfidf = TfidfVectorizer(lowercase = True, stop_words = 'english')
X_train_tfidf = tfidf.fit_transform(X_train)
#Create and train the Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)

#Save the trained TF-IDF vecorizer and Naive Bayes model to files
with open ('vectorizer.pkl' , 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
with open('model.pkl', 'wb') as model_file:
    pickle.dump(mnb, model_file)


# In[ ]:




