import re
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize


df = pd.read_csv('complaints.csv', skipfooter=1496058, engine='python') #reading only 2000 of 1498058


# Clean the complaints text / Pre-Processing

#Step 1. All cases have been converted to low
df['Issue'] = df['Issue'].str.lower()

#Step 2. Each word from each row has been tokenized
df['Issue'] = df['Issue'].apply(word_tokenize)

#Step 3. The English stop words have been removed
def stops_removal(text):
    t = [token for token in text if token not in stopwords.words("english")]
    text = ' '.join(t)
    return text
df['Issue'] = df['Issue'].apply(stops_removal)
df['Issue'] = df['Issue'].apply(word_tokenize)

#Step 4. The words have been stemmed
stemmer = SnowballStemmer("english")
df['Issue'] = df['Issue'].apply(lambda x: [stemmer.stem(y) for y in x])

#Step 5. The words have been lemmatized
lmtzr = WordNetLemmatizer()
df['Issue'] = df['Issue'].apply(lambda lz:[lmtzr.lemmatize(z) for z in lz])


#The complaints are formatted with the purpose of creating the vocabulary
complaints = []
for row in df['Issue']:
    complaints.append(row)

res = [' '.join(ele) for ele in df['Issue']] #will be used later at BoW with Sklearn
complaints = ' '.join(res)
complaints = word_tokenize(complaints)
#print(df['Issue'])
#print(res)
#print(complaints)

#The vocabulary (wordset) has been created [each clean word from complaints appears just one time in vocabulary]
vocabulary = []

for w in complaints:
    if w not in vocabulary:
        vocabulary.append(w)
#print(vocabulary)

#Creating the dictionary for Bag of words which counts how often a word appears in a complaint
def calculateBOW(vocabulary,complaint):
    tf_diz = dict.fromkeys(vocabulary,0)
    for word in complaint:
        tf_diz[word]=complaint.count(word)
    return tf_diz


#Creating the Bag of Words for all 2000 complaints.
bows = []
for r in df['Issue']:
    b = calculateBOW(vocabulary, r)
    bows.append(b)
df_bow = pd.DataFrame(bows)
#print (df_bow.head())

#Creating the Bag of Words using sklearn:
vect = CountVectorizer()
data = vect.fit_transform(res)
data = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())
#Uncomment the row below to print the first five rows of BoW using sklearn
#print(data.head())


#TF-IDF
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(res)
data_TF_IDF=pd.DataFrame(model.toarray(),columns=vectorizer.get_feature_names_out())
#Uncomment the row below to print the first five rows of TF-IDF
#print(data_TF_IDF.head())

#LDA
lda_model=LatentDirichletAllocation(n_components=5,learning_method='online',random_state=42,)
lda_top=lda_model.fit_transform(model)
#Uncomment the rows below to print LDA Model
'''
print("Topics: ")
for i,topic in enumerate(lda_top[0]):
    print("Topic ",i,": ",topic*100,"%")  '''

# uncomment the code below to see the most important words for each topic
'''vocab = vect.get_feature_names_out()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")'''

#LSA
LSA_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=10)
lsa = LSA_model.fit_transform(model)
l=lsa[0]

#Uncomment the rows below to print LSA Model

'''
print("Topics :")
for i,topic in enumerate(l):
    print("Topic ",i," : ",topic*100) '''

# uncomment the code below to see the most important words for each topic
'''vocab = vect.get_feature_names_out()

for i, comp in enumerate(LSA_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")'''
