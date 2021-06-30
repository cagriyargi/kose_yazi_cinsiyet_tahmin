1-veri okuma_yükleme
#!/usr/bin/env python
# coding: utf-8
# In[20]:
import numpy as np #matematik işlemleri, lineer cebir
import pandas as pd #dataframe, matris işlemleri
import os #işletim sistemi
# In[21]:
#ayrı klasör içinden ayrı ayrı veriler çağıracağımız için veriyolu
ekleyip döngü ile veri çağırabileceğimiz
#fonksiyon tanımlandı
#veri yolu
data_path = '/home/Desktop/NLP/NLP_project/data/'
#klasörler
print('Available data', os.listdir(data_path))
#text dosyaları ubuntuda encoding="Windows-1254",
windowsda encoding="utf-8" ile açılabilmiştir.
def read_data(partition):
data = []
for fn in os.listdir(os.path.join(data_path, partition)):
text = open(os.path.join(data_path, partition,
fn),'r',encoding="Windows-1254").read()
data.append(text)
return data
# In[22]:
#kadın ve erkek klasör isimleri path olarak verilerek fonksiyon
ile veri okundu/yüklendi
df_kadin = read_data('kadın')
df_erkek = read_data('erkek')
df_kadin
# In[23]:
#veri dataframe e çevrildi
df_kadin = pd.DataFrame(df_kadin, columns =['text'])
df_kadin.tail()
# In[24]:
#veri dataframe e çevrildi
df_erkek = pd.DataFrame(df_erkek, columns =['text'])
df_erkek.tail()
# In[25]:
#kadın için 250 tane female etketi oluşturuldu
label_kadin=[]
for x in range(250):
label_kadin.append('female')
label_kadin
# In[26]:
#erkek için 250 tane male etketi oluşturuldu
label_erkek=[]
for x in range(250):
label_erkek.append('male')
label_erkek
# In[27]:
#etiket dataframe e çevrildi
label_erkek = pd.DataFrame(label_erkek, columns =['label'])
label_erkek.head()
# In[28]:
#etiket dataframe e çevrildi
label_kadin = pd.DataFrame(label_kadin, columns =['label'])
label_kadin.head()
# In[29]:
#kadın text dataframe ve kadın label dataframe leri birleştirildi
df1=pd.concat([df_kadin,label_kadin], axis=1)
df1.head()
# In[30]:
#erkek text dataframe ve erkek label dataframe leri birleştirildi
df2=pd.concat([df_erkek,label_erkek], axis=1)
df2.head()
# In[31]:
#wordcloud ta kullanmak için cinsiyetler ayrı ayrı csv yapıldı
df1.to_csv("kadin.csv")
df2.to_csv("erkek.csv")
32# In[32]:
#kadın ve erkek dataframeleri birleştirildi
data = pd.concat([df1, df2], ignore_index=True)
data.tail()
# In[33]:
#analizde kullanılacak veriseti kaydedildi
data.to_csv("dataall.csv")
2-Veri önişlem, EDA
#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np #lineer cebir kütüphanesi
import pandas as pd #veri okuma/ analiz kütüphanesi
import os #işletim sistemi uyumu için
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl
import matplotlib.pyplot as plt #grafik plot
import seaborn as sns #grafik plot
# In[29]:
mpl.rcParams['figure.figsize']=(8.0,6.0) #(6.0,4.0)
mpl.rcParams['font.size']=12 #10
mpl.rcParams['savefig.dpi']=100 #72
mpl.rcParams['figure.subplot.bottom']=.1
# In[2]:
data=pd.read_csv('dataall.csv',encoding="utf8")
kadin =
pd.read_csv("/home/Desktop/NLP/NLP_project/kadin.csv"
)
erkek =
pd.read_csv("/home/Desktop/NLP/NLP_project/erkek.csv"
)
# In[31]:
data.head()
# In[32]:
kadin.head()
# In[33]:
import nltk #doğal dil işleme kütüphanesi
nltk.download('stopwords') #stopwords yükleme
# In[34]:
stopwords = set(STOPWORDS)
# In[35]:
wordcloud1 = WordCloud(
background_color='white',
stopwords=stopwords,
max_words=200,
max_font_size=40,
random_state=42
).generate(str(kadin['text']))
# In[36]:
wordcloud2 = WordCloud(
background_color='white',
stopwords=stopwords,
max_words=200,
max_font_size=40,
random_state=42
).generate(str(erkek['text']))
# In[37]:
print(wordcloud1)
fig = plt.figure(1)
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()
fig.savefig("wordkadin.png", dpi=900)
# In[38]:
print(wordcloud2)
fig = plt.figure(1)
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()
fig.savefig("worderkek.png", dpi=900)
# In[39]:
#işleme etkisi olmayan kelimeler kaldırılır
etkisiz_kelimeler = 'İclal Aydın - iclal@gazetevatan.com'
#stopwords kaldırma fonksiyonu
33def etkisizleriKaldir(text):
return " ".join([word for word in str(text).split() if word not in
etkisiz_kelimeler])
# In[40]:
kadin["text_i"] = kadin["text"].apply(lambda text :
etkisizleriKaldir(text))
kadin.head()
# In[41]:
wordcloud3 = WordCloud(
background_color='white',
stopwords=stopwords,
max_words=200,
max_font_size=40,
random_state=42
).generate(str(kadin['text_i']))
# In[42]:
print(wordcloud3)
fig = plt.figure(1)
plt.imshow(wordcloud3)
plt.axis('off')
plt.show()
fig.savefig("wordkadin_i.png", dpi=900)
# In[43]:
#tüm verideki yazar adı kaldırılır
data["text"] = data["text"].apply(lambda text :
etkisizleriKaldir(text))
data.head()
# In[44]:
#csv yapıp tekrar okutmadan kaynaklanan gereksiz column u
silme
data = data.drop(['Unnamed: 0'],axis=1)
data.head()
# In[45]:
#tekrar eden verileri kontrol etme
print("Duplitace toplam sayısı:")
print(data.duplicated().sum())
#Null veri olup olmadığını kontrol etme
print("Null value toplam sayısı")
print(data.isnull().sum())
# In[46]:
#tekrar eden verileri silme
data = data.drop_duplicates()
#index kontrolü
data
# In[48]:
#analizde kullanılacak veriseti kaydedildi
data.to_csv("dataall2.csv")
# In[49]:
#Data da 249 kadın, 234 erkek içerir (performansı etkilemesin
diye duplicate sildiğimiz için azaldı)
data['label'].value_counts()
# In[52]:
#Data label grafiği, hemen hemen eşit
sns.countplot(data.label)
plt.title('Label sayısı')
plt.show()
3-Veriseti önişlemleri NLP
#!/usr/bin/env python
# coding: utf-8
# In[21]:
import numpy as np #lineer cebir kütüphanesi
import pandas as pd #veri okuma/ analiz kütüphanesi
import os #işletim sistemi uyumu için
import re #regular ifade işlemleri için kütüphane
pd.options.mode.chained_assignment = None #uyarıyı
görmezden gelmek için
import sys
import nltk #doğal dil işleme kütüphanesi
nltk.download('stopwords') #stopwords yükleme
from nltk.corpus import stopwords #stopwords işlemleri
from TurkishStemmer import TurkishStemmer #Türkçe
stemming işlemleri
import string
from sklearn.model_selection import train_test_split #train test
split ayırma
from sklearn.feature_extraction.text import CountVectorizer
#BoW ile vektörleştirme
from sklearn.feature_extraction.text import TfidfVectorizer
#TF-IDF ile vektörleştirme
34import matplotlib.pyplot as plt #grafik plot
import seaborn as sns #grafik plot
# In[22]:
#data tablosunun gösterimi,
data=pd.read_csv('dataall2.csv',encoding="utf8")
data
# In[23]:
#csv yapıp tekrar okutmadan kaynaklanan gereksiz column u
silme
data = data.drop(['Unnamed: 0'],axis=1)
# In[28]:
#index kontrolü
data.tail()
# In[29]:
#NLP işlemleri için text colomnu yeni dataframe e atandı
df = data[['text']]
df
# In[30]:
#NLP işlemleri
#gereksiz boşlukların kaldırılması
df['text'] = (df['text'].str.split()).str.join(' ')
df.head()
# In[31]:
#harfleri küçültme
df["text_lower"] = df['text'].str.lower()
df.head()
# In[32]:
#işleme etkisi olmayan noktama işaretleri kaldırılır
#string punc'un içindeki default noktalama işaretleri dikkate alır,
ayrıca kesme işareti eklendi
kaldirilan_noktalamalar = string.punctuation+u"’"
#noktalama kaldırma fonksiyonu
def noktalamaKaldir(text):
return
text.translate(str.maketrans('','',kaldirilan_noktalamalar))
# In[33]:
#noktalama işaretleri kaldırma
df["text_isaretsiz"] = df["text_lower"].apply(lambda text:
noktalamaKaldir(text))
df.head()
# In[34]:
#işleme etkisi olmayan kelimeler kaldırılır
etkisiz_kelimeler = set(stopwords.words('turkish'))
#stopwords kaldırma fonksiyonu
def etkisizleriKaldir(text):
return " ".join([word for word in str(text).split() if word not in
etkisiz_kelimeler])
# In[35]:
#stopwords kaldırma
df["text_stopsuz"] = df["text_isaretsiz"].apply(lambda text :
etkisizleriKaldir(text))
df.head()
# In[38]:
#kelime kelime ayırma işlemi (tokenize)
df['tokenize'] = df['text_stopsuz'].str.split() #yazıdaki tüm
kelimeleri ayırma işlemi
# In[39]:
#stemmer, ayrılan her kelimenin kökünü bulma işlemi
stemmer = TurkishStemmer()
df['stemmed'] = df['tokenize'].apply(lambda x: [stemmer.stem(y)
for y in x])
df.head()
# In[40]:
#data framein son hali
data["text"] = df['stemmed']
data.head()
# In[41]:
data.to_csv("stemmed.csv")
4-Encoding_veri analizi_model geliştirme_metrics
#!/usr/bin/env python
# coding: utf-8
# In[170]:
import numpy as np #lineer cebir kütüphanesi
import pandas as pd #veri okuma/ analiz kütüphanesi
import os #işletim sistemi uyumu için
import re #regular ifade işlemleri için kütüphane
pd.options.mode.chained_assignment = None #uyarıyı
görmezden gelmek için
35import sys
import nltk #doğal dil işleme kütüphanesi
nltk.download('stopwords') #stopwords yükleme
from nltk.corpus import stopwords #stopwords işlemleri
from TurkishStemmer import TurkishStemmer #Türkçe
stemming işlemleri
import string
#from sklearn.model_selection import train_test_split #train test
split ayırma
from sklearn.feature_extraction.text import CountVectorizer
#BoW ile vektörleştirme
from sklearn.feature_extraction.text import TfidfVectorizer
#TF-IDF ile vektörleştirme
import matplotlib.pyplot as plt #grafik plot
import seaborn as sns #grafik plot
# In[171]:
data=pd.read_csv('stemmed.csv',encoding="utf8")
data.head()
# In[172]:
data=data.drop(['Unnamed: 0'], axis=1)
data.head()
# In[173]:
#işleme etkisi olmayan noktama işaretleri kaldırılır
#string punc'un içindeki default noktalama işaretleri dikkate alır,
ayrıca kesme işareti eklendi
kaldirilan_noktalamalar = string.punctuation+u"’"
#noktalama kaldırma fonksiyonu
def noktalamaKaldir(text):
return
text.translate(str.maketrans('','',kaldirilan_noktalamalar))
# In[174]:
#vectörleştirme için stemming de oluşan noktalama işaretleri
kaldırılır
data['text'] = data['text'].apply(lambda text:
noktalamaKaldir(text))
data.head()
# In[175]:
#Label sayısallaştırılır
#female:0 male:1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label']=le.fit_transform(data['label'])
data
# In[176]:
#train test ayırmak için X ve y belirlenir
X= data['text']
y= data['label']
# In[177]:
X.head()
# In[178]:
y.head()
# In[179]:
#Bag of Words ile vektörleştirme
#Döküman başına
#X verisivektörleştirilir
# en az 5 belgede yer alan sözcükleri dikkate alır
vectorizer = CountVectorizer(min_df=5)
X_bow= vectorizer.fit_transform(X)
X_bow
# In[180]:
print(X_bow)
#(döküman indexi,terim indexi) terim sayısı
# In[181]:
#TF-IDF 1
##TF: Term Frequency / Terim Sıklığı
#Terim sıklığı; seçili terimimizin, metin içinde bulunan toplam
terimler sayısına bölümüdür.
##IDF: Inverse Document Frequency / Ters Döküman Sıklığı
#Ters Döküman Sıklığı; metinlerimizin kaçında terimimiz var
bunu gösterir.
#Toplam metin adetimizin terimi içeren metin adetine
bölümünün logaritmasıdır.
#TF-IDF Değeri; bu iki değerin çarpımı ile elde edilir.
vectorizer = TfidfVectorizer(min_df=5) #en az 5 belgede yer
alan sözcükleri dikkate alır
X_tfidf1 = vectorizer.fit_transform(X)
X_tfidf1
# In[182]:
print(X_tfidf1)
#(döküman indexi,terim indexi) TF*IDF Değeri
36# In[183]:
#TF-IDF 2
#parametreler
#analyzer özelliğin kelime ngramlarından oluşacağı,
#ngram_range(1, 2) unigram ve bigram , farklı n-gramlar için
n-değerleri aralığının alt ve üst sınırı.
#max_features, terim sıklığına göre en üst sınır
##en az 5 belgede yer alan sözcükleri dikkate alır
vectorizer = TfidfVectorizer(analyzer = "word", ngram_range
=(1, 3),max_features = 750, min_df=5)
X_tfidf2 = vectorizer.fit_transform(X)
X_tfidf2
# In[184]:
print(X_tfidf2)
#(döküman indexi,terim indexi) TF-IDF Değeri
# In[185]:
#VERİ ANALİZİ-MODEL GELİŞTİRME
from sklearn.metrics import accuracy_score, confusion_matrix
#metrikler için kütüphane
from sklearn import linear_model, naive_bayes, svm #modeller
için kütüphane
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #train test
ayırma için kütüphane
# In[186]:
#BoW ile hazırlanan vektör ile model geliştirme
type(X_bow)
# In[187]:
#modeli uygulayabilmek için arraya çevrilir
X_bow= X_bow.toarray()
type(X_bow)
# In[188]:
X_bow.size
# In[189]:
#Bow ile vektörleştirilmiş veriyi modele vermeden train test
split ile rastgele ayırdık
X_train, X_test, y_train, y_test = train_test_split(X_bow,
y,test_size = 0.2, random_state=42)
# In[190]:
#Naive Bayes
model = naive_bayes.MultinomialNB()
model.fit(X_train,y_train)
NB_prediction = model.predict(X_test)
ACC_bow_nb=accuracy_score(NB_prediction, y_test)
print("Naive Bayes :",ACC_bow_nb)
cm_bow_nb=confusion_matrix(y_test, NB_prediction)
cm_bow_nb
# In[191]:
#Logistic Regresyon
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
LR_prediction = model.predict(X_test)
ACC_bow_lr=accuracy_score(LR_prediction, y_test)
print("Logistic regression :",ACC_bow_lr)
cm_bow_lr=confusion_matrix(y_test, LR_prediction)
cm_bow_lr
# In[192]:
#Support Vector Machine
model = svm.SVC()
model.fit(X_train,y_train)
SVM_prediction = model.predict(X_test)
ACC_bow_svm=accuracy_score(SVM_prediction, y_test)
print("Support Vector Machines :",ACC_bow_svm)
cm_bow_svm=confusion_matrix(y_test, SVM_prediction)
cm_bow_svm
# In[193]:
#KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
KNN_prediction = model.predict(X_test)
ACC_bow_knn=accuracy_score(KNN_prediction, y_test)
print("K Nearest Neighbor :",ACC_bow_knn)
cm_bow_knn=confusion_matrix(y_test, KNN_prediction)
cm_bow_knn
# In[194]:
#TF IDF ile hazırlanan vektör ile model geliştirme
37type(X_tfidf1)
# In[195]:
#modeli uygulayabilmek için arraya çevrilir
X_tfidf1= X_tfidf1.toarray()
type(X_tfidf1)
# In[196]:
X_tfidf1.size
# In[197]:
#TF-IDF 1 ile vektörleştirilmiş veriyi modele vermeden train
test split ile rastgele ayırdık
X_train, X_test, y_train, y_test = train_test_split(X_tfidf1,
y,test_size = 0.2, random_state=42)
# In[198]:
#Naive Bayes
model = naive_bayes.MultinomialNB()
model.fit(X_train,y_train)
NB_prediction = model.predict(X_test)
ACC_tfidf1_nb=accuracy_score(NB_prediction, y_test)
print("Naive Bayes :",ACC_tfidf1_nb)
cm_tfidf1_nb=confusion_matrix(y_test, NB_prediction)
cm_tfidf1_nb
# In[199]:
#Logistic Regresyon
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
LR_prediction = model.predict(X_test)
ACC_tfidf1_lr=accuracy_score(LR_prediction, y_test)
print("Logistic regression :",ACC_tfidf1_lr)
cm_tfidf1_lr=confusion_matrix(y_test, LR_prediction)
cm_tfidf1_lr
# In[200]:
#Support Vector Machine
model = svm.SVC()
model.fit(X_train,y_train)
SVM_prediction = model.predict(X_test)
ACC_tfidf1_svm=accuracy_score(SVM_prediction, y_test)
print("Support Vector Machines :",ACC_tfidf1_svm)
cm_tfidf1_svm=confusion_matrix(y_test, SVM_prediction)
cm_tfidf1_svm
# In[201]:
#KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
KNN_prediction = model.predict(X_test)
ACC_tfidf1_knn=accuracy_score(KNN_prediction, y_test)
print("K Nearest Neighbor :",ACC_tfidf1_knn)
cm_tfidf1_knn=confusion_matrix(y_test, KNN_prediction)
cm_tfidf1_knn
# In[202]:
#TF IDF ile parametrler değitirilerek hazırlanan vektör ile model
geliştirme
type(X_tfidf2)
# In[203]:
#modeli uygulayabilmek için arraya çevrilir
X_tfidf2= X_tfidf2.toarray()
type(X_tfidf2)
# In[204]:
X_tfidf2.size
# In[205]:
#TF-IDF 2 ile vektörleştirilmiş veriyi modele vermeden train
test split ile rastgele ayırdık
X_train, X_test, y_train, y_test = train_test_split(X_tfidf2,
y,test_size = 0.2, random_state=42)
# In[206]:
#Naive Bayes
model = naive_bayes.MultinomialNB()
model.fit(X_train,y_train)
NB_prediction = model.predict(X_test)
ACC_tfidf2_nb=accuracy_score(NB_prediction, y_test)
print("Naive Bayes :",ACC_tfidf2_nb)
cm_tfidf2_nb=confusion_matrix(y_test, NB_prediction)
cm_tfidf2_nb
# In[207]:
#Logistic Regresyon
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
38LR_prediction = model.predict(X_test)
ACC_tfidf2_lr=accuracy_score(LR_prediction, y_test)
print("Logistic regression :",ACC_tfidf2_lr)
cm_tfidf2_lr=confusion_matrix(y_test, LR_prediction)
cm_tfidf2_lr
# In[208]:
#Support Vector Machine
model = svm.SVC()
model.fit(X_train,y_train)
SVM_prediction = model.predict(X_test)
ACC_tfidf2_svm=accuracy_score(SVM_prediction, y_test)
print("Support Vector Machines :",ACC_tfidf2_svm)
cm_tfidf2_svm=confusion_matrix(y_test, SVM_prediction)
cm_tfidf2_svm
# In[209]:
#KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
KNN_prediction = model.predict(X_test)
ACC_tfidf2_knn=accuracy_score(KNN_prediction, y_test)
print("K Nearest Neighbor :",ACC_tfidf2_knn)
cm_tfidf2_knn=cm_tfidf2_nb=confusion_matrix(y_test,
KNN_prediction)
cm_tfidf2_knn
# In[210]:
accuracy=np.array([ACC_bow_nb,ACC_bow_lr,ACC_bow_sv
m,ACC_bow_knn,ACC_tfidf1_nb,ACC_tfidf1_lr,ACC_tfidf1_s
vm,ACC_tfidf1_knn,ACC_tfidf2_nb,ACC_tfidf2_lr,ACC_tfidf2
_svm,ACC_tfidf2_knn])
accuracy
# In[211]:
clf=np.array(['NB_bow','LR_bow','SVM_bow','KNN_bow','NB_
tfidf1','LR_tfidf1','SVM_tfidf1','KNN_tfidf1','NB_tfidf2','LR_tfi
df2','SVM_tfidf2','KNN_tfidf2'])
clf
# In[212]:
df_a = pd.DataFrame(accuracy, columns = ['accuracy'])
df_a
# In[213]:
df_c = pd.DataFrame(clf, columns = ['model'])
df_c
# In[214]:
grafik=pd.concat([df_a, df_c], axis=1)
grafik
# In[215]:
sns.set(rc={'figure.figsize':(13.7,8.27)})
ax = sns.barplot(x="model", y="accuracy", data=grafik)
# In[216]:
y_test.value_counts()
# In[258]:
y_train.value_counts()
# In[245]:
ax = sns.heatmap(cm_bow_nb, square=True, annot=True,
annot_kws={"size": 30}, cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of NB (Bow vector)',fontsize =
16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[246]:
ax = sns.heatmap(cm_bow_lr, square=True, annot=True,
annot_kws={"size": 30}, cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of LR (Bow vector)',fontsize =
16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[247]:
ax = sns.heatmap(cm_bow_svm, square=True,
annot=True,annot_kws={"size": 30}, cbar=False, cmap='Blues',
fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of SVM (Bow vector)',fontsize =
16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
39ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[248]:
ax = sns.heatmap(cm_bow_knn, square=True,
annot=True,annot_kws={"size": 30}, cbar=False, cmap='Blues',
fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of KNN (Bow vector)',fontsize =
16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[249]:
ax = sns.heatmap(cm_tfidf1_nb, square=True,
annot=True,annot_kws={"size": 30}, cbar=False, cmap='Blues',
fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of NB (TF-IDF-1 vector)',fontsize
= 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[250]:
ax = sns.heatmap(cm_tfidf1_lr, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of LR (TF-IDF-1 vector)',fontsize
= 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[251]:
ax = sns.heatmap(cm_tfidf1_svm, square=True, annot=True,
annot_kws={"size": 30}, cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of SVM (TF-IDF-1
vector)',fontsize = 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[252]:
ax = sns.heatmap(cm_tfidf1_knn, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of KNN (TF-IDF-1
vector)',fontsize = 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[253]:
ax = sns.heatmap(cm_tfidf2_nb, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of NB(TF-IDF-2 vector)',fontsize
= 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[254]:
ax = sns.heatmap(cm_tfidf2_lr, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of LR (TF-IDF-2 vector)',fontsize
= 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
# In[255]:
ax = sns.heatmap(cm_tfidf2_svm, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of SVM (TF-IDF-2
vector)',fontsize = 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
40plt.show()
# In[256]:
ax = sns.heatmap(cm_tfidf2_knn, square=True, annot=True,
annot_kws={"size": 30},cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
ax.set_title('Confusion Matrix of KNN (TF-IDF-2
vector)',fontsize = 16)
ax.xaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize = 15)
plt.show()
