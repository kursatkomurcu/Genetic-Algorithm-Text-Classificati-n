import pandas as pd # for datasets
import numpy as np 
import random
import nltk # for tokenization
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

"""
VIDEO LINKI
https://www.youtube.com/watch?v=egLfjYHu69M
"""


def predict(best_individual, sentences):
    """
    best_individual: en iyi birey
    sentences: siniflandirilacak cümle 
    """
    results = []
    pos_arr = best_individual[:len(best_individual) // 2]
    neg_arr = best_individual[len(best_individual) // 2:]

    for i in range(len(sentences)):
        count_pos = 0
        count_neg = 0

        for j in sentences[i]:
            tokens = nltk.word_tokenize(j)
            
            for token in tokens:
                if (token in pos_arr) and (token not in neg_arr): # eğer kelime bireyin ilk yarısında var ikinci yarısında yok ise olumludur
                    count_pos += 1
                if (token in neg_arr) and (token not in pos_arr): # eğer kelime bireyin ilk yarısında yok ikinci yarısında var ise olumsuzdur
                    count_neg += 1
            
            if count_pos > count_neg:
                result = 1 # positive
            elif count_neg > count_pos:
                result = 0 # negative
            else:
                a = ["Negative", "Positive"] # olumlu ve olumsuz kelime sayıları eşit ise rastgele seçim yapılır
                random_a = random.choice(a)
                
                if random_a == a[0]:
                    result = 0
                if random_a == a[1]:
                    result = 1

            results.append(result)

    return results

def fitness_function(individual, x):
    """
    individual: siniflandirma için kullanilacak birey
    x: sadece cümlelerin olduğu veriseti

    İyilik fonksiyonu oluşturulurken öncelikle birey ikiye bölünmüştür.
    Daha sonra veri setindeki cümleler kelimelere ayrilmiştir.
    Ayrilan kelimeler sadece bireyin ilk yarisinda ise olumlu, sadece bireyin ikinci yarisinda ise olumsuz sayilmiştir.
    Daha sonra ise olumlu ve olumsuz sayilarindan fazla olani toplam değişkenine eklenerek fonksiyon toplam değişkenini döndürmüştür.
    """
    toplam = 0
    pos_arr = individual[:len(individual) // 2]
    neg_arr = individual[len(individual) // 2:]
    
    for i in range(len(x)):
        count_pos = 0
        count_neg = 0
        for j in x[i]:
            tokens = nltk.word_tokenize(j)

            for token in tokens:
                if (token in pos_arr) and (token not in neg_arr):
                    count_pos += 1
                if (token in neg_arr) and (token not in pos_arr):
                    count_neg += 1

            if count_pos > count_neg:
                toplam += count_pos

            elif count_neg > count_pos:
                toplam += count_neg

    return toplam

def selection(population, scores, k=3):
    """
    population: bütün bireylerin toplami
    scores: fitness fonksiyonunun döndürdüğü değerlerin oluşturduğu dizi
    k: bireyin kaç parçaya ayrilacağinin sayisi. Örneğin k = 3 ise bireyin 2 parçaya ayrilmasi için seçim yapilir
    """
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]: 
            selection_ix = ix
    return population[selection_ix]

def crossover(parent1, parent2, r_cross):
    """
    parent1: selection fonksiyonu ile seçilen bireylerden birisi
    parent2: selection fonksiyonu ile seçilen bireylerden birisi
    r_cross: crossover olasiliği
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(parent1)-2)
        child1 = np.concatenate((parent1[:pt], parent2[pt:]), axis=None)
        child2 = np.concatenate((parent2[:pt], parent1[pt:]), axis=None)
    return [child1, child2]

def mutation(bitstring, r_mut):
    """
    bitstring: crossoverdan sonra bazi kisimlari değiştirilecek birey
    r_mut: mutasyon olasiliği
    """
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            bitstring[i] = random.choice(word_pool_tokens)
    return bitstring #sonradan eklendi

def genetic_algorithm(x, population, fitness, n_iter, n_pop, r_cross, r_mut):
    """
    x: sadece cümlelerin olduğu veriseti
    population: başta rastgele oluşturduğumuz populasyon
    fitness: fitness function
    n_iter: nesil sayisi
    n_pop: populasyondaki birey sayisi
    r_cross: crossover olasiliği
    r_mut: mutasyon olasiliği
    """
    best_eval_arr = []
    avg_arr = []
    best = 0
    best_eval = fitness(population[0], x)
    for gen in range(n_iter):
        scores = [] # sonradan eklendi
        for c in population:
            score = fitness(c, x)
            scores.append(score)

        for i in range(n_pop):
            if scores[i] > best_eval:
                print("En iyi bireyde Artiş")
                best, best_eval = population[i], scores[i]

        avg = sum(scores) / len(scores)
        print(">%d, new best f(%s) = %f" % (gen,  best, best_eval))
        print("Average fitness function is: ", avg)
        print("Best fitness function is: ", best_eval)

        best_eval_arr.append(best_eval)
        avg_arr.append(avg)
        selected = [selection(population, scores) for _ in range(n_pop)] # ilgili bireyler seçilir
        children = list()

        for i in range(0, n_pop, 2):
            parent1, parent2 = selected[i], selected[i+1]
        
            for c in crossover(parent1, parent2, r_cross): # crossover
                c = mutation(c, r_mut) # mutasyon
                children.append(c)

        population = children

    return [best_eval_arr, avg_arr, best, best_eval]



# Load the data1
data1 = pd.read_csv('datasets/spam.csv')
data1 = data1.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data1['Sentiment'] = data1['Sentiment'].map({'ham': 0, 'spam': 1}) # ham 0, spam 1 olarak etiketlendi
data1 = data1.dropna()
data1 = pd.concat([
    data1[data1['Sentiment'] == 0].sample(250), # verisetinden rastgele 250 adet 0, 250 adet 1 etiketli satır seçildi
    data1[data1['Sentiment'] == 1].sample(250)
])

# Load the data2
# sentiment 0: negative 1: positive
data2 = pd.read_csv('datasets/beyazperde.csv')
data2 = data2[['Sentence', 'Sentiment']]
data2 = data2.dropna() 
data2 = pd.concat([
    data2[data2['Sentiment'] == 0].sample(250),  # verisetinden rastgele 250 adet 0, 250 adet 1 etiketli satır seçildi
    data2[data2['Sentiment'] == 1].sample(250)
])

# Load the data3
data3 = pd.read_csv('datasets/financial_sentiment.csv')
data3['Sentiment'] = data3['Sentiment'].map({'negative': 0, 'positive': 1}) # negative 0, positive 1 olarak etiketlendi
data3 = data3.dropna() 
data3 = pd.concat([
    data3[data3['Sentiment'] == 0].sample(250),  # verisetinden rastgele 250 adet 0, 250 adet 1 etiketli satır seçildi
    data3[data3['Sentiment'] == 1].sample(250)
])

# Load the data4
data4 = pd.read_csv('datasets/magaza_yorumlari_duygu_analizi.csv', encoding='utf-16')
data4['Sentiment'] = data4['Sentiment'].map({'Olumsuz': 0, 'Olumlu': 1}) # Olumsuz 0, Olumlu 1 olarak etiketlendi
data4 = data4.dropna() 
data4 = pd.concat([
    data4[data4['Sentiment'] == 0].sample(250),  # verisetinden rastgele 250 adet 0, 250 adet 1 etiketli satır seçildi
    data4[data4['Sentiment'] == 1].sample(250)
])

# Load the data5
# Sentiment 0: negative 1: positive
data5 = pd.read_csv('datasets/movie.csv')
data5 = data5.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
data5 = data5.dropna() 
data5 = pd.concat([
    data5[data5['Sentiment'] == '0'].sample(250),  # verisetinden rastgele 250 adet 0, 250 adet 1 etiketli satır seçildi
    data5[data5['Sentiment'] == '1'].sample(250)
])


"""
Verisetinin sentence ve sentiment şeklinde ayrildi
Örnek olarak data1 gösterilmiştir
"""
x =data1[['Sentence']] 
y = data1[['Sentiment']]

# x diziye çevrildi
x = np.array(x)

"""
Cümleler kelimelere ayrildiktan sonra stopwords word denilen gereksiz kelimeler ve noktalama işaretleri cümlelerden atilmiştir.
Daha sonra ise her kelime tek bir listede toplanarak veri setinin kelime havuzu oluşturulmuştur.
"""
stop_words = set(stopwords.words('english')) # türkçe veride turkish yazilmali
word_pool = x
word_pool = word_pool.ravel()

word_pool_tokens = []
for i in word_pool:
    token = nltk.word_tokenize(i)
    for word in token:
        if (word not in stop_words) and (word.isalnum()):
            word_pool_tokens.append(word)


length = 100 # ödevdeki N değeri
n_pop = 100 # populasyon büyüklüğü
n_iter = 50  # nesil sayısı
r_cross = 0.7 # crossover rate
r_mut = 0.1 # mutation rate
population = []

for i in range(n_pop): # for döngüsü içerisinde rastgele kelimelerden oluşan populasyon oluşturuluyor
    individual = []
    for j in range(length): # rastgele kelime listesi
        element = random.choice(word_pool_tokens)
        individual.append(element)
    individual = [word for word in individual if not word in stop_words]
    population.append(individual)


#GENETIK ALGORITMA
best_eval_arr, avg_arr, best_individual, best_score = genetic_algorithm(x, population, fitness_function, n_iter, n_pop, r_cross, r_mut)

print('Done!')
print('f(%s) = %f' % (best_individual, best_score))

plt.plot(best_eval_arr, label='En iyi birey')
plt.plot(avg_arr, label='Popülasyon ortalamasi')
plt.legend()
plt.show()

# etiket verimiz diziye dönüştürüldü
y = y.to_numpy()
y = y.ravel()

pred = predict(best_individual, x) # tahmin fonksiyonumuzu çağırdık
print(y[0:10])
print(pred[0:10])

cm = confusion_matrix(y, pred)
sns.heatmap(cm, annot=True)
plt.show()