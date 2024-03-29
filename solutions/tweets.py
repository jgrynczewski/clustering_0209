# data from https://www.kaggle.com/benhamner/clinton-trump-tweets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Na początku wstęp teoretyczny - przypomnienie działania metody TF-IDF z NLP.

# Grupowanie hierarchiczne działa na odległościach.
# Więc jak zamierzamy reprezentować odległość pomiędzy dwoma tweetami ?

# Jednym z prostszych sposobów jest użycie miary TF-IDF (ang. Term Frequency - Inverse Document Frequency)

# TF-IDF

# Konwertujemy tweet do postaci wektorowej (po zrobieniu tego obliczenie odległości
# będzie trywialne). Żeby to zrobić musimy zdefiniować jakąś przestrzeń wektorową
# żeby umieszczać w niej wektory. Wartością tego wektora może być liczba wystąpień 
# słowa reprezentowanego przez ten wektor w tweecie. Na przykład"

# "Ala ma kota"
# Words count:
# Ala: 1
# ma: 1
# kota: 1

# I w ten sposób zamieniamy zdanie/tweet "Ala ma kota" na werkto (1, 1, 1)
# (ala, kota, ma) = (1, 1, 1)  # (tutaj w kolejnosci alfabetycznej)

# Ten sposób przedtawienia zdania za pomocą wektora można nazwać TF - Term Frequency.
# Ale z TF jest jeden problem. Są słowa które występują nieproporcjonalnie częściej
# od innych. I najczęściej słowa te nie są ważne (co będzie mylące, kiedy
# wektoryzację oprzemy wyłącznie na liczbie wystąpień słowa w pojedynczym tweecie),
# np. the, a, this. 
# Wyobraźmy sobie, że próbujemy rozróżnić dokumenty/tweety na temat fizyki od
# dokumentów/tweetów na temat biologi. Te słowa nam w tym nie pomogą, a częstością
# swojego występowania będą umniejszały znaczenia słową na prawdę znaczącym.
# Nas bardziej interesowałyby słowa np. DNA, ewolucja, tkanka, grawitacja, tarcie, elektorn

# Czyli sformułujmy problem:
# 1. Słowa mało znacznące występują znacznie częściej niż słowa znaczące
# 2. Słowa mało znaczące pojawiają się we wszystkich dokumentach.

# I tutaj wchodzi IDF
# TF - term frequency - to co przed chwilą robiliśmy - zliczanie słów
# IDF - 1/(liczba dokuemntów, w którym to słowo się pojawiło) -> 0 < IDF < 1

# A TF-IDF oznacza:
# IDF * TF

# czyli ten człon IDF będzie zmniejszał (skalował) nam znaczenie słów które pojawią 
# się we wszystkich dokumentach. Jeżeli słowo pojawia się w wielu dokumentach to człon
# IDF będzie odpowiednio mały (mniejszy od 1) i tym samym wartość TF zostanie odpowiednio
# przeskalowana (zmniejszona).

# I tak robimy dla każdego tweeta w efekcie otrzymując macierz.
# To jak przekonwetujemy tweete na macierz?
# Będziemy używać modelu sklear.feature_extraction.text.TfidfVectorizer

# Jak każdy model, TfidfVectorizer posiada metodę fit i transform no i fit_transform
# wszystko co trzeba zrobić to przekazać do niej korpus (listę dokumentów, u nas tweetów)

# W odpowiedzi dostajemy NxD matrix, gdzie 
# N-liczba tweeetów 
# D-rozmiar słownika (wszystkich znalezionych w tweetach słów)

# Po wektoryzacji jesteśmy gotowi, żeby wykonać klastrowanie hierarchiczne.
# Liczymy na znalezienie dwóch odrębnych grup klastrów (oddzielny dla
# Hilary i oddzielny dla Trumpa). Oczywiście to jest uczenie nienadzorowane, 
# więc w zasadzie otrzymane klastry mogą oznaczać cokolwiek.

# Używamy oznakowanych danych, więc tak naprawdę po prostu sprawdzamy jak poradzi
# sobie z tym problemem algorytm klasteryzacji hierarchicznej.

# Średni tweet
# --------------
# Ponadto, ponieważ każdy tweet jest teraz wektorem, możemy spróbować znaleźć średni wektor 
# (centroid) dla naszych klastrów. Spodziewamy się znaleźć w nim słowa, które są centralne
# dla wyłonionego klastra.

# Usuńmy z analizy słowa, które nic nie znaczą, a często mogą występować.
# Lista stopwords została wybrana na podstawie wstępnej analizy danych.
stopwords = [
  'the',
  'about',
  'an',
  'and',
  'are',
  'at',
  'be',
  'can',
  'for',
  'from',
  'if',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'this',
  'to',
  'you',
  'your',
  'with',
]


# W tweetach występuję sporo urli, które nie niosą ze sobą żadnej treści. Ten
# wniosek został wysnuty również na podstawie wstępnej analizy danych. Usuńmy te urle.
# regex do znajdywania url - chcemy je usuwać:
import re
url_finder = re.compile(r"(?:\@|https?\://)\S+")


# Funkcja zamienia wszystkie wielkie litery na małe i usuwa urle 
def filter_tweet(s):
    s = s.lower()  # downcase
    s = url_finder.sub("", s) # każdy ciąg znaków pasujący do wzorca url_finder 
    # zastępujemy pustym ciągiem znaków
    
    return s

### Wcyztajmy dnae ###
df = pd.read_csv('data/tweets.csv')  # wczytujemy
text = df.text.tolist()  # konwerujemy do listy
text = [filter_tweet(s) for s in text]  # filtrujemy każdy tweet

# Teraz TF-IDF
# max_feature=100, interesuje nas 100 najczęstszych słów, nie więcej

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()  
# Po co metoda todense ?
# fit_transform zwraca obiekt obiekt klasy scipy.SparseMatrix
# (reprezentujący macierz rzadką), którego poniższy kod nie umiałby obsłużyć.
# Dlatego wołamy todense(), żeby skonwertować wynik do odpowiedniego
# formatu. TFIDFVectorizer zwraca obiekt klasy scipy.SparseMatrix, ponieważ 
# większość elementów w macierzy TF-IDF kończy z wartością 0. 
# Więc reprezentowanie jej jako SparseMatrix jest pamięciowo znacznie wydajniejsze.

# Robimy hierarchiczną klastreyzację O(n^2) więc reprupkujemy (taki mini-batch, o którym
# mówiliśmy przy algorytmie mini-batch k-menas), w przeciwnym razie wyliczanie odległości 
# będzie ekstremalnie wolne
N = X.shape[0]
idx = np.random.choice(N, size=2000, replace=False)
x = X[idx]
labels = df.handle[idx].tolist()  # przy okazji przypisujemy etykiety

# Upewniamy się, że oba rodzaje tweetów są równo reprezentowane w naszych dnaych
pTrump = sum(1.0 if e == 'realDonaldTrump' else 0.0 for e in labels) / len(labels)
print("proportion @realDonaldTrump: %.3f" % pTrump)  # Proporcja tweetów Donalda Trumpa
print("proportion @HillaryClinton: %.3f" % (1 - pTrump))  # Proporcja tweetów Hilary Clinton

# liczymy odległości za pomocą funkcji pdsit. Wcześniej tego nie robiliśmy, ponieważ
# funkcja linkage potrafi sobie wyliczyć odległości (o ile w danych są wyłącznie wartości
# liczbowe). Ale możemy też podać funkcji linkage macierz z gotowymi odległościami 
# i tak zrobimy w tym przypadku.
from scipy.spatial.distance import pdist
dist_array = pdist(x)

# Liczymy
from scipy.cluster.hierarchy import linkage
Z = linkage(dist_array, 'ward')
plt.title("Ward")  # simple, całkowicie zawiedzie, complete - słabe wyniki

# popatrzmy na dendogram
from scipy.cluster.hierarchy import dendrogram
dendrogram(Z, labels=labels)
plt.show()

# Sprawdźmy czystość (ang. purity) - coś jak accuracy tylko, że w klasteryzacji.
# Możemy to zrobić, bo dysponujemy przecież etykietami (ground true).
# Żeby to zrobić musimy przypisać klastry do etykiet.
# Ważne, żeby wiedzieć że scipy zwraca przypisane klastry
# od 1 do k, a nie od 0 do k-1. Więc kiedy przypisujemy etykiety
# musimy pamiętać żeby przypisać 1 i 2 a nie 0, 1
Y = np.array([1 if e == 'realDonaldTrump' else 2 for e in labels])

# rozcinamy (przypisujemy skupiska), za pomocą funkcji fcluster
# 9 wybraliśmy empirycznie, na podstawie wyświetlonego dendogramu
# (po rozcięciu na poziomie 9 dostajemy 2 skupiska)
from scipy.cluster.hierarchy import fcluster
y_ahc = fcluster(Z, 9, criterion='distance')

categories = set(y_ahc)
print("values in C:", categories)  # dla pewności sprawdzamy. Powinniśmy
# dostać wyłącznie 1 i 2

# funkcja, która liczy purity poszczególnych klastrów.
# (wcześniej do tej analizy wykorzystywaliśmy pd.crosstab)
def purity(true_labels, cluster_assignments, categories):
  # maksymalna wartość purity to 1. Im większa wartość purity tym lepiej.
  N = len(true_labels)

  total = 0.0
  for k in categories:
    max_intersection = 0
    for j in categories:
      intersection = ((cluster_assignments == k) & (true_labels == j)).sum()
      if intersection > max_intersection:
        max_intersection = intersection
    total += max_intersection
  return total / N

# Wyświetlamy sumaryczne purity
print("purity:", purity(Y, y_ahc, categories))
# purity jest 0.551 czyli nienajgorzej

# Teraz purity dla poszczególnych skupis.

# Wiemy, że mniejszym skupiskiem jest skupisko z tweetami
# Donalda Trumpa. Uwaga! Nazywamy  ten klaster
# klastrem Donalda Trumpa po zaobserwowaniu, że większość
# tweetów w tym skupisku jest Donalda Trumpa. Tak naprawdę
# nie mamy pojęcia czym jest ten klaster. Ale istnieje
# i większość należących do niego tweetów jest Donalda 
# Trumpa. I ten klaster jest mały
if (y_ahc == 1).sum() < (y_ahc == 2).sum():
  d = 1  # donald
  h = 2  # hilary
else:
  d = 2  # donald
  h = 1  # hilary

# ustawiamy tweety donalda truma jako przecięcie
# klastra d i label 1
actually_donald = ((y_ahc == d) & (Y == 1)).sum()

# to podzielone przez liczbę faktycznych tweetów trumpa jest proporcją
# tweetów w tym klastrze tweetów Donalda Trumpa
donald_cluster_size = (y_ahc == d).sum()
print("purity of @realDonaldTrump cluster:", float(actually_donald) / donald_cluster_size)
# ponad 90%, jestesmy bliscy pewności że to klaster Trumpa

actually_hillary = ((y_ahc == h) & (Y == 2)).sum()
hillary_cluster_size = (y_ahc == h).sum()
print("purity of @HillaryClinton cluster:", float(actually_hillary) / hillary_cluster_size)
# > 50 % hilary - ten klaster nie jest taki czysty, ale poprzedni na pewno należy do Donalda
# Trumpa, wiec ten przypisujemy do Hilary Clinton.

# Wniosek - znacznie łatwiej rozpoznać tweety trumpa

# Sprawdźmy jeszcze jakie są najbardziej reprezentatywne słowa w posczegłonych klastrach

# słowa z najwyższą wartością tf-idf w klastrze 1? in cluster 2?
w2i = tfidf.vocabulary_


# przetwarzamy dane z macierzy do listy
d_avg = np.array(x[y_ahc == d].mean(axis=0)).flatten()
d_sorted = sorted(w2i.keys(), key=lambda w: -d_avg[w2i[w]])
print("\n10 najczęstszucj słów w klastrze 'Donald Trump'")
print("\n".join(d_sorted[:10]))

h_avg = np.array(x[y_ahc == h].mean(axis=0)).flatten()
h_sorted = sorted(w2i.keys(), key=lambda w: -h_avg[w2i[w]])
print("\n10 najczęstszych słów w klastrze 'Hillary Clinton'")
print("\n".join(h_sorted[:10]))
# Drugi klaster w ogóle nie jest ciekawy, ogólniki

# Wniosek
# Kampania Donalda Trumpa była bogata w zapadające w pamięć slogany.
# Patrząc na drugi klaster widzimy bardzo ogólne, nieiwiele znaczące słowa.

# Warto spróbować rozwiązać to zadanie za pomocą algorytmu k-means, ponieważ
# algorytm k-means również daje w tym przypadku sensowne rezultaty.