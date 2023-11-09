from itertools import count
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

from datetime import datetime

from dateutil import relativedelta

from sklearn import preprocessing




#2.1 Visualisation des données


data = pd.read_csv("customer_database.csv",";")
data = pd.DataFrame(data)


#resumé 
print(data.describe())

#Histogramme
data.hist(column='Meat')
plt.show()

#Types des variables
print("types")
print(data.dtypes)

print("Number of values")
print(data.size)
#64360

print("Shape")
print(data.shape)
#(2240,39)


#2.2 Pré-traitement




#Affichage des valeurs uniques 
#print(set(data["Education"]))
#print(set(data["CivilStatus"]))

#transformation des variables qualitatives en numériques
le = preprocessing.LabelEncoder()
education = le.fit_transform(data["Education"])
data["Education"] = education #remplacement
civilstatus = le.fit_transform(data["CivilStatus"])
data["CivilStatus"] = civilstatus  #remplacement

#Vérification
#print(data["CivilStatus"])



#Tranformation de la date d'enregistrement en nombre d'années

list_registration = data["Registration"] 
#transform date of registration into date object
dates_registration = [datetime.strptime(date,'%d-%m-%Y') for date in list_registration]
#list of number of year
list_registration_nby = [relativedelta.relativedelta(datetime.today(),date ).years for date in dates_registration]
data["Registration"] = list_registration_nby #remplacement
#Vérification
#print(data["Registration"])



#Transformation de l'année de naissance en âge
list_year = data["BirthYear"]
data["BirthYear"] = [ datetime.today().year - y for y in list_year]



#Suppression des variables inintérressantes 
data = data.drop(["Accept3","Accept4","Accept5","Accept1","Accept2","Claims","Group","SubGroup","AcceptLast","ID"],axis=1)



#retourne le nombre d'élément NA pour chaque variable
for head in data.head() :
    print( head ,"nb of VA :", data[head].isna().sum())
    
#Income possède 24 valeur manquante
# Remplissage de Income par sa valeur moyenne
value = {"Income": data["Income"].mean()}
data = data.fillna(value)


#Normalisation et centrage
scaler = StandardScaler()
data_Z = scaler.fit_transform(data)
#Vérification 
#print(np.mean(data_Z,0))  #moyenne nulle

#print(np.var(data_Z,0))  #varinance à 1

#Après pré-taitement : shape (2240,19), size 42560








#2.3 Analyse de la corrélation

corr =data.corr()
print(corr)

#Avec Seaborn
#print(sns.heatmap(corr))
# plt.title("corrélation")
# plt.show()





#2.4 Réalisation de l'ACP
# pca = PCA(19)
# fit = pca.fit(data_Z)

#Qualité globale pour 7 axes
# QG = np.cumsum(pca.explained_variance_[7])
# print(QG)

#Graphe des valeurs propres et du cumul d'inertie
# plt.plot(  [ i for i in range(19)], pca.explained_variance_ )
# plt.plot(  [ i for i in range(19)], np.cumsum(pca.explained_variance_ratio_))

# plt.title("Inertie analyse")
# plt.legend(["valeurs propes","Inertie cumulée"], loc='upper right')
# plt.show()

#Critère de selection du nombre d'axes
#qualité globale à 70% pour 7 axes  (90% pour 14)
#Coude  --> 4 axes

#Réalisation de l'ACP pour 4 axes
pca4 = PCA(4)
fit1 = pca4.fit(data_Z)
Zp4 = pca4.transform(data_Z)   #Projection sur les 4 axes (calcul des coordonnées)

#représentation des individus

#plt.scatter(Zp4[:,0],Zp4[:,1])

#for i in range(len(Zp4[:,0])):
#    plt.text(Zp4[i,0],Zp4[i,1],data[i])
# plt.title("Représentation des individus")
# plt.xlabel("Axe1")
# plt.ylabel("Axe2")
# plt.xlim(-5,8)
# plt.ylim(-3,7)
# plt.show()

#représentation des variables, cerlce de corrélation
var = ["BirthYear","Education","CivilStatus","Income","Kids","Teens","Registration","LastPurchase","Wines","Fruits","Meat","Fish","Sweet","Luxury","DiscountPurchases","WebPurchases","CatalogPurchases","StorePurchases","WebVisits"]


# axe1 = 0
# axe2 = 1
# for j in range(len(pca4.components_[0])) :

#     corr1 = pca4.components_[axe1,j]*np.sqrt(pca4.explained_variance_[axe1])
#     corr2 = pca4.components_[axe2,j]*np.sqrt(pca4.explained_variance_[axe2])
    
#     plt.arrow(0,0,corr1,corr2)
#     plt.text(corr1,corr2, var[j])
    
# c = np.linspace(0,2*np.pi,100)
# plt.plot(np.cos(c),np.sin(c))

# plt.title("Représentaion des variables")
# plt.xlabel("Axe1")
# plt.ylabel("Axe2")
# plt.show()






#2.5 Clustering (même méthodologie que la partie 1.3)

#rVecteur de couleurs
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])




#Kmeans  
resK = []
resInertie = []
resSil = []

#Choix du meilleur K

# for k in range(2,11):   
#     kmeans = KMeans(n_clusters=k,n_init=10,init='k-means++').fit(Zp4)
#     resInertie.append(kmeans.inertia_)
#     sil= skm.silhouette_score(Zp4,kmeans.labels_)
#     resSil.append(sil)
#     print("inertie k =",k," :", kmeans.inertia_)

#Analyse score de silhouette
#plt.scatter(range(2,11),resInertie)  #k=3
#plt.scatter(range(2,11),resSil)     #k =2-3

#Meilleur représentation
#k = 3
#kmeans = KMeans(n_clusters=k,n_init=10,init='k-means++').fit(Zp4).labels_

#plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(kmeans))
# plt.title(" Kmeans silhouette")
# plt.show()




#Mélange de gaussiennes

#Choix des meilleurs paramètres
methods = ["kmeans","k-means++","random","random_from_data"]
# for method in methods :
#     list_sil = []
#     for n in range(2,15):
#         gm_labels = GaussianMixture(n_components=n,init_params = method ,random_state=0).fit_predict(Zp4)
#         if len(set(gm_labels))>1:
#             sil_test = skm.silhouette_score(Zp4,gm_labels)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#     plt.scatter(range(2,15),list_sil)

# plt.title("Gaussain mixture silhouette")
# plt.xlabel("n_components")
# plt.ylabel("silhouette")
# plt.legend(methods, loc='lower right')
# plt.show()

#Meilleur représentation 
#best : n=2 kmeans or random_from_data, n=3 kmeans, n=3 random, n=4 random

# gm = GaussianMixture(n_components=3,init_params = "random" ,random_state=0).fit_predict(Zp4)
# plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(gm))
# plt.title(" GM , n= 4, kmeans")
# plt.show()






#CAH   

#Choix des meilleurs paramètres
list_method = ["complete","ward","single","average","centroid"]

# for methode in list_method:
#     list_sil = []
#     for t in range(1,15):
#         Htest= linkage(Zp4, method = methode,optimal_ordering = True)
#         labels_test = fcluster(Htest, t ,criterion="distance")
#         if len(set(labels_test))>1:
#             sil_test = skm.silhouette_score(Zp4,labels_test)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#         print("Silhouette pour ",methode," et t = ",t,", = ",sil_test) 
#     plt.scatter([i for i in range(1,15)],list_sil) 


# plt.title("CAH silhouette")
# plt.xlabel("threshold")
# plt.ylabel("silhouette")
#plt.legend(list_method, loc='upper left')
#plt.show()


#Meilleurs représentation
#best , t=4 single, t=8 centroid,  t=10 complete (k=5), t=48 ward (k=4) 
# CAH= linkage(Zp4, method = "ward",optimal_ordering = True)
# CAH_labels = fcluster(CAH,48,criterion="distance")

# plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(CAH_labels))
# plt.title("CAH t=48, ward")
# plt.show()

#Analyse avec le Dendrogram
# H= linkage(Zp4, method = "complete",optimal_ordering = True)
# D = dendrogram(H,color_threshold = 10)  
# plt.title("Dendrogram complete t = 10")
# plt.show()






# Spectral clustering

#Choix des meilleurs paramètres
assign_labels = ["kmeans", "discretize", "cluster_qr"]  #n_inti = 10 for kmeans
# for assign_label in assign_labels :
#     list_sil = []
#     for n in range(2,15):
#         spectral_labels = SpectralClustering(n_clusters=n,assign_labels=assign_label,random_state=0).fit(Zp4).labels_
#         if len(set(spectral_labels))>1:
#             sil_test = skm.silhouette_score(Zp4,spectral_labels)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#     plt.scatter(range(2,15),list_sil) 
    
# plt.title(" Spectral silhouette")
# plt.xlabel("n_cluster")
# plt.ylabel("silhouette")
# plt.legend(assign_labels, loc='center right')
# plt.show()

#Meilleurs représentations
# best = n=2 kmeans, n=4 discretize , cluster_qr n=3

# spectral = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0).fit(Zp4).labels_
# plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(spectral))
# plt.title("Spectral results n=3, discretize")
# plt.show()








#BDSCAN  


#Choix du meilleur eps (sientific article )
# nearestN = NearestNeighbors(n_neighbors=2)
# NN = nearestN.fit(Zp4)
# d,i = NN.kneighbors(Zp4)

# distances = np.sort(d, axis=0)
# distances = distances[:,1]

# plt.title("Best eps")
# plt.plot(distances)  #eps = 1.30
# plt.show()

res_sil = []
#choix du meilleur min_samples

# for n in range(2,11):   
#    dbs = DBSCAN(eps=1.3,min_samples=n).fit(Zp4)
#    print(dbs.labels_)
#    sil= skm.silhouette_score(Zp4,dbs.labels_)
#    resSil.append(sil)
   
# plt.title("BDSCAN silhouette")
# plt.scatter(range(2,11),resSil) 
# plt.show()

#Meilleur représentation
#Best min_samples=3
# dbs_t = DBSCAN(eps=1.3,min_samples=3).fit(Zp4).labels_

# plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(dbs_t))
# plt.title("BDSCAN eps=1.3 min_samples=3")
# plt.show()







#Interpretation, cerlce de corrélation ajouté sur le nuage des individus pour le mélange des gaussiennes

#Nuage des individus après clustering
gm = GaussianMixture(n_components=3,init_params = "random" ,random_state=0).fit_predict(Zp4)
plt.scatter(Zp4[:,0],Zp4[:,1], c=vectorizer(gm))

#Cercle de corrélation
axe1 = 0
axe2 = 1
for j in range(len(pca4.components_[0])) :

    corr1 = pca4.components_[axe1,j]*np.sqrt(pca4.explained_variance_[axe1])*5
    corr2 = pca4.components_[axe2,j]*np.sqrt(pca4.explained_variance_[axe2])*5
    
    plt.arrow(0,0,corr1,corr2)
    plt.text(corr1,corr2, var[j])
    
c = np.linspace(0,2*np.pi,100)
plt.plot(np.cos(c)*5,np.sin(c)*5)

plt.xlim(-6,6)
plt.ylim(-6,6)
plt.title("Représentaion des variables et individus, k=3")
plt.xlabel("Axe1")
plt.ylabel("Axe2")
plt.show()