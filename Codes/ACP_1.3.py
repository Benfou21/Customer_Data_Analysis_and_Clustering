

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




#Visualisation des données
#Jain
data_jain = pd.read_csv("jain.txt","\t")
X_jain = data_jain.to_numpy()
#plt.scatter(X_jain[:,0],X_jain[:,1])

#Aggregation
data_aggregation = pd.read_csv("aggregation.txt","\t")
X_aggregation = data_aggregation.to_numpy()
#plt.scatter(X_aggregation[:,0],X_aggregation[:,1])

#Pathbased
data_pathbased = pd.read_csv("pathbased.txt","\t")
X_pathbased = data_pathbased.to_numpy()
#plt.scatter(X_pathbased[:,0],X_pathbased[:,1])

#plt.show()


#Normalisation des données
scaler = StandardScaler()

Z_jain = scaler.fit_transform(X_jain)
Z_aggregation = scaler.fit_transform(X_aggregation)
Z_pathbased = scaler.fit_transform(X_pathbased)


#vecteur de couleurs pour la visualisation 
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])



#On va tester pour chaque méthode les meilleurs paramètres pour chaque donnée





# K-means

#Meilleur stabilité pour methode kmeans++ et  n_inti = 10 
kmeans4 = KMeans(n_clusters=4,n_init=10,init='k-means++').fit(Z_jain)
kmeans42 = KMeans(n_clusters=4,n_init=10,init='k-means++').fit(Z_jain)
print("ARI  :" , adjusted_rand_score(kmeans4.labels_,kmeans42.labels_))   # =1


resK = []
resInertie = []
resSil = []
#On boucle sur le nombre k, en prenant les meilleurs paramètres 
for k in range(2,11):   
   kmeans = KMeans(n_clusters=k,n_init=10,init='k-means++').fit(Z_jain)
   resInertie.append(kmeans.inertia_)
   sil= skm.silhouette_score(Z_jain,kmeans.labels_)
   resSil.append(sil)
   print("inertie k =",k," :", kmeans.inertia_)

plt.title("Silhouette Jain kmeans")
#plt.scatter(range(2,11),resInertie)  #grah de l'inertie de la représentation selon k
plt.scatter(range(2,11),resSil) #grah du score de silhouette selon k
plt.show()


#Réalisation des clustering pour kmeans

#Pathbased : k = 4   (coude à 3-4, meilleur sil pour k = 4,5,6,7)
kmeans_pathbased = KMeans(n_clusters=4,n_init=10,init='k-means++').fit(Z_pathbased).labels_

#Aggregation : k = 5  (coude à 4-5, meilleur sil pour k=5)
kmeans_aggregation = KMeans(n_clusters=5,n_init=10,init='k-means++').fit(Z_aggregation).labels_

#Jain : k = 2  (coude à k=2-4 et meilleur sil pour k=2 )
kmeans_jain = KMeans(n_clusters=2,n_init=10,init='k-means++').fit(Z_jain).labels_


#Representation des résulats (un à un )

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer(kmeans_jain))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer(kmeans_aggregation))
#plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer(kmeans_pathbased))

#plt.title("Pathbased Kmeans results")
#plt.show()









#BDSCAN


#Choix du meilleur eps théorique
# nearestN = NearestNeighbors(n_neighbors=2)
# NN_jain = nearestN.fit(Z_pathbased)
# d,i = NN_jain.kneighbors(Z_pathbased)
# distances = np.sort(d, axis=0)
# distances = distances[:,1]
# plt.plot(distances)  #La meilleur valeur eps théorique s'obtient juste avant la croissance de la courbe
# plt.title("Eps Pathbased")
# plt.xlabel("min_samples")
# plt.ylabel("Sil")
# plt.show()

#Choix du meilleur min_samples, avec eps fixé
# for n in range(2,11):   
#    dbs = DBSCAN(eps=0.20,min_samples=n).fit(Z_pathbased)
#    print(dbs.labels_)
#    sil= skm.silhouette_score(Z_pathbased,dbs.labels_)
#    resSil.append(sil)
# plt.scatter(range(2,11),resSil) #graph du score de silhouette selon min_samples
# plt.title("Silhouette PathBased, eps=0.20")
# plt.xlabel("min_samples")
# plt.ylabel("Sil")
# plt.show()

#Réalisation des clustering pour BDSCAN

#Jain, best eps = 0.10, min_sample = 2,  eps=0.20, min_sample =4
dbs_jain = DBSCAN(eps=0.20,min_samples=3).fit(Z_jain).labels_
#Aggregation, best eps = 0.11, min_sample = 3 , au dessus de 7 pas de cluster(sil = 0.6)
dbs_aggregation = DBSCAN(eps=0.11,min_samples=3).fit(Z_aggregation).labels_
#PathBased, best eps = 0.20, min_sample = 4
dbs_pathbased = DBSCAN(eps=0.2,min_samples=4).fit(Z_pathbased).labels_


#Représentation des resusltats

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer(dbs_jain))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer(dbs_aggregation))
#plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer(dbs_pathbased))

# plt.title("Pathbased DBSCAN results")
# plt.show()
            
            
            
            
            
            
            
            

#CAH

#Représentation du Dendrogram pour une méthode et un threshold

# H= linkage(Z_jain, method = "ward",optimal_ordering = True)
# D = dendrogram(H,color_threshold = 14)  
# plt.title("Dendrogram Pathbased, ward t = 14")
# plt.show()

#Test de la meilleure méthode et du meilleur threshold

list_method = ["complete","ward","single","average","centroid"]
# for methode in list_method:
#     list_sil = []
#     for t in range(1,15):
#         Htest= linkage(Z_aggregation, method = methode,optimal_ordering = True)
#         labels_test = fcluster(Htest, t ,criterion="distance")
#         if len(set(labels_test))>1:
#             sil_test = skm.silhouette_score(Z_aggregation,labels_test)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#         print("Silhouette pour ",methode," et t = ",t,", = ",sil_test) 
#     plt.scatter([i for i in range(1,15)],list_sil)  

#Graph de tous les scores de silhouette selon les couples

# plt.title("Aggregation CAH silhouette")
# plt.xlabel("threshold")
# plt.ylabel("silhouette")
# plt.legend(list_method, loc='center right')
# plt.show()   


#Réalisation des clusterings ascendants hiérarchiques 

#Jain : methode = complete , t= 4
CAH_best_jain= linkage(Z_jain, method = "complete",optimal_ordering = True)
CAH_labels_jain = fcluster(CAH_best_jain,4,criterion="distance")
# aggregation : methode = ward, t= 14  (ou complete et t=2)   #très lent
CAH_best_aggregation= linkage(Z_aggregation, method = "ward",optimal_ordering = True)
CAH_labels_aggregation = fcluster(CAH_best_aggregation,8,criterion="distance")
# pathbased : methode =ward , t= 6
CAH_best_pathbased = linkage(Z_pathbased, method = "ward",optimal_ordering = True)
CAH_labels_pathbased = fcluster(CAH_best_pathbased,14,criterion="distance")

#Représentation des résultats

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer(CAH_labels_jain))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer(CAH_labels_aggregation))
# plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer(CAH_labels_pathbased))
# plt.title("Pathbased CAH , ward , t= 14")
# plt.show()









#Spectral Clustering

#Choix du meilleur n_cluster (k) et de la meilleure méthode assign_label

assign_labels = ["kmeans", "discretize", "cluster_qr"]  #n_inti = 10 for kmeans
# for assign_label in assign_labels :
#     list_sil = []
#     for n in range(2,15):
#         spectral_labels = SpectralClustering(n_clusters=n,assign_labels=assign_label,random_state=0).fit(Z_jain).labels_
#         if len(set(spectral_labels))>1:
#             sil_test = skm.silhouette_score(Z_jain,spectral_labels)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#     plt.scatter(range(2,15),list_sil) 

#Graph de tous les scores de silhouette selon les couples

# plt.title("Jain Silhouette Spectral")
# plt.xlabel("gamma")
# plt.ylabel("silhouette")

# plt.show()

# Réalisation des clusterings spéctrales 

#Jain : cluster_qr, n = 2
spectral_labels_jain = SpectralClustering(n_clusters=2,assign_labels="cluster_qr",random_state=0).fit(Z_jain).labels_
#Aggregation : cluster_qr, n = 5 #très lent
spectral_labels_aggregation = SpectralClustering(n_clusters=5,assign_labels="cluster_qr",random_state=0).fit(Z_aggregation).labels_
#PathBased : discretize , n = 8
spectral_labels_pathbased = SpectralClustering(n_clusters=8,assign_labels="discretize",random_state=0).fit(Z_pathbased).labels_


#Représentation des résultats

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer(spectral_labels_jain))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer(spectral_labels_aggregation))
#plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer(spectral_labels_pathbased))

# plt.title("Aggregation spectral results ")
# plt.show()









#Gaussian mixture

#Choix de la meilleure méthode init_params et du meilleur n_components (k)

methods = ["kmeans","k-means++","random","random_from_data"]
# for method in methods :
#     list_sil = []
#     for n in range(2,15):
#         gm_labels = GaussianMixture(n_components=n,init_params = method ,random_state=0).fit_predict(Z_aggregation)
#         if len(set(gm_labels))>1:
#             sil_test = skm.silhouette_score(Z_aggregation,gm_labels)
#             list_sil.append(sil_test)
#         else :
#             list_sil.append(0)
#     plt.scatter(range(2,15),list_sil)

#Graph de tous les scores de silhouette selon les couples

# plt.title("Aggregation Gaussain mixture")
# plt.xlabel("n_components")
# plt.ylabel("silhouette")
# plt.legend(methods, loc='lower right')
# plt.show()

#Réalisation des clusterings par mélange de gaussiennes

#Jain : n_components = 2, methods = random_from_data
gm_labels_jain = GaussianMixture(n_components=2,init_params = "random_from_data" ,random_state=0).fit_predict(Z_jain)
#Aggregation : n_components = 5, methods = random_from_data
gm_labels_aggregation = GaussianMixture(n_components=5,init_params = "random_from_data" ,random_state=0).fit_predict(Z_aggregation)
#Pathbased : n_components = 6, methods = kmeans
gm_labels_pathbased = GaussianMixture(n_components=6,init_params = "kmeans" ,random_state=0).fit_predict(Z_pathbased)

#Représentation des résultats

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer(gm_labels_jain))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer(gm_labels_aggregation))
#plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer(gm_labels_pathbased))
# plt.title("Jain GM results")
# plt.show()






#Test quantitatif

#Affichage des classes réelles pour chaque jeu de donnée

#plt.scatter(Z_jain[:,0],Z_jain[:,1], c=vectorizer( [int(y) for y in X_jain[:,2] ]))
#plt.scatter(Z_aggregation[:,0],Z_aggregation[:,1], c=vectorizer( [int(y) for y in X_aggregation[:,2] ]))
# plt.scatter(Z_pathbased[:,0],Z_pathbased[:,1], c=vectorizer( [int(y) for y in X_pathbased[:,2] ]))
# plt.title("Real clustering Pathbased")
# plt.show()





#Test qualitatif, calcul de l'indice ARI
#On va pour chaque résultat obtenu précédement calculer l'indice ARI avec les classes réelles 

#Résultats pour Jain
jain_labels = [gm_labels_jain,spectral_labels_jain,CAH_labels_jain,dbs_jain,kmeans_jain]
#Résultats pour Aggregation 
aggregation_labels =  [gm_labels_aggregation,spectral_labels_aggregation,CAH_labels_aggregation,dbs_aggregation,kmeans_aggregation]
#Résultats pour Pathbased
pathbased_lables = [gm_labels_pathbased,spectral_labels_pathbased,CAH_labels_pathbased,dbs_pathbased,kmeans_pathbased]

#Liste de listes des résultats pour chaque jeu de données
groups_labels=[jain_labels,aggregation_labels,pathbased_lables]
#Liste des partionnement réels
data_classes = [X_jain[:,2],X_aggregation[:,2],X_pathbased[:,2]]

names = ["Jain","Aggregation","pathbased"]

#On boucle sur la liste de tous les résulats en même temps que la liste des vraies classes
for i in range(3) :
    print(names[i])
    for label in groups_labels[i] :
        print("ARI :")
        print(adjusted_rand_score( data_classes[i],label)) # Calcul de l'indice ARI


