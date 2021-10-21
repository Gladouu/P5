import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import preprocessing, decomposition


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', 
                                 rotation=label_rotation, color="#4cb2ff", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='#4cb2ff')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            #plt.show(block=False)
            
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            
            # initialisation de la figure       
            #fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            #plt.show(block=False)
            
            
def zoom(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
            
            # initialisation de la figure       
            fig = plt.figure(figsize=(14,8))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]) :
                    if 0<=x<=4 and 0<=y<=6 : 
                        plt.text(x, y , labels[i],
                                  fontsize='12', ha='left', va='bottom')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([0,6])
            plt.ylim([0,4])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.savefig('Zoom.jpg', dpi=500, bbox_inches='tight', pad_inches=0.5)
            plt.show(block=False)
            
            
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c='#61ba86', marker='o')
    plt.xlabel("Rang de l'axe d'inertie",labelpad=20)
    plt.ylabel("Inertie (%)", labelpad=20)
    plt.title("Éboulis des valeurs propres",fontsize=20, pad=30)
    plt.xticks(np.arange(1,7)) #desable for other project
    plt.savefig('Scree Plot.jpg', dpi=500, bbox_inches='tight', pad_inches=0.5)
    plt.show(block=False)
    

def plot_dendrogram(Z, names, k):
    plt.figure(figsize=(30,10))
    plt.ylabel('Distance', fontsize=18, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=25, pad=30)
    
    # Control number of clusters in the plot + add horizontal line
    # Don't forget that parametre k is the number of clusters, remove it if is not necessary...
    ct = Z[-(k-1),2]  
    
    dendrogram(Z, labels=names, color_threshold=ct)
    
    plt.axhline(y=ct, c='white', lw=1, linestyle='dashed')
    
    plt.savefig('Dendrogram.jpg', dpi=500, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    

def plot_truncated_dendrogram(Z, method, p):
    plt.figure(figsize=(6,6))

    plt.xlabel('Sample index', labelpad=20)
    plt.ylabel('Distance', labelpad=20)
    plt.title('Hierarchical Clustering Dendrogram (truncated)', fontsize=16, pad=30)

    # show only the last p merged clusters
    dendrogram(Z, truncate_mode = method, p=p) 
    
    plt.savefig('Truncated Dendogram.jpg', dpi=500, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    
    
def heatmap_corr(i) :

    plt.figure(figsize=(15,5))

    mask = np.zeros_like(i.corr())
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(i.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')

    plt.xticks(rotation=25, ha='right')
    plt.title('Triangle de Corrélation',  fontsize=18, pad=20)

    plt.show()
