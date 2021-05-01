import numpy as np
from scipy.special import gammainc
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns 
sns.set()
import streamlit as st


# required functions
def sample_logistic(center,radius,n_samples):
    ndim = center.size
    x = np.random.logistic(0,0.7,size=(n_samples, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = radius*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_samples,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p

def get_labels(data, eps=0.3, min_samples=15):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    core_points = clustering.core_sample_indices_
    labels = clustering.labels_
    n_clusters = labels.max()
    return labels, core_points, n_clusters

def show_cores_and_clusters(data, labels,core_points,eps, min_samples):
    fig= plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(data[:,0],data[:,1], color='b')
    ax1.set_xlim(-1.5,1.5)
    ax1.set_ylim(-1.5,1.5)
    for k in core_points:
        cir=plt.Circle((data[k,0],data[k,1]), eps, color='r',fill=False)
        ax1.add_patch(cir)
    ax1.set_aspect('equal')
    ax1.set_title('CorePoints')
    ax2 = fig.add_subplot(1,2,2)
    for i, label in enumerate(labels):
        ax2.scatter(data[i,0], p[i,1], c = COLORS[label], marker='o', s=400)
    ax2.set_xlim(-1.5,1.5)
    ax2.set_ylim(-1.5,1.5)
    ax2.set_aspect('equal')
    ax2.set_title('cluster')
    return fig

#########################################################################################

st.title('DBSCAN')

#provide introduction
with st.beta_expander("See introduction"):
    st.markdown('''The Density-Based Spatial Clustering of Applications with Noise
                    (DBSCAN) is used here to cluster density-connected points (clusters)
                    and detect outliers in a spherical dataset with logistic distributed
                    two dimanesional data points.''')

    st.markdown(''' The main idea behind the DBSCAN algorithm is the ε-neighborhood,
                    which determines the density metric. The number of points j
                    within within the ε-neighborhood of i determines wether i is
                    a corner- or borderpoint, or an outlier. There are two relevant
                    hyperparameter esp, wich describes a radius, and min_samples which
                    describes the minimal number of neighbouring points required to
                    be a core point. Each cluster contains at least one core point,
                    plus the direct or indirect reachable points.''')

    st.markdown('''DBSCAN Is non-hierachical and non parametric algorithm with
                    linear complexity (without further indexing problems).''') 



st.sidebar.header('Interactive bord')

# make interactive sidebar
n_samples = st.sidebar.slider('Number of sample points', 0, 300, 250)
eps = st.sidebar.slider('Distance to determin cluster membership', 0.01, 1.00, 0.30)
min_samples = st.sidebar.slider('Minimum amount to build a cluster', 0, 30, 15)

if st.sidebar.button('Random'):
    ran= np.random.randint(0,100)
    np.random.seed(ran)
else:
    np.random.seed(42)

center = np.array([0,0])
radius = 1.2

# set and get variables
COLORS = np.array(['#C67052','#8C1E92','#6D8325','#4F51FE', '#7A989A','#849271','#C67052',
                    '#3F6F76','#69B7CE','#C1AE8D','#CF9546','#FD814E','#FD814E','#162B3D'])
p = sample_logistic(center,radius,n_samples )
labels, core_points, n_clusters = get_labels(p, eps, min_samples)
fig = show_cores_and_clusters(p, labels,core_points ,eps, min_samples)

######################################################################################
st.header('Outlier Detection')

#plot score
if len(np.unique(labels)) > 1:
    score = metrics.silhouette_score(p, labels, metric='sqeuclidean')
    st.write('Silhouette Score:',score)
else:
    st.write('Silhouette Score: Not available in the particular case')

#plot corepoints and cluster
st.pyplot(fig)

#provide theory
with st.beta_expander("See Theory"):
    st.markdown(''' 
        **ε-Neighborhood:**  
        The radius of the ε-neighborhood is specified by the distance function  
        and the metric (here Euclidian). The ε-neighborhood serves as a measure  
        for similarity and is determined by the hyperparamter esp. DBSCAN can be  
        used with any distance function. Hence, the distance function can be  
        considered as an additional hyperparameter.  
        DBSCAN is non-parametric in the sense that no shape or latent  
        generative distribution of the clusters is presupposed. Hence, DBSCAN can  
        find arbitrarily-shaped clusters and avoid the so-called single-link effect,  
        due to the min_samples parameter.  DBSCAN is non-hierachical in the sense that there  
        exists no meta-clustering for clusters. 
        
        The DBSCAN Algorithm is based on the following concepts.  

        **CorePoints:**
        A point is a corepoint if equal or more than min_samples (hyperparameter) are in
        its ε-neighborhood (directly reachable). Each cluster contains at least one corepoint.

        **BorderPoints:**
        Border points are connected components of corepoints, however, they themself have
        less than min_sample points in there ε-neighborhood.
        Border points are part of the cluster (reachable points). Border points can be edge
        points that have two corepoints from different clusters in their ε-neighborhood.
        If (and only if) edge points exists, the order of clustering dynamics starts playing
        a role (asymmetric clustering).

        **Outliers:**
        Outlieres (here darkblue) have neither min_sample points nor any corepoint in
        there ε-neighborhood. Outliers lie in low-density regions and are not reachable from
        any corepoint.

        The advantages of DBSCAN are that it does not require one to specify the number
        of clusters a priori. The hyperparameters min_samples and ε can be set
        by a domain expert, if the data is well understood, as well as if not. 
        ''')

# provide contact information
with st.beta_expander("Contact"):
    st.markdown(''' For more information contact: christoph.guettner@t-online.de''')

