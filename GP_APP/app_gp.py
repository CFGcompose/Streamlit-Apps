
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns 
sns.set()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, Matern, RationalQuadratic,
                                            ExpSineSquared, DotProduct,
                                            ConstantKernel, Exponentiation)

import streamlit as st

def get_gp_output(i):
    Scores, Y_Means, S_Deviation, Y_Samples = [], [], [], []
    gp = GaussianProcessRegressor(kernel=kernels[i])
    gp.fit(data[0].values.reshape(-1,1), data[1].values)
    y_samples = gp.sample_y(X_[:, np.newaxis], 30)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    score = gp.score(test[0].values.reshape(-1,1), test[1].values)
    Scores.append(score)
    Y_Means.append(y_mean)
    S_Deviation.append(y_std)
    Y_Samples.append(y_samples)
    return Scores[0], Y_Means[0], S_Deviation[0], Y_Samples[0]

def plot_posterior_sample():
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(15,10))
    ax1.plot(X_,latent_function,lw=2.0,alpha=0.8,color='k', label='True Latent Function', zorder=9)
    ax1.plot(X_,Y_Means, color='#8A3230', label='Mean Function', zorder=9)
    ax1.scatter(data[0], data[1], color='k', label='Observed Data', zorder=9)
    ax1.plot(X_, Y_Samples, lw=1, alpha=0.8)
    ax1.legend()
    ax2.plot(X_,latent_function,lw=2.0,alpha=0.8,color='k', label='True Latent Function', zorder=9)
    ax2.plot(X_,Y_Means, color='#8A3230', label='Mean Function', zorder=9)
    ax2.scatter(data[0], data[1], color='k', label='Observed Data', zorder=9)
    ax2.fill_between(X_, Y_Means - S_Deviation, Y_Means + S_Deviation, alpha=0.2, color='r')
    ax2.legend()
    return fig


##############################################################################

st.title('Gaussian Process Regression')

with st.beta_expander("See introduction"):
    st.markdown('''Here, three different kernels(RBF, Rational Quadradtic, Matern)  
    are used to fit a Gaussian Process Model. The respective scores of the GP  
    are compared afterwards. Thereby, the training sample of 15 data points   
    has a bias to avoid heights and lows in the observed data.''')  

    st.markdown('''  
    In general, a Gaussian Process is complete  
    described by its mean function''')  
    
    st.latex(r'''
    \mu(x) = \boldsymbol{\Sigma}_{xy}  \boldsymbol{\Sigma}_{xx}^{-1} x''')

    st.markdown(''' 
    and its covariance function,''')  
    st.latex(r'''
    K(x,y) = \boldsymbol{\Sigma}_{yy} - \boldsymbol{\Sigma}_{xy}  \boldsymbol{\Sigma}_{xx}^{-1} \boldsymbol{\Sigma}_{yx} ''') 
    
    st.markdown('''
    which determine what and how much information about the unobserved  
    values $f(y)$  can be derived from information about known values $f(x)$.''')

kernel_names=['RBF Kernel','RationalQuadratic Kernel', 'Matern Kernel' ]
kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
        1.0 * RationalQuadratic(length_scale=1.0, alpha=0.6),
        1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]
true_kernel =  [1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)]

 

gp = GaussianProcessRegressor(kernel=true_kernel[0])

X_ = np.linspace(0, 10, 550)
#gp.predict(X)
latent_function = gp.sample_y(X_[:, np.newaxis], 1)[:,0]

dict = {'X':X_, 'Latent Function':latent_function}
df=pd.DataFrame(dict)

selected = st.sidebar.selectbox(
'Select Kernel', kernel_names)
#biased sampeling


#test_sampling
df['test_data'] = df['Latent Function'].sample(25)
test_X = df['X'][df['test_data'].notna()]
test_data = df['test_data'][df['test_data'].notna()]
test = (test_X,test_data)

observation_bias= ['Unbiased', 'Bias to the mean', 'Bias to the extreme']
selected_bias = st.sidebar.selectbox(
'Select Observation Bias', observation_bias)
if selected_bias=='Unbiased':

    if st.sidebar.button('Random'):
        ran= np.random.randint(0,100)
        np.random.seed(ran)
    else:
        np.random.seed(42)

    df['observed_data'] = df['Latent Function'].sample(16)

elif selected_bias=='Bias to the mean':
    mean = df['Latent Function'].mean()

    bias = st.sidebar.slider('How biased are your observations', 0.0, 2.9, 0.2)
    std = (3-bias) * df['Latent Function'].std()
    with st.sidebar.beta_expander("See explanation"):
        st.markdown("""
            **Minimum bias**: The observations are quasi randomly distributed 
            over the whole range of the latent function (3 std from the mean).        
            **Maximum bias**: All 15 observation are very close 
            to the y-mean of the latent function.
            
        """)

    if st.sidebar.button('Random'):
        ran= np.random.randint(0,100)
        np.random.seed(ran)
    else:
        np.random.seed(42)

    df['observed_data'] = df['Latent Function'][df['Latent Function'].between((mean-std),(mean+std))].sample(16)

elif selected_bias=='Bias to the extreme':
    maxim = df['Latent Function'].max()
    minim = df['Latent Function'].min()

    bias = st.sidebar.slider('How biased are your observations', 0.0, 2.9, 0.2)
    std = (3-bias) * df['Latent Function'].std()
    with st.sidebar.beta_expander("See explanation"):
        st.markdown("""
            **Minimum bias**: The observations are quasi randomly distributed 
            over the whole range of the latent function (3 std from each extreme).        
            **Maximum bias**: All 15 observation are close 
            to the maximum or minimum of the latent function.
            
        """)

    if st.sidebar.button('Random'):
        ran= np.random.randint(0,100)
        np.random.seed(ran)
    else:
        np.random.seed(42)

    df['observed_data'] = df['Latent Function'][df['Latent Function'].between((maxim-std), maxim)].sample(8).append(df['Latent Function'][df['Latent Function'].between(minim,(minim+std))].sample(8))

observed_X = df['X'][df['observed_data'].notna()]
observed_data = df['observed_data'][df['observed_data'].notna()]

data=(observed_X, observed_data)


if selected == kernel_names[0]:
    st.header(kernel_names[0])
    Scores, Y_Means, S_Deviation, Y_Samples = get_gp_output(0)
    st.write('Score',Scores)
    fig = plot_posterior_sample()
    st.pyplot(fig)
    with st.beta_expander("See kernel function"):
        st.latex(r"""
           k(x_i, x_j) = \exp\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)
        """)
elif selected == kernel_names[1]:
    st.header(kernel_names[1])
    Scores, Y_Means, S_Deviation, Y_Samples = get_gp_output(1)
    st.write('Score',Scores)
    fig = plot_posterior_sample()
    st.pyplot(fig)
    with st.beta_expander("See kernel function"):
        st.latex(r"""
           
           k(x_i, x_j) = \left(
            1 + \frac{d(x_i, x_j)^2 }{ 2\alpha  l^2}\right)^{-\alpha} 
        """)
        
elif selected == kernel_names[2]:
    st.header(kernel_names[2])
    Scores, Y_Means, S_Deviation, Y_Samples = get_gp_output(2)
    st.write('Score',Scores)
    fig = plot_posterior_sample()
    st.pyplot(fig)
    with st.beta_expander("See kernel function"):
        st.latex(r"""
            k(x_i, x_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(
            \frac{\sqrt{2\nu}}{l} d(x_i , x_j )
            \Bigg)^\nu K_\nu\Bigg(
            \frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)
        """)
        
    # provide contact information
with st.beta_expander("Contact"):
    st.markdown(''' For more information contact: christoph.guettner@t-online.de''')