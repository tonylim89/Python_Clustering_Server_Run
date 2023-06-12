import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def save_plot(fig, filename):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig('plots/' + filename)
    plt.close(fig)

'''
Display histograms.
'''
def plot_histograms(df):
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    # age distribution
    sns.histplot(
        data=df['Age'], 
        color='coral', 
        kde='True',
        ax=ax[0,0]
    )

    # income distribution
    sns.histplot(
        data=df['Income'], 
        color='cornflowerblue', 
        kde='True',
        ax=ax[0,1]
    )

    # years employed distribution
    sns.histplot(
        data=df['Years Employed'], 
        color='violet', 
        kde='True',
        ax=ax[1,0]
    )

    # debt income ratio distribution
    sns.histplot(
        data=df['DebtIncomeRatio'], 
        color='limegreen', 
        kde='True',
        ax=ax[1,1]
    )

    ax[0,0].set_title('Age Distribution')         
    ax[0,1].set_title('Income Distribution')          
    ax[1,0].set_title('Years Employed Distribution')
    ax[1,1].set_title('Debt Distribution')

    fig.tight_layout(pad=3.0)
    save_plot(fig, 'histograms.png')


'''
Plot the number of Defaults and Non-Defaults.
'''
def plot_defaults(df):
    fig, ax = plt.subplots()
    sns.countplot(x=df['Defaulted'], 
                palette = ['coral', 'deepskyblue'], 
                edgecolor = 'black')
    plt.title('Credit card default cases')
    ax.set_xticklabels(['Non-default', 'Default'])
    plt.xlabel('')
    plt.ylabel('Number of People')
    save_plot(fig, 'defaults.png')


'''
Perform Default / Non-Default classification for missing entries
'''
def knn(df):
    # get rows with values (either 0 or 1) in Defaulted column
    df_notna = df[df.Defaulted.notnull()]

    # use all columns except the Defaulted column for training
    X_train = df_notna.drop(['Defaulted'], axis=1).values

    # the Defaulted column are used as ground-truths in our training
    y_train = df_notna['Defaulted'].values

    knn = KNeighborsClassifier(n_neighbors = 7) 

    # train the model
    knn.fit(X_train, y_train)    

    # get rows with missing values in Defaulted column
    df_isnull = df[df.Defaulted.isnull()]
    X_test = df_isnull.drop(['Defaulted'], axis=1).values

    # predicting if those missing entries should be 
    # Default or Non-Default
    default_predicts = knn.predict(X_test)

    return default_predicts


'''
Replace Default/Non-Default predictions into missing entries
'''
def replace_missing_defaults(df, default_predicts):
    df_isnull = df[df.Defaulted.isnull()]
    
    for i in range(len(default_predicts)):
        df.loc[df_isnull.index[i], 'Defaulted'] = default_predicts[i]

    return df


'''
Performs K-Means clustering
'''
def kmeans(X, n_clusters):
    # apply kmeans to the dataset
    algo = KMeans(n_clusters=n_clusters, random_state=10)
    
    # gives us cluster-assignment for each row
    clusters = algo.fit_predict(X)

    return clusters


'''
Plot cluster-assignments in 3D.
'''
def plot3d_cluster_assignments(X, clusters):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in np.unique(clusters):
        ax.scatter(xs=X[clusters==i, 0], 
            ys=X[clusters==i, 3],
            zs=X[clusters==i, -1],
            label='Cluster ' + str(i+1))
            
    ax.set_xlabel('Age')     
    ax.set_ylabel('Income')      
    ax.set_zlabel('DebtIncomeRatio')

    ax.set_title('3D View of Clustering Results')
    plt.legend()
    save_plot(fig, '3D_clusters.png')


'''
Main Program
'''

# read dataset into a Pandas DataFrame
df = pd.read_csv('cust_seg.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# remove running-numbers as they are not useful as 
# they don't add any value in terms of differentiation
# among the rows
df.drop(['Customer Id'], axis=1, inplace=True)

# plot 4 histograms
plot_histograms(df)

# show count-plot of Default/Non-Default
plot_defaults(df)

# performs Default/Non-Default predictions for
# missing entries
default_predicts = knn(df)

# replace missing entries with the predictions
df = replace_missing_defaults(df, default_predicts)

# perform k-means clustering
clusters = kmeans(X=df.values, n_clusters=4)

# 3d plot of our clustering results
plot3d_cluster_assignments(X=df.values, clusters=clusters)
