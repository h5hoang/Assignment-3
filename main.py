import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(k=3, max_iter=1000, tol=0.000001, metric='euclidean')
    
    # Fit model
    kmeans.fit(og_iris) 

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    prediction = kmeans.predict(df)  
    
    # You can choose which scoring method you'd like to use here:
    final_error = kmeans.get_error()
    print("final error (inertia):", final_error)   
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, prediction, kmeans.get_centroids(), final_error)
    
    # Try different numbers of clusters
    k_values = range(1, 12)
    errors = []

    for k in k_values:
        kmeans = KMeans(k, max_iter=1000, tol=0.000001, metric='euclidean')
        kmeans.fit(og_iris)
        prediction = kmeans.predict(df)  
        final_error = kmeans.get_error()
        print("final error (inertia):", final_error)   
        plot_3d_clusters(df, prediction, kmeans.get_centroids(), final_error)
        errors.append(kmeans.get_error())
    
    # Plot the elbow plot
    plt.plot(k_values, errors)
    plt.title('Elbow Plot')
    plt.xlabel('# of Clusters, k')
    plt.ylabel('Error (inertia)')
    plt.show()
    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 
    3

    Reasoning: 
    The elbow plot's inertia/error flattens out significantly aat at k = 3, so there are 3 species of flowers total.
    """

    
if __name__ == "__main__":
    main()