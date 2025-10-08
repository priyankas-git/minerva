import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
import os

PROCESSED_DATA_PATH = os.path.join('data', 'processed_data.csv')

def train_cf_model(data_path: str = PROCESSED_DATA_PATH):
    """
    Trains a Collaborative Filtering model using Sparse Matrix and SVD (Matrix Factorization).
    
    Returns:
        tuple: (Predictions DataFrame, Pivot table structure for mapping)
    """
    print("\n--- Starting Collaborative Filtering (Scipy SVD) Model Training ---")
    
    df = pd.read_csv(data_path)
    
    # CRITICAL FIX: Aggregate ratings for duplicate (user_id, book_id) pairs
    # This prevents the "Index contains duplicate entries" error during pivot.
    df_unique = df.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()
    
    # 1. Pivot the data: Create the User-Item Matrix
    R_df = df_unique.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    
    # 2. Normalize the data
    R = R_df.values
    user_ratings_mean = np.mean(R, axis=1)
    
    # Subtracting the mean rating from each user's ratings
    R_normalized = R - user_ratings_mean.reshape(-1, 1)
    R_sparse = csr_matrix(R_normalized)
    
    # 3. Perform SVD (Matrix Factorization)
    print("Performing SVD (Matrix Factorization)...")
    # Using a smaller k=20 for faster execution on smaller datasets
    U, sigma, Vt = svds(R_sparse, k=20) 
    
    # Convert sigma to a diagonal matrix
    sigma = np.diag(sigma)
    
    # 4. Reconstruct the predicted ratings matrix
    R_predicted = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    
    # 5. Convert the prediction matrix back to a DataFrame
    preds_df = pd.DataFrame(R_predicted, columns=R_df.columns, index=R_df.index)
    
    print(f"CF Model trained and predicted ratings matrix created. Shape: {preds_df.shape}")
    
    # Return the prediction matrix and the original pivot structure for lookup
    return preds_df, R_df