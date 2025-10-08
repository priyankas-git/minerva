import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

PROCESSED_DATA_PATH = os.path.join('data', 'processed_data.csv')

def get_content_model(data_path: str = PROCESSED_DATA_PATH):
    """
    Trains a Content-Based model using TF-IDF on combined content features (title, authors, tags).
    
    Returns:
        tuple: (Book features DataFrame, book_id to index map, Cosine Similarity Matrix)
    """
    print("\n--- Starting Content-Based Model (TF-IDF) Preparation ---")
    
    df = pd.read_csv(data_path)
    
    # 1. Isolate unique books and their features
    # FIX APPLIED: Added 'weighted_rating' to the list of features to ensure Demand Forecasting works later.
    book_features = df[['book_id', 'title', 'authors', 'content_features', 
                        'average_rating', 'ratings_count', 'weighted_rating']].drop_duplicates(subset=['book_id']).reset_index(drop=True)
    
    # 2. Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(min_df=5, max_features=None, stop_words='english')
    
    # 3. Fit and Transform the Content Features
    print("Fitting TF-IDF Vectorizer on book content...")
    tfidf_matrix = tfidf.fit_transform(book_features['content_features'].fillna('')) 
    
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape} (Books x Features)")

    # 4. Compute Cosine Similarity
    print("Calculating Cosine Similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print("Content-Based similarity matrix computed successfully.")
    
    # 5. Create mapping
    book_to_index = {
        book_id: i for i, book_id in enumerate(book_features['book_id'])
    }
    
    return book_features, book_to_index, cosine_sim