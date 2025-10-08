import pandas as pd
import numpy as np
import os

# ========================================================================
# FUNCTION 0: LOAD AND MERGE ALL FILES
# ========================================================================
def load_and_merge_data(data_dir: str = 'data') -> pd.DataFrame:
    """Loads and merges the four core CSV files into a single DataFrame."""
    print("--- Starting Data Load and Merge ---")
    
    # 1. Load Core Files
    books = pd.read_csv(os.path.join(data_dir, 'books.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    book_tags = pd.read_csv(os.path.join(data_dir, 'book_tags.csv'))
    tags = pd.read_csv(os.path.join(data_dir, 'tags.csv'))
    
    # 2. Merge Ratings with Book Metadata
    # We join ratings (book_id) to books (book_id) - This creates the base table.
    base_df = ratings.merge(
        books[[
            'book_id', 'best_book_id', 'title', 'authors', 
            'average_rating', 'ratings_count', 'work_text_reviews_count'
        ]], 
        on='book_id', 
        how='left'
    )
    print(f"Merged Ratings and Books. Shape: {base_df.shape}")

    # 3. Aggregate Tags for Content-Based Filtering
    # The tags are linked using books.best_book_id -> book_tags.goodreads_book_id
    
    # Map tag IDs to tag names
    book_tags_with_names = book_tags.merge(tags, on='tag_id', how='left')
    
    # Group by book and combine all tags into a single string for NLP
    book_tag_list = (
        book_tags_with_names
        .groupby('goodreads_book_id')['tag_name']
        .apply(lambda x: " ".join(x.astype(str).str.lower()))
        .reset_index(name='all_tags')
    )
    print(f"Prepared Content Tag features. Shape: {book_tag_list.shape}")
    
    # 4. Final Merge: Add Tags to the Base DataFrame
    # Note: best_book_id (from books) is the key for book_tags.goodreads_book_id
    final_df = base_df.merge(
        book_tag_list, 
        left_on='best_book_id', 
        right_on='goodreads_book_id', 
        how='left'
    )
    
    # Drop redundant columns
    final_df.drop(columns=['best_book_id', 'goodreads_book_id'], inplace=True, errors='ignore')
    
    print(f"Final merged dataset created. Shape: {final_df.shape}")
    return final_df

# ========================================================================
# FUNCTION 1: CLEANING AND TYPE CONVERSION (Refined for new columns)
# ========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs initial data cleaning and type conversion."""
    print("Starting data cleaning...")
    
    # A. Drop Duplicates and Handle Core Missing Data
    df.drop_duplicates(inplace=True)
    # The columns we care about are now guaranteed to be present from the merge
    df.dropna(subset=['user_id', 'book_id', 'rating', 'title'], inplace=True) 
    
    # B & C. Convert Data Types
    # The columns in the new dataset are consistently named: user_id, book_id, rating
    df['user_id'] = df['user_id'].astype('Int64')
    df['book_id'] = df['book_id'].astype('Int64')
    df['rating'] = df['rating'].astype(float)
    df['average_rating'] = df['average_rating'].astype(float)
    df['ratings_count'] = df['ratings_count'].astype('Int64')
    
    # D. Clean Text for Content-Based Model
    df['authors'].fillna('Unknown', inplace=True)
    df['all_tags'].fillna('', inplace=True) # Fill missing tags
    
    # Convert text to lowercase
    df['title'] = df['title'].str.lower()
    df['authors'] = df['authors'].str.lower()
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

# ========================================================================
# FUNCTION 2: FILTERING FOR COLLABORATIVE FILTERING (Same Logic, Different Data)
# ========================================================================
def filter_sparse_data(df: pd.DataFrame, min_user_ratings: int = 5, min_book_ratings: int = 50) -> pd.DataFrame:
    """Filters out users and books with insufficient activity (key step for CF)."""
    print("Starting sparsity filtering...")
    
    # Filter Inactive Users
    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    df_filtered = df[df['user_id'].isin(active_users)].copy()

    # Filter Unpopular Books
    book_counts = df_filtered['book_id'].value_counts()
    popular_books = book_counts[book_counts >= min_book_ratings].index
    df_final = df_filtered[df_filtered['book_id'].isin(popular_books)].copy()
    
    print(f"Sparsity filtering complete. Final shape: {df_final.shape}")
    return df_final

# ========================================================================
# FUNCTION 3: FEATURE ENGINEERING (NLP Content Features and Demand Forecasting)
# ========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Content Features and Weighted Rating feature."""
    print("Starting feature engineering...")
    
    # A. Combine Content Features for NLP (Content-Based and Mood-Based Recs)
    # Note: We don't have 'description' in this dataset, so we use title, authors, and tags.
    df['content_features'] = (
        df['title'] + ' ' + 
        df['authors'] + ' ' + 
        df['all_tags']
    ) 

    # B. Calculate Weighted Rating (For Demand Forecasting / Top N Rank)
    
    # C = The mean rating across all books (using the pre-calculated 'average_rating' from books.csv)
    C = df['average_rating'].mean()

    # m = Minimum number of votes required (90th percentile of total ratings_count)
    # We use the total 'ratings_count' from the books file
    m = df['ratings_count'].quantile(0.90) 

    def calculate_weighted_rating(row, m_val, C_val):
        v = row['ratings_count'] # actual count of ratings for the book
        R = row['average_rating'] # book's average rating
        
        # Bayesian Average formula: (v / (v + m) * R) + (m / (v + m) * C)
        return (v / (v + m_val) * R) + (m_val / (v + m_val) * C_val)

    # Apply the formula ONLY on the unique book records (to avoid re-calculating for every user rating)
    unique_books = df[['book_id', 'average_rating', 'ratings_count']].drop_duplicates().copy()
    
    unique_books['weighted_rating'] = unique_books.apply(
        lambda row: calculate_weighted_rating(row, m, C), axis=1
    )
    
    # Merge the new feature back into the main DataFrame
    df = df.merge(
        unique_books[['book_id', 'weighted_rating']], 
        on='book_id', 
        how='left'
    )
    
    print("Feature engineering complete.")
    return df