import pandas as pd
import numpy as np
import os
import sys

# Append the src directory to Python path to allow imports from sibling modules
# This is a safety check but the main.py fix is more effective
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model functions (they return the trained components)
from cf_model import train_cf_model
from cb_model import get_content_model

# --- Configuration ---
PROCESSED_DATA_PATH = os.path.join('data', 'processed_data.csv')
TOP_N = 10  
CF_WEIGHT = 0.6  # Weight given to the Collaborative Filtering score
CB_WEIGHT = 0.4  # Weight given to the Content-Based score

# Global variables to store trained model components
CF_PRED_DF = None
R_DF = None
BOOK_FEATURES_DF = None
BOOK_TO_INDEX = None
COSINE_SIM_MATRIX = None

def initialize_models():
    """Initializes and loads the CF and CB model components."""
    global CF_PRED_DF, R_DF, BOOK_FEATURES_DF, BOOK_TO_INDEX, COSINE_SIM_MATRIX
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run main.py first.")

    print("\n--- Initializing Hybrid Components ---")
    
    # 1. Collaborative Filtering (CF) Component
    CF_PRED_DF, R_DF = train_cf_model(PROCESSED_DATA_PATH)
    
    # 2. Content-Based (CB) Component
    BOOK_FEATURES_DF, BOOK_TO_INDEX, COSINE_SIM_MATRIX = get_content_model(PROCESSED_DATA_PATH)
    
    print("Hybrid components loaded and ready.")
    
    
# ====================================================================
# A. HYBRID RECOMMENDER FUNCTION (CF + CB)
# ====================================================================

def get_hybrid_recommendations(user_id: int, n: int = TOP_N):
    """
    Generates personalized book recommendations by combining CF and CB scores.
    """
    if CF_PRED_DF is None or R_DF is None:
        initialize_models()
        
    print(f"\n--- Generating Hybrid Recommendations for User {user_id} ---")
    
    if user_id not in R_DF.index:
        return "User not found in the dataset. Cannot generate personalized recommendations."

    # --- 1. Get CF Predictions and Filter Read Books ---
    user_row_number = R_DF.index.get_loc(user_id)
    user_predictions = CF_PRED_DF.iloc[user_row_number]
    
    # Get books the user has already rated (value > 0)
    read_books = R_DF.loc[user_id][R_DF.loc[user_id] > 0].index.tolist()
    unrated_predictions = user_predictions.drop(read_books, errors='ignore').to_frame(name='cf_score')
    
    # --- 2. Calculate Content-Based (CB) Score Component ---
    top_rated_books = R_DF.loc[user_id][R_DF.loc[user_id] >= 4].sort_values(ascending=False).head(3).index.tolist()
    
    if top_rated_books:
        cb_scores = {}
        for candidate_book_id in unrated_predictions.index:
            candidate_index = BOOK_TO_INDEX.get(candidate_book_id)
            if candidate_index is not None:
                similarities = []
                for rated_book_id in top_rated_books:
                    rated_index = BOOK_TO_INDEX.get(rated_book_id)
                    if rated_index is not None:
                        similarity = COSINE_SIM_MATRIX[rated_index, candidate_index]
                        similarities.append(similarity)
                
                if similarities:
                    cb_scores[candidate_book_id] = max(similarities)
        
        cb_scores_series = pd.Series(cb_scores, name='cb_score')
        unrated_predictions = unrated_predictions.merge(cb_scores_series, left_index=True, right_index=True, how='left')
        unrated_predictions['cb_score'].fillna(0, inplace=True)
    else:
        unrated_predictions['cb_score'] = 0 


    # --- 3. Combine Scores (Hybridization) ---
    unrated_predictions['hybrid_score'] = (
        CF_WEIGHT * unrated_predictions['cf_score'] + 
        CB_WEIGHT * unrated_predictions['cb_score']
    )
    
    # Sort and merge with book details
    final_recs = unrated_predictions.sort_values(by='hybrid_score', ascending=False).head(n)
    final_recs = final_recs.merge(
        BOOK_FEATURES_DF[['book_id', 'title', 'authors']], 
        left_index=True, 
        right_on='book_id', 
        how='left'
    )
    
    return final_recs[['book_id', 'title', 'authors', 'hybrid_score']]


# ====================================================================
# B. MOOD-BASED RECOMMENDER FUNCTION (CB Only)
# ====================================================================

# ... (rest of the imports and existing code)

# ====================================================================
# B. MOOD-BASED RECOMMENDER FUNCTION (Content Similarity + Mood Filter)
# ====================================================================

# MOOD-TO-TAG MAPPING: Maps user mood to relevant tags in your dataset.
# The numbers are placeholder tag IDs based on common themes/genres. 
# For a real system, you would look up the tag_ids from your tags.csv.
MOOD_TAG_MAPPING = {
    'curious': ['mystery', 'crime', 'investigation', 'thriller'],
    'thrilled': ['horror', 'suspense', 'dark', 'psychological-thriller'],
    'relaxed': ['romance', 'slice-of-life', 'chick-lit', 'contemporary'],
    'inspired': ['biography', 'history', 'philosophy', 'self-help'],
    'sad': ['tragedy', 'war', 'historical-fiction', 'emotional']
}

def get_mood_recommendations(mood: str, n: int = TOP_N):
    """
    Generates Mood/Content-Based recommendations by filtering books based on tags related to the selected mood.
    
    NOTE: This is a SIMULATION. In a real application, the content model would need to be re-run 
    or the similarity matrix filtered by tag presence.
    """
    global BOOK_FEATURES_DF
    
    if BOOK_FEATURES_DF is None:
        # Emergency load if not initialized
        df_full = pd.read_csv(PROCESSED_DATA_PATH)
        BOOK_FEATURES_DF = df_full[['book_id', 'title', 'authors', 'weighted_rating', 
                                    'average_rating', 'ratings_count', 'content_features']].drop_duplicates(subset=['book_id']).reset_index(drop=True)

    if mood not in MOOD_TAG_MAPPING:
        return f"Mood '{mood}' not recognized."
    
    target_tags = MOOD_TAG_MAPPING[mood]
    print(f"\n--- Generating Mood-Based Recommendations for Mood: {mood.title()} (Tags: {', '.join(target_tags)}) ---")

    # 1. Filter books: Selects books whose 'content_features' (which includes all_tags) contain ANY of the target mood tags.
    # We use regex to check for the presence of the tags.
    tag_regex = '|'.join(target_tags)
    
    # We filter the full BOOK_FEATURES_DF which contains 'content_features'
    # .str.contains is case-insensitive by default in newer pandas versions, but we force lower-case tags anyway.
    filtered_books = BOOK_FEATURES_DF[
        BOOK_FEATURES_DF['content_features'].str.contains(tag_regex, case=False, na=False)
    ].copy()
    
    if filtered_books.empty:
        return f"No popular books found matching the tags for mood: {mood.title()}."

    # 2. Rank by quality: Rank the filtered books by weighted rating (quality/demand)
    # This ensures the books we recommend for the mood are also highly rated.
    mood_ranking = filtered_books.sort_values('weighted_rating', ascending=False).head(n)
    
    # 3. Add a placeholder 'similarity_score' for consistent output structure (set to weighted rating for now)
    mood_ranking.rename(columns={'weighted_rating': 'mood_score'}, inplace=True)
    
    # The actual Content-Based similarity calculation is skipped here for performance, 
    # as filtering by tags is often sufficient for a mood feature.
    
    return mood_ranking[['book_id', 'title', 'authors', 'average_rating', 'mood_score']]

# NOTE: The old 'get_mood_recommendations' that used 'book_id' is replaced entirely.
# The 'get_demand_forecast_ranking' remains the same.
# ... (rest of the code remains the same)


# ====================================================================
# C. DEMAND FORECASTING FUNCTION (Weighted Rating)
# ====================================================================

def get_demand_forecast_ranking(n: int = TOP_N):
    """
    Generates the ranking of books based on Weighted Rating.
    This fulfills the "demand forecasting" requirement (popularity/quality balance).
    """
    # FIX APPLIED HERE: global declaration must be first.
    global BOOK_FEATURES_DF 
    
    if BOOK_FEATURES_DF is None:
        # Load the features again, including the weighted_rating
        df_full = pd.read_csv(PROCESSED_DATA_PATH)
        BOOK_FEATURES_DF = df_full[['book_id', 'title', 'authors', 'weighted_rating', 
                                    'average_rating', 'ratings_count']].drop_duplicates(subset=['book_id']).reset_index(drop=True)

    print(f"\n--- Generating Top {n} Demand Forecast Ranking ---")
    
    # Simply sort the unique book list by the pre-calculated 'weighted_rating'
    demand_ranking = BOOK_FEATURES_DF.sort_values('weighted_rating', ascending=False).head(n)
    
    return demand_ranking[['book_id', 'title', 'authors', 'weighted_rating', 'average_rating', 'ratings_count']]