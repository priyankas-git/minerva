# main.py

import pandas as pd
import os
import sys

# CRITICAL FIX: Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all modules
from src.data_prep import load_and_merge_data, clean_data, filter_sparse_data, engineer_features
from src.hybrid_recommender import initialize_models, get_hybrid_recommendations, get_mood_recommendations, get_demand_forecast_ranking

# Configuration
PROCESSED_DATA_PATH = os.path.join('data', 'processed_data.csv')
TEST_USER_ID = 314 # A user ID known to be active
# FIX APPLIED: Changed to Book ID 3, which is confirmed to exist in the filtered data (from Demand Forecasting output)
TEST_BOOK_ID = 3 

def run_pipeline():
    """Main function to run the full data preparation and model pipeline."""
    data_dir = 'data'
    
    # --- 1. Data Preparation (Skipped if file exists) ---
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("Processed data not found. Running Data Preparation Pipeline...")
        try:
            df = load_and_merge_data(data_dir=data_dir)
            df_cleaned = clean_data(df)
            df_filtered = filter_sparse_data(df_cleaned, min_user_ratings=5, min_book_ratings=50)
            df_final = engineer_features(df_filtered)
            df_final.to_csv(PROCESSED_DATA_PATH, index=False)
            print(f"\n✅ Data preparation complete and saved.")
        except FileNotFoundError as e:
            print(f"Error: Required source files were not found. Details: {e}")
            return
    else:
        print(f"Found existing processed data at {PROCESSED_DATA_PATH}. Skipping data prep.")

    # -----------------------------------------------------------------
    # --- 2. Initialize Hybrid Models (Loads CF/CB components) ---
    # -----------------------------------------------------------------
    initialize_models()

    # -----------------------------------------------------------------
    # --- 3. Run and Display CORE Minerva Features ---
    # -----------------------------------------------------------------
    
    # A. Hybrid Recommendation (Personalization)
    hybrid_recs = get_hybrid_recommendations(user_id=TEST_USER_ID, n=10)
    print("\n==========================================================")
    print(f"HYBRID RECOMMENDATIONS for User {TEST_USER_ID} (CF + CB Blend)")
    print("==========================================================")
    if isinstance(hybrid_recs, str):
        print(hybrid_recs)
    else:
        print(hybrid_recs.to_string(index=False))


    # B. Demand Forecasting (Popularity/Quality Ranking)
    demand_recs = get_demand_forecast_ranking(n=10)
    print("\n==========================================================")
    print("DEMAND FORECASTING (Top 10 Weighted Rating/Popularity)")
    print("==========================================================")
    print(demand_recs.to_string(index=False))


    # C. Mood-Based Recommendation (Content Similarity)
    # This requires looking up the title of the test book first for context
    book_df = pd.read_csv(PROCESSED_DATA_PATH)
    # The title must be looked up in the *full* processed data before calling the mood function
    test_title = book_df[book_df['book_id'] == TEST_BOOK_ID]['title'].iloc[0]
    
    mood_recs = get_mood_recommendations(book_id=TEST_BOOK_ID, n=10)
    print("\n==========================================================")
    print(f"MOOD-BASED RECOMMENDATIONS (Similar to: {test_title.title()})")
    print("==========================================================")
    if isinstance(mood_recs, str):
        print(mood_recs)
    else:
        print(mood_recs.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()