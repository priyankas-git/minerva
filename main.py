# main.py

import pandas as pd
import os
import sys
from flask import Flask, jsonify

# Add 'src' to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.data_prep import load_and_merge_data, clean_data, filter_sparse_data, engineer_features
from src.hybrid_recommender import initialize_models, get_hybrid_recommendations, get_mood_recommendations, get_demand_forecast_ranking

# Configuration
PROCESSED_DATA_PATH = os.path.join('data', 'processed_data.csv')
TEST_USER_ID = 314
TEST_BOOK_ID = 3

# Flask app
app = Flask(__name__)

def run_pipeline():
    """Main pipeline logic (unchanged)."""
    data_dir = 'data'
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        try:
            df = load_and_merge_data(data_dir=data_dir)
            df_cleaned = clean_data(df)
            df_filtered = filter_sparse_data(df_cleaned, min_user_ratings=5, min_book_ratings=50)
            df_final = engineer_features(df_filtered)
            df_final.to_csv(PROCESSED_DATA_PATH, index=False)
        except FileNotFoundError as e:
            return {"error": f"Source files not found: {e}"}

    initialize_models()

    hybrid_recs = get_hybrid_recommendations(user_id=TEST_USER_ID, n=10)
    demand_recs = get_demand_forecast_ranking(n=10)
    book_df = pd.read_csv(PROCESSED_DATA_PATH)
    test_title = book_df[book_df['book_id'] == TEST_BOOK_ID]['title'].iloc[0]
    mood_recs = get_mood_recommendations(book_id=TEST_BOOK_ID, n=10)

    # Convert DataFrames to dicts for JSON output
    return {
        "hybrid_recommendations": hybrid_recs.to_dict(orient="records") if not isinstance(hybrid_recs, str) else hybrid_recs,
        "demand_forecasting": demand_recs.to_dict(orient="records"),
        "mood_based_recommendations": mood_recs.to_dict(orient="records") if not isinstance(mood_recs, str) else mood_recs,
        "test_book_title": test_title
    }

# Flask route to trigger the pipeline
@app.route('/')
def home():
    result = run_pipeline()
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
