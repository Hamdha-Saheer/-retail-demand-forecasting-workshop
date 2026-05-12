import pandas as pd
import joblib
import os
from src.feature_engineering import add_features
from src.config.config import INFERENCE_OUTPUT_PATH

def run_inference_future_predict(
    model_path: str,
    input_path: str,
    output_path: str = INFERENCE_OUTPUT_PATH
):
    # 1. Load and Clean Future Data
    df_future = pd.read_csv(input_path)
    df_future['date'] = pd.to_datetime(df_future['date'])
    df_future = df_future.dropna()

    # 2. Add Engineered Features (Must match training logic exactly)
    df_future = add_features(df_future)

    # 3. Load and Apply Label Encoders
    store_le_path = 'models/store_le.pkl'
    item_le_path = 'models/item_le.pkl'

    if os.path.exists(store_le_path) and os.path.exists(item_le_path):
        store_le = joblib.load(store_le_path)
        item_le = joblib.load(item_le_path)

        # Transform categorical IDs to numeric codes
        df_future['store_id'] = store_le.transform(df_future['store_id'])
        df_future['item_id'] = item_le.transform(df_future['item_id'])
    else:
        raise FileNotFoundError('Label encoders not found in /models. Run training first.')

    # 4. Prepare Feature Matrix (X)
    cols_to_drop = ['date']
    if 'sales_qty' in df_future.columns:
        cols_to_drop.append('sales_qty')

    X_future = df_future.drop(columns=cols_to_drop)

    # 5. Load Model and Predict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}')

    model = joblib.load(model_path)

    # Generate predictions using the loaded model
    df_future['sales_qty_pred'] = model.predict(X_future)

    # 6. Save Results to the specified output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_future.to_csv(output_path, index=False)

    print(f'Future predictions saved to {output_path}')