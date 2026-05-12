# main.py

from src.data_preprocessing import preprocess_run
from src.feature_engineering import add_features, prepare_features
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.run_inference_future_predict import run_inference_future_predict
from src.config.config import (
    MODEL_PATH,
    INFERENCE_INPUT_PATH,
    INFERENCE_OUTPUT_PATH
)
from src.utils.logger import get_logger

# Create logger
logger = get_logger("RetailForecastPipeline")


def main():

    logger.info("Starting Retail Demand Forecasting Pipeline")

    # 1️⃣ Data Preprocessing
    df = preprocess_run()
    logger.info("Data preprocessing completed")

    # 2️⃣ Feature Engineering
    df = add_features(df)
    logger.info("Feature engineering completed")

    # 3️⃣ Prepare Train/Test Split
    x_train, x_test, y_train, y_test = prepare_features(df)
    logger.info("Features prepared & train/test split done")

    # 4️⃣ Train Model
    model = train_model(x_train, y_train, model_path=MODEL_PATH)
    logger.info(f"Model saved at {MODEL_PATH}")

    # 5️⃣ Evaluate Model
    evaluate_model(model, x_test, y_test)
    logger.info("Model evaluation completed")

    # 6️⃣ Future Inference
    run_inference_future_predict(
        model_path=MODEL_PATH,
        input_path=INFERENCE_INPUT_PATH,
        output_path=INFERENCE_OUTPUT_PATH,
    )

    logger.info("Inference completed & predictions saved")
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()