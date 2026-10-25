import os
import pandas as pd

from src.customer_segmentation.logger import logger


class DataIngestion:
    def __init__(self, input_path, processed_path):
        self.input_path = input_path
        self.processed_path = processed_path

    def initiate_data_ingestion(self):

        logger.info("Starting data ingestion.")

        try:

            # ------------------------------------------------
            # 1. Load raw dataset
            # ------------------------------------------------
            df = pd.read_csv(self.input_path)

            logger.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            # ------------------------------------------------
            # 2. Check duplicate rows
            # ------------------------------------------------
            duplicate_count = df.duplicated().sum()

            logger.info(
                f"Number of duplicate rows: {duplicate_count}"
            )

            # ------------------------------------------------
            # 3. Remove duplicates
            # ------------------------------------------------
            if duplicate_count > 0:

                df = df.drop_duplicates()

                logger.info(
                    f"Duplicates removed. New shape: {df.shape}"
                )

            # ------------------------------------------------
            # 4. Create processed directory
            # ------------------------------------------------
            os.makedirs(
                os.path.dirname(self.processed_path),
                exist_ok=True
            )

            # ------------------------------------------------
            # 5. Save processed dataset
            # ------------------------------------------------
            df.to_csv(
                self.processed_path,
                index=False
            )

            logger.info(
                f"Processed dataset saved at: {self.processed_path}"
            )

            return df

        except Exception as e:

            logger.exception(
                f"Error occurred during data ingestion: {e}"
            )

            raise e