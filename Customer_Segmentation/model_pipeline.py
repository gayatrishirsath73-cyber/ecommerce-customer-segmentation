import os

from src.customer_segmentation.logger import logger
from src.customer_segmentation.data_ingestion import DataIngestion
from src.customer_segmentation.data_eda import DataEDA
from src.customer_segmentation.model_trainer import ModelTrainer


# ============================================================
# PATH CONFIGURATION
# ============================================================

RAW_DATA_PATH = (
    "artifacts/data/raw/"
    "ecommerce_customer_segmentation_dataset.csv"
)

PROCESSED_DATA_PATH = (
    "artifacts/data/processed/"
    "processed_customer_data.csv"
)

VISUAL_PATH = "artifacts/visuals"

MODEL_PATH = "artifacts/models"


# ============================================================
# CLUSTERING FEATURES
# ============================================================

CLUSTERING_FEATURES = [

    "Annual_Income",

    "Customer_Tenure_Months",

    "Total_Orders",

    "Total_Spending",

    "Average_Order_Value",

    "Website_Visits_Per_Month",

    "Mobile_App_Usage_Rate",

    "Discount_Usage_Rate",

    "Cart_Abandonment_Rate",

    "Return_Rate",

    "Average_Days_Between_Orders",

    "Product_Categories_Purchased",

    "Customer_Support_Calls",

    "Loyalty_Points",

    "Last_Purchase_Days_Ago",

    "Marketing_Response_Rate"
]


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    try:

        logger.info(
            "=================================================="
        )

        logger.info(
            "E-Commerce Customer Segmentation Pipeline Started"
        )

        logger.info(
            "=================================================="
        )

        # ----------------------------------------------------
        # STEP 1: DATA INGESTION
        # ----------------------------------------------------

        logger.info("STEP 1: Starting data ingestion.")

        data_ingestion = DataIngestion(
            input_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH
        )

        df = data_ingestion.initiate_data_ingestion()

        logger.info(
            f"Data ingestion completed. Shape: {df.shape}"
        )


        # ----------------------------------------------------
        # STEP 2: EDA
        # ----------------------------------------------------

        logger.info("STEP 2: Starting EDA.")

        eda = DataEDA(
            visual_path=VISUAL_PATH
        )

        eda.dataset_overview(df)

        eda.missing_value_analysis(df)

        numerical_columns, categorical_columns = (
            eda.identify_columns(df)
        )

        eda.numerical_summary(
            df,
            numerical_columns
        )

        eda.correlation_analysis(
            df,
            numerical_columns
        )

        eda.univariate_analysis(
            df,
            numerical_columns
        )

        eda.boxplot_analysis(
            df,
            numerical_columns
        )

        eda.categorical_analysis(
            df,
            categorical_columns
        )

        logger.info(
            "EDA completed successfully."
        )


        # ----------------------------------------------------
        # STEP 3: MODEL TRAINER
        # ----------------------------------------------------

        logger.info(
            "STEP 3: Preparing clustering model."
        )

        model_trainer = ModelTrainer(
            model_path=MODEL_PATH,
            visual_path=VISUAL_PATH
        )


        # ----------------------------------------------------
        # STEP 4: FEATURE PREPARATION
        # ----------------------------------------------------

        logger.info(
            "STEP 4: Preparing clustering features."
        )

        (
            X_scaled,
            imputer,
            scaler
        ) = model_trainer.prepare_features(
            df,
            CLUSTERING_FEATURES
        )


        # ----------------------------------------------------
        # STEP 5: ELBOW ANALYSIS
        # ----------------------------------------------------

        logger.info(
            "STEP 5: Running elbow analysis."
        )

        model_trainer.elbow_analysis(
            X_scaled,
            k_range=range(2, 11)
        )


        # ----------------------------------------------------
        # STEP 6: SILHOUETTE ANALYSIS
        # ----------------------------------------------------

        logger.info(
            "STEP 6: Running silhouette analysis."
        )

        silhouette_scores = (
            model_trainer.silhouette_analysis(
                X_scaled,
                k_range=range(2, 11)
            )
        )

        print("\nSilhouette Scores:")

        for k, score in silhouette_scores.items():

            print(
                f"K = {k} --> "
                f"Silhouette Score = {score:.4f}"
            )


        # ----------------------------------------------------
        # STEP 7: SELECT OPTIMAL K
        # ----------------------------------------------------

        optimal_k = 7

        logger.info(
            f"Optimal K : {optimal_k}"
        )

        print(
            f"\nOptimal K: "
            f"{optimal_k}"
        )


        # ----------------------------------------------------
        # STEP 8: HIERARCHICAL CLUSTERING
        # ----------------------------------------------------

        logger.info(
            "STEP 8: Running hierarchical clustering."
        )

        model_trainer.hierarchical_clustering(
            X_scaled
        )


        # ----------------------------------------------------
        # STEP 9: DBSCAN
        # ----------------------------------------------------

        logger.info(
            "STEP 9: Running DBSCAN."
        )

        dbscan_labels = (
            model_trainer.dbscan_analysis(
                X_scaled
            )
        )

        print("\nDBSCAN Cluster Distribution:")

        print(
            __import__("pandas")
            .Series(dbscan_labels)
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # STEP 10: FINAL K-MEANS
        # ----------------------------------------------------

        logger.info(
            "STEP 10: Training final K-Means model."
        )

        (
            final_model,
            cluster_labels
        ) = model_trainer.train_final_model(
            X_scaled,
            optimal_k
        )


        # ----------------------------------------------------
        # STEP 11: SAVE PREPROCESSING OBJECTS
        # ----------------------------------------------------

        logger.info(
            "STEP 11: Saving preprocessing objects."
        )

        model_trainer.save_preprocessor(
            imputer,
            scaler
        )


        # ----------------------------------------------------
        # STEP 12: CLUSTER PROFILE
        # ----------------------------------------------------

        logger.info(
            "STEP 12: Creating cluster profiles."
        )

        (
            clustered_df,
            cluster_profile
        ) = model_trainer.create_cluster_profile(
            df,
            cluster_labels,
            CLUSTERING_FEATURES
        )


        # ----------------------------------------------------
        # STEP 13: SAVE FINAL SEGMENTED DATA
        # ----------------------------------------------------

        segmented_path = (
            "artifacts/data/processed/"
            "segmented_customer_data.csv"
        )

        clustered_df.to_csv(
            segmented_path,
            index=False
        )

        logger.info(
            f"Segmented customer data saved at "
            f"{segmented_path}"
        )


        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print("\n==============================================")
        print("CUSTOMER SEGMENTATION COMPLETED")
        print("==============================================")

        print(
            f"\nFinal Number of Clusters: {optimal_k}"
        )

        print(
            "\nCluster Distribution:"
        )

        print(
            clustered_df[
                "Cluster"
            ].value_counts().sort_index()
        )

        print(
            "\nCluster Profile:"
        )

        print(
            cluster_profile
        )

        logger.info(
            "E-Commerce Customer Segmentation Pipeline "
            "completed successfully."
        )


    except Exception as e:

        logger.exception(
            f"Pipeline failed: {e}"
        )

        raise e


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()