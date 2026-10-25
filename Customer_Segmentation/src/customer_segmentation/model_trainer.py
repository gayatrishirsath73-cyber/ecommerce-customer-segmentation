import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from src.customer_segmentation.logger import logger


class ModelTrainer:

    def __init__(
        self,
        model_path,
        visual_path
    ):

        self.model_path = model_path
        self.visual_path = visual_path

        os.makedirs(
            self.model_path,
            exist_ok=True
        )

        os.makedirs(
            self.visual_path,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Prepare Features
    # ---------------------------------------------------------

    def prepare_features(
        self,
        df,
        clustering_features
    ):

        logger.info(
            "Preparing clustering features."
        )

        X = df[
            clustering_features
        ].copy()

        # -----------------------------------------------------
        # Missing Value Imputation
        # -----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )

        X_imputed = imputer.fit_transform(
            X
        )

        # -----------------------------------------------------
        # Feature Scaling
        # -----------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X_imputed
        )

        logger.info(
            "Feature imputation and scaling completed."
        )

        return (
            X_scaled,
            imputer,
            scaler
        )

    # ---------------------------------------------------------
    # Elbow Method
    # ---------------------------------------------------------

    def elbow_analysis(
        self,
        X_scaled,
        k_range=range(2, 11)
    ):

        logger.info(
            "Running K-Means elbow analysis."
        )

        inertia_values = []

        for k in k_range:

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            model.fit(
                X_scaled
            )

            inertia_values.append(
                model.inertia_
            )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            list(k_range),
            inertia_values,
            marker="o"
        )

        plt.xlabel(
            "Number of Clusters (K)"
        )

        plt.ylabel(
            "Inertia"
        )

        plt.title(
            "Elbow Method"
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "elbow_plot.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Elbow plot saved at {output_path}"
        )

        return inertia_values

    # ---------------------------------------------------------
    # Silhouette Analysis
    # ---------------------------------------------------------

    def silhouette_analysis(
        self,
        X_scaled,
        k_range=range(2, 11)
    ):

        logger.info(
            "Running silhouette analysis."
        )

        silhouette_scores = {}

        for k in k_range:

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(
                X_scaled
            )

            score = silhouette_score(
                X_scaled,
                labels
            )

            silhouette_scores[k] = score

            logger.info(
                f"K={k}, Silhouette Score={score:.4f}"
            )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            list(silhouette_scores.keys()),
            list(silhouette_scores.values()),
            marker="o"
        )

        plt.xlabel(
            "Number of Clusters (K)"
        )

        plt.ylabel(
            "Silhouette Score"
        )

        plt.title(
            "Silhouette Score Analysis"
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "silhouette_score.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        return silhouette_scores

    # ---------------------------------------------------------
    # Hierarchical Clustering
    # ---------------------------------------------------------

    def hierarchical_clustering(
        self,
        X_scaled
    ):

        logger.info(
            "Running hierarchical clustering."
        )

        linkage_matrix = linkage(
            X_scaled,
            method="ward"
        )

        plt.figure(
            figsize=(15, 7)
        )

        dendrogram(
            linkage_matrix,
            truncate_mode="level",
            p=5
        )

        plt.title(
            "Hierarchical Clustering Dendrogram"
        )

        plt.xlabel(
            "Customers"
        )

        plt.ylabel(
            "Distance"
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "hierarchical_dendrogram.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Dendrogram saved at {output_path}"
        )

        return linkage_matrix

    # ---------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------

    def dbscan_analysis(
        self,
        X_scaled,
        eps=1.5,
        min_samples=10
    ):

        logger.info(
            "Running DBSCAN."
        )

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan.fit_predict(
            X_scaled
        )

        unique_labels = set(
            labels
        )

        logger.info(
            f"DBSCAN clusters: {unique_labels}"
        )

        return labels

    # ---------------------------------------------------------
    # Final K-Means
    # ---------------------------------------------------------

    def train_final_model(
        self,
        X_scaled,
        optimal_k
    ):

        logger.info(
            f"Training final K-Means model with K={optimal_k}"
        )

        final_model = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10
        )

        cluster_labels = final_model.fit_predict(
            X_scaled
        )

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------

        model_file = os.path.join(
            self.model_path,
            "kmeans_model.joblib"
        )

        joblib.dump(
            final_model,
            model_file
        )

        logger.info(
            f"K-Means model saved at {model_file}"
        )

        return (
            final_model,
            cluster_labels
        )

    # ---------------------------------------------------------
    # Save Preprocessing Objects
    # ---------------------------------------------------------

    def save_preprocessor(
        self,
        imputer,
        scaler
    ):

        imputer_path = os.path.join(
            self.model_path,
            "imputer.joblib"
        )

        scaler_path = os.path.join(
            self.model_path,
            "scaler.joblib"
        )

        joblib.dump(
            imputer,
            imputer_path
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        logger.info(
            "Imputer and scaler saved successfully."

        )

    # ---------------------------------------------------------
    # Cluster Profile
    # ---------------------------------------------------------

    def create_cluster_profile(
        self,
        df,
        cluster_labels,
        clustering_features
    ):

        logger.info(
            "Creating cluster profile."
        )

        profile_df = df.copy()

        profile_df[
            "Cluster"
        ] = cluster_labels

        cluster_profile = (
            profile_df
            .groupby("Cluster")[clustering_features]
            .mean()
            .round(2)
        )

        cluster_counts = (
            profile_df[
                "Cluster"
            ]
            .value_counts()
            .sort_index()
        )

        cluster_profile[
            "Customer_Count"
        ] = cluster_counts

        cluster_profile[
            "Percentage"
        ] = (
            cluster_counts /
            len(profile_df) *
            100
        ).round(2)

        profile_path = os.path.join(
            self.model_path,
            "cluster_profile.csv"
        )

        cluster_profile.to_csv(
            profile_path
        )

        logger.info(
            f"Cluster profile saved at {profile_path}"
        )

        return (
            profile_df,
            cluster_profile
        )