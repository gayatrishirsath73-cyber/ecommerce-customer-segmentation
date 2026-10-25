import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.customer_segmentation.logger import logger


class DataEDA:

    def __init__(self, visual_path):

        self.visual_path = visual_path

        os.makedirs(
            self.visual_path,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------

    def dataset_overview(self, df):

        logger.info("Generating dataset overview.")

        print("\nDataset Shape:")
        print(df.shape)

        print("\nDataset Information:")
        print(df.info())

        print("\nFirst 5 Rows:")
        print(df.head())

        print("\nDuplicate Rows:")
        print(df.duplicated().sum())

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def missing_value_analysis(self, df):

        logger.info("Performing missing value analysis.")

        missing = df.isnull().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        print("\nMissing Values:")
        print(missing)

        return missing

    # ---------------------------------------------------------
    # Numerical / Categorical Columns
    # ---------------------------------------------------------

    def identify_columns(self, df):

        numerical_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(
            f"Numerical columns: {numerical_columns}"
        )

        logger.info(
            f"Categorical columns: {categorical_columns}"
        )

        return numerical_columns, categorical_columns

    # ---------------------------------------------------------
    # Numerical Summary
    # ---------------------------------------------------------

    def numerical_summary(self, df, numerical_columns):

        logger.info("Generating numerical summary.")

        summary = df[numerical_columns].describe().T

        print("\nNumerical Summary:")
        print(summary)

        return summary

    # ---------------------------------------------------------
    # Correlation Heatmap
    # ---------------------------------------------------------

    def correlation_analysis(
        self,
        df,
        numerical_columns
    ):

        logger.info("Generating correlation heatmap.")

        correlation = df[
            numerical_columns
        ].corr()

        plt.figure(
            figsize=(16, 12)
        )

        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Correlation Heatmap"
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "correlation_heatmap.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Correlation heatmap saved at {output_path}"
        )

        return correlation

    # ---------------------------------------------------------
    # Distribution Plots
    # ---------------------------------------------------------

    def univariate_analysis(
        self,
        df,
        numerical_columns
    ):

        logger.info(
            "Generating univariate analysis plots."
        )

        for column in numerical_columns:

            plt.figure(
                figsize=(8, 5)
            )

            sns.histplot(
                df[column].dropna(),
                kde=True
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.tight_layout()

            file_name = (
                f"univariate_{column}.png"
            )

            output_path = os.path.join(
                self.visual_path,
                file_name
            )

            plt.savefig(
                output_path,
                dpi=300
            )

            plt.close()

        logger.info(
            "Univariate analysis completed."
        )

    # ---------------------------------------------------------
    # Boxplots
    # ---------------------------------------------------------

    def boxplot_analysis(
        self,
        df,
        numerical_columns
    ):

        logger.info(
            "Generating boxplots."
        )

        for column in numerical_columns:

            plt.figure(
                figsize=(8, 5)
            )

            sns.boxplot(
                y=df[column]
            )

            plt.title(
                f"Boxplot of {column}"
            )

            plt.tight_layout()

            file_name = (
                f"boxplot_{column}.png"
            )

            output_path = os.path.join(
                self.visual_path,
                file_name
            )

            plt.savefig(
                output_path,
                dpi=300
            )

            plt.close()

        logger.info(
            "Boxplot analysis completed."
        )

    # ---------------------------------------------------------
    # Categorical Distribution
    # ---------------------------------------------------------

    def categorical_analysis(
        self,
        df,
        categorical_columns
    ):

        logger.info(
            "Generating categorical analysis."
        )

        for column in categorical_columns:

            plt.figure(
                figsize=(10, 5)
            )

            sns.countplot(
                data=df,
                x=column
            )

            plt.xticks(
                rotation=45
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.tight_layout()

            file_name = (
                f"categorical_{column}.png"
            )

            output_path = os.path.join(
                self.visual_path,
                file_name
            )

            plt.savefig(
                output_path,
                dpi=300
            )

            plt.close()

        logger.info(
            "Categorical analysis completed."
        )