import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛍️ E-Commerce Customer Segmentation")

st.write(
    """
    Segment customers based on purchasing behavior,
    engagement, spending patterns and shopping activity
    using K-Means clustering.
    """
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "artifacts/models"

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "kmeans_model.joblib"
)

SCALER_FILE = os.path.join(
    MODEL_PATH,
    "scaler.joblib"
)

IMPUTER_FILE = os.path.join(
    MODEL_PATH,
    "imputer.joblib"
)


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
# PERSONA MAPPING
# ============================================================

PERSONA_MAPPING = {

    0: "At-Risk High-Value Customers",

    1: "Occasional Buyers",

    2: "Window Shoppers",

    3: "Discount-Driven Customers",

    4: "Premium Loyal Customers",

    5: "Moderate-Value / Growth Customers",

    6: "Low-Conversion / Low-Value Customers"

}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    imputer = joblib.load(
        IMPUTER_FILE
    )

    return model, scaler, imputer


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # ========================================================
    # READ DATASET
    # ========================================================

    df = pd.read_csv(
        uploaded_file
    )

    st.success(
        "Dataset uploaded successfully!"
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(
        "📄 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:

        st.metric(
            "Total Features",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    missing_features = [

        feature
        for feature in CLUSTERING_FEATURES
        if feature not in df.columns

    ]


    if missing_features:

        st.error(
            "The uploaded dataset is missing "
            "required clustering features:"
        )

        st.write(
            missing_features
        )

        st.stop()


    # ========================================================
    # RUN SEGMENTATION
    # ========================================================

    if st.button(
        "🚀 Run Customer Segmentation",
        type="primary"
    ):

        with st.spinner(
            "Running customer segmentation..."
        ):

            # ------------------------------------------------
            # Load trained objects
            # ------------------------------------------------

            model, scaler, imputer = load_models()


            # ------------------------------------------------
            # Select clustering features
            # ------------------------------------------------

            X = df[
                CLUSTERING_FEATURES
            ].copy()


            # ------------------------------------------------
            # Handle missing values
            # ------------------------------------------------

            X_imputed = imputer.transform(
                X
            )


            # ------------------------------------------------
            # Scale features
            # ------------------------------------------------

            X_scaled = scaler.transform(
                X_imputed
            )


            # ------------------------------------------------
            # Predict clusters
            # ------------------------------------------------

            clusters = model.predict(
                X_scaled
            )


            # ------------------------------------------------
            # Add cluster number
            # ------------------------------------------------

            result_df = df.copy()

            result_df[
                "Cluster"
            ] = clusters


            # ------------------------------------------------
            # Add business persona
            # ------------------------------------------------

            result_df[
                "Customer_Segment"
            ] = result_df[
                "Cluster"
            ].map(
                PERSONA_MAPPING
            )


        st.success(
            "Customer segmentation completed!"
        )


        # ====================================================
        # KPI SECTION
        # ====================================================

        st.subheader(
            "📊 Segmentation Summary"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Total Customers",
                len(result_df)
            )


        with col2:

            st.metric(
                "Number of Clusters",
                result_df[
                    "Cluster"
                ].nunique()
            )


        with col3:

            highest_value_cluster = (
                result_df
                .groupby("Cluster")[
                    "Total_Spending"
                ]
                .mean()
                .idxmax()
            )

            st.metric(
                "Highest Value Cluster",
                highest_value_cluster
            )


        with col4:

            st.metric(
                "Average Spending",
                f"₹{result_df['Total_Spending'].mean():,.0f}"
            )


        # ====================================================
        # CLUSTER DISTRIBUTION
        # ====================================================

        st.subheader(
            "👥 Customer Distribution"
        )

        cluster_counts = (
            result_df[
                "Customer_Segment"
            ]
            .value_counts()
        )

        st.bar_chart(
            cluster_counts
        )


        # ====================================================
        # CLUSTER PROFILE
        # ====================================================

        st.subheader(
            "📈 Cluster Profile"
        )

        profile = (

            result_df

            .groupby(
                [
                    "Cluster",
                    "Customer_Segment"
                ]
            )

            [

                [
                    "Annual_Income",
                    "Total_Orders",
                    "Total_Spending",
                    "Average_Order_Value",
                    "Website_Visits_Per_Month",
                    "Cart_Abandonment_Rate",
                    "Loyalty_Points",
                    "Last_Purchase_Days_Ago"
                ]

            ]

            .mean()

            .round(2)

        )


        st.dataframe(
            profile,
            use_container_width=True
        )


        # ====================================================
        # CUSTOMER SEGMENTED DATA
        # ====================================================

        st.subheader(
            "👤 Segmented Customers"
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )


        # ====================================================
        # DOWNLOAD RESULT
        # ====================================================

        csv = (
            result_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(

            label="⬇️ Download Segmented Customer Data",

            data=csv,

            file_name="segmented_customer_data.csv",

            mime="text/csv"

        )


else:

    st.info(
        "Please upload a customer CSV file to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "E-Commerce Customer Segmentation | "
    "K-Means Clustering"
)