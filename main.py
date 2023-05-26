import socketserver
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

#Data Processing
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

st.set_page_config(page_title='Auto Prep')

def hide_streamlit_style():
    hide_streamlit_css = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_css, unsafe_allow_html=True)

def impute_missing_values(data):
    imputed_data = pd.DataFrame()
    for col in data.columns:
        col_data = data[col]

        if col_data.dtype == 'O' or col_data.dtype == 'object':  # categorical columns
            missing_values = col_data.isnull()
            non_missing_values = col_data[~missing_values].unique()

            if len(non_missing_values) > 0 and missing_values.sum() > 0:
                mode_values = col_data.mode()
                col_data.fillna(mode_values[0], inplace=True)

                if len(non_missing_values) > 0 and missing_values.sum() > 0:
                    mode_values = col_data.mode()
                    col_data.fillna(mode_values[0], inplace=True)

                    # Encode categorical values as integers
                    encoded_values, encoded_labels = pd.factorize(col_data)

                    # Calculate distances using Jaccard metric
                    distances = cdist(encoded_values[missing_values].reshape(-1, 1),
                                      encoded_values[~missing_values].reshape(-1, 1),
                                      metric='hamming')

                    # Find nearest non-missing values for imputation
                    nearest_idx = np.argmin(distances, axis=1)
                    nearest_value = non_missing_values[nearest_idx]

                    # Create a copy of the column for imputation
                    col_data_imputed = col_data.copy()
                    col_data_imputed.loc[missing_values] = nearest_value

                    imputed_data[col] = col_data_imputed
                else:
                    imputed_data[col] = col_data

            imputed_data[col] = col_data
            print("Categorical Imputing")

        elif col_data.isnull().sum() < len(col_data) * 0.05:  # numerical columns with < 5% missing data
            imputer = SimpleImputer(strategy='mean')
            imputed_col = imputer.fit_transform(col_data.values.reshape(-1, 1)).ravel()
            imputed_data[col] = imputed_col
            print("Mean Imputing")

        elif col_data.isnull().sum() < len(col_data) * 0.2:  # numerical columns with < 20% missing data
            imputer = IterativeImputer()
            imputed_col = imputer.fit_transform(col_data.values.reshape(-1, 1)).ravel()
            imputed_data[col] = imputed_col
            print("Iterative Imputing")

        else:  # numerical columns with >= 20% missing data
            imputer = KNNImputer()
            imputed_col = imputer.fit_transform(col_data.values.reshape(-1, 1)).ravel()
            imputed_data[col] = imputed_col
            print("KNN Imputing")

    return imputed_data


# Converting categorical columns to numerical
# Args: Args: ohe - True if columns to be one hot encoded, False for label encoding
#       dropFirst - if ohe is True, indicates if first column for each ohe conversion is to be dropped
#       threshold - the default vaule is 5 which means if the unique values present is the less then threshold vaue
#                   we will enocode the columns
def catEncoding(df, threshold, dropFirst, ohe):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Get the number of unique values for each categorical column
    num_unique_values = df[cat_cols].nunique()

    # Create a list of tuples containing the column name, the OneHotEncoder object, and the list of columns to be encoded
    encoders = [(col, OneHotEncoder(sparse=False, drop='first'), [col]) for col in cat_cols if num_unique_values[col] <= threshold]

    # Encode the categorical columns
    for col, encoder, cols in encoders:
        if not df[col].any():
            continue
        encoded_cols = pd.get_dummies(df[col], prefix=col)

        if dropFirst:
            encoded_cols = encoded_cols.iloc[:, 1:]
        df = pd.concat([df, encoded_cols], axis=1)

    # Drop the original categorical columns if ohe=True
    if ohe:
        df.drop(columns=cat_cols, inplace=True)

    return df


# Remove outliers from the dataset
# Args: threshold - specifies the number of standard deviations upto which the values are to be kept
#                   default = 3
def remOutliers(df, method='winsorize', threshold=3):
    """
    Function to handle outliers in a dataset.

    Parameters:
    - data: a numpy array or pandas dataframe containing the data
    - method: the method to use for handling outliers (default='remove')
        - 'remove': remove all data points that are more than 'threshold' standard deviations from the mean
        - 'winsorize': replace all data points that are more than 'threshold' standard deviations from the mean with the value at the   'threshold' percentile
    - threshold: the number of standard deviations from the mean to consider as an outlier (default=3)

    Returns:
    - a numpy array or pandas dataframe with the outliers handled according to the chosen method
    """

    # Calculate the mean and standard deviation of the data
    mean = np.mean(df)
    std = np.std(df)

    # Identify the outliers using the chosen method
    if method == 'remove':
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outliers = (df < lower_bound) | (df > upper_bound)
        df = df[~outliers]

    elif method == 'winsorize':
        p_low = np.percentile(df, threshold)
        p_high = np.percentile(df, 100 - threshold)
        df = np.clip(df, p_low, p_high)

    else:
        raise ValueError("Invalid method: must be 'remove' or 'winsorize'")

    return df


def processDataFrame(dataframe, threshold=5, ohe=False, dropFirst=False):
    df_missing = dataframe
    if df_missing.isnull().sum().sum() != 0:
        df_missing = impute_missing_values(df_missing)


    df_encoded = catEncoding(df_missing, threshold=threshold, ohe=ohe, dropFirst=dropFirst)

    # Separating out the numerical columns
    numerical_cols = df_encoded.select_dtypes(include=['float', 'int'])

    # Drop the original numerical columns from the original dataframe
    df_drop_cat = df_encoded.drop(numerical_cols.columns, axis=1)
    df_drop_num = remOutliers(numerical_cols)

    # Append the separated numerical columns to the original dataframe
    df = pd.concat([df_drop_num, df_drop_cat], axis=1)
    return df


def preprocess(data):
    st.subheader("Automated Data Pre-processing")

    df = pd.read_csv(data)

    new_df = processDataFrame(df)
    processedData = new_df.to_csv(index=False)

    st.write("Dataset Preprocessing Done Successfully")

    return new_df



def eda(processed_df):
    st.subheader("Exploratory Data Analysis")
    st.dataframe(processed_df.head())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    all_columns = processed_df.columns.to_list()

    if st.checkbox("Show Shape"):
        shape_placeholder = st.empty()  # Create a placeholder for shape output
        shape_placeholder.write("Shape: ", end="")
        shape_placeholder.write(processed_df.shape)
        # st.write(processed_df.shape)

    if st.checkbox("Show Columns"):
        st.write(all_columns)

    if st.checkbox("Summary"):
        st.write(processed_df.describe())

    if st.checkbox("Show Selected Columns"):
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = processed_df[selected_columns]
        st.dataframe(new_df)

    if st.checkbox("Show Value Counts", key="show_shape_checkbox"):
        st.write(processed_df.iloc[:, -1].value_counts())





def plots(df):
    st.subheader("Data Visualization")

    if st.checkbox("Correlation Plot(Matplotlib)"):
        plt.matshow(processed_df.corr())
        st.pyplot()

    if st.checkbox("Correlation Plot(Seaborn)"):
        st.write(sns.heatmap(processed_df.corr(), annot=True))
        st.pyplot()

    if st.checkbox("Pie Plot"):
        all_columns = processed_df.columns.to_list()
        column_to_plot = st.selectbox("Select 1 Column", all_columns)
        pie_plot = processed_df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
        st.write(pie_plot)
        st.pyplot()

        # Customizable Plot

    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box"])
    selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))

        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)

        # Custom Plot
        elif type_of_plot:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()



if __name__ == '__main__':
    hide_streamlit_style()
    st.title("Automated Dataset Preprocessing and Visualization")
    data = st.file_uploader("Upload a Dataset", type=["csv", "xlsx"])

    if data is not None:
        processed_df = preprocess(data)

        processedData = processed_df.to_csv(index=False)

        st.write("Want to Download Cleaned Dataset ?", allow_output_mutation=True)
        # Create a download button
        download_button = st.download_button(label="Download Dataset", data=processedData, file_name="Clean_Data.csv")

        eda(processed_df)

        plots(processed_df)

