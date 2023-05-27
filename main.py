import socketserver
import streamlit as st
import autoPrep

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

st.set_page_config(page_title='Auto Prep')

def hide_streamlit_style():
    hide_streamlit_css = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_css, unsafe_allow_html=True)


def preprocess(data):
    st.subheader("Automated Data Pre-processing")

    df = pd.read_csv(data)

    new_df = autoPrep.processDataFrame(df)
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
    data = st.file_uploader("Upload a Dataset", type=["csv"])

    if data is not None:
        processed_df = preprocess(data)

        processedData = processed_df.to_csv(index=False)

        st.write("Want to Download Cleaned Dataset ?", allow_output_mutation=True)
        # Create a download button
        download_button = st.download_button(label="Download Dataset", data=processedData, file_name="Clean_Data.csv")

        eda(processed_df)

        plots(processed_df)

