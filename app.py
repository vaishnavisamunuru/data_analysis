import streamlit as st
import pandas as pd
from loading_module import HeartDataLoader
from preprocessing_module import HeartDataPreprocessor
from EDA_module import HeartDataEDA
from MLModels_module import HeartClassifier
from io import BytesIO
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('Data/bg1.jpg')

# Streamlit app
def main():
    st.title("Data Analysis App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    tab1, tab2_0,tab2_1, tab3, tab4 = st.tabs(["Raw Data","Descriptive statistics", "EDA", "Preprocessed Data",'Modeling'])
    if uploaded_file is not None:
        uploaded_file_copy = BytesIO(uploaded_file.getvalue())
        data_loader = HeartDataLoader(uploaded_file)
        data_loader.load_data()

        eda_instance = HeartDataEDA(uploaded_file_copy)
        preprocessing = HeartDataPreprocessor(data_loader.data)
        with tab1:
            st.subheader("Raw Data")
            st.dataframe(data_loader.data)
        with tab2_0:
            st.subheader("Descriptive statistics")
            st.write(eda_instance.descriptive_stats())
        with tab2_1:
            cat_columns = st.multiselect('Select Categorical varibales',eda_instance.data.columns,default=None)
            eda_instance.data[cat_columns] = eda_instance.data[cat_columns].astype('category') 
            st.subheader("Plots")
            option = st.selectbox(label='Plot Type',options=['Correlation plot','Frequency plot','Distribution Plot','Bar Plot','Pie plot','Box Plot','Scatter Plot'])
            if option == 'Correlation plot':
                st.pyplot(eda_instance.plot_correlation_heatmap(),use_container_width=100)
            elif option == 'Frequency plot':
                col = st.selectbox(label='Select column',options=eda_instance.data.select_dtypes(include='category').columns)
                if col:
                    st.pyplot(eda_instance.variable_frequencies(col),use_container_width=100)
                else:
                    st.warning('No category columns')
            elif option == 'Distribution Plot':
                col = st.selectbox(label='Select column',options=eda_instance.data.select_dtypes(include='number').columns)
                st.pyplot(eda_instance.plot_distribution(col),use_container_width=100)
            elif option == 'Pie plot':
                col = st.selectbox(label='Select column',options=eda_instance.data.select_dtypes(include='category').columns)
                st.pyplot(eda_instance.pie_chart(col),use_container_width=100)
            elif option == 'Bar Plot':
                col1 = st.selectbox(label='Select X',options=eda_instance.data.select_dtypes(include='category').columns,index=1)
                col2 = st.selectbox(label='Select Y',options=eda_instance.data.select_dtypes(include='number').columns)
                col3 = st.selectbox(label='Select Hue',options=[None]+list(eda_instance.data.select_dtypes(include='category').columns))
                st.pyplot(eda_instance.grouped_bar_plot(col1,col2,col3),use_container_width=100)
            elif option == 'Box Plot':
                col1 = st.selectbox(label='Select X',options=eda_instance.data.select_dtypes(include='category').columns,index=1)
                col2 = st.selectbox(label='Select Y',options=eda_instance.data.select_dtypes(include='number').columns)
                col3 = st.selectbox(label='Select Hue',options=[None]+list(eda_instance.data.select_dtypes(include='category').columns))
                st.pyplot(eda_instance.box_plot(col1,col2,col3),use_container_width=100)
            elif option == 'Scatter Plot':
                col1 = st.selectbox(label='Select X',options=eda_instance.data.select_dtypes(include='number').columns,index=1)
                col2 = st.selectbox(label='Select Y',options=eda_instance.data.select_dtypes(include='number').columns)
                col3 = st.selectbox(label='Select Hue',options=[None]+list(eda_instance.data.select_dtypes(include='category').columns))
                st.pyplot(eda_instance.scatter_plot(col1,col2,col3),use_container_width=100)
            pass
        with tab3:
            options = st.multiselect("Select Preprocess steps",['Handle Missing Values','Encoding Categorical Variables','Feature Scaling'],default=None)
            if 'Handle Missing Values' in options:
                preprocessing.handle_missing_values()
            if 'Encoding Categorical Variables'in options:
                preprocessing.encode_categorical_variables()
            if 'Feature Scaling' in options:
                scale_columns = st.multiselect('Select columns to scale',eda_instance.data.columns,default=None)
                if scale_columns:
                    preprocessing.scale_numerical_features(scale_columns)
            st.caption('Data after preproessing')
            st.dataframe(preprocessing.data)

        with tab4:
            cols = [column for column in preprocessing.data.columns  if preprocessing.data[column].nunique() <= 5   ]
            option1 = st.selectbox(label='Target variable',options=cols)
            indept = list(preprocessing.data.columns)
            if option1: 
                indept.remove(option1)
            option2 = list(st.multiselect("Independent variable",indept,default=indept))
            if option1 and option2:
                if st.button('submit'):
                    classifier = HeartClassifier(preprocessing.data,option1,option2)
                    st.caption('Logistic Regression')
                    with st.spinner('Running logistic model'):
                        st.text(classifier.model_fit('Logistic Regression'))
                    st.caption('Random Forest')
                    with st.spinner('Running Random forest model'):
                        st.text(classifier.model_fit('Random Forest'))
                    st.caption('SVC')
                    with st.spinner('Running SVC model (might run longer)'):
                        st.text(classifier.model_fit('SVC'))

if __name__ == "__main__":
    main()
