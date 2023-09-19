import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the model
try:
    model = pickle.load(open('prediksi_co2.sav', 'rb'))
except FileNotFoundError:
    st.error("File 'prediksi_co2.sav' tidak ditemukan. Pastikan model tersedia.")

# Load the data from URL
url = 'https://github.com/adyanamul/dataset/raw/main/CO2_dataset.xlsx'
try:
    df = pd.read_excel(url)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index(['Year'], inplace=True)
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca data dari URL: {str(e)}")

st.title('Forecasting Kualitas Udara Depok')
year = st.slider("Tentukan Tahun", 1, 30, step=1)

pred = model.forecast(year)
pred = pd.DataFrame(pred, columns=['Predicted CO2'])

if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Hasil Prediksi")
        st.dataframe(pred)

        # Calculate evaluation metrics
        actual_values = df['CO2'].loc[df.index.year >= df.index.year.max() - year]
        rmse = np.sqrt(mean_squared_error(actual_values, pred))
        mae = mean_absolute_error(actual_values, pred)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAE: {mae:.2f}")

        # Print additional information for debugging
        print("Panjang actual_values:", len(actual_values))
        print("Panjang pred:", len(pred))
        print("Indeks data aktual:", actual_values.index)
        print("Indeks data prediksi:", pred.index)

    with col2:
        st.subheader("Grafik Prediksi")
        fig, ax = plt.subplots()
        df['CO2'].plot(style='--', color='gray', legend=True, label='Actual CO2')
        pred['Predicted CO2'].plot(color='b', legend=True, label='Predicted CO2')
        plt.xlabel('Tahun')
        plt.ylabel('CO2')
        plt.title('Prediksi Kualitas Udara Depok')
        st.pyplot(fig)

# Informasi tambahan
st.info("Ini adalah aplikasi untuk meramalkan kualitas udara Depok menggunakan model prediksi CO2.")
st.markdown("""
### Tentang Model
Model ini merupakan model prediksi kualitas udara Depok yang menggunakan data historis CO2. Model ini menggunakan metode time series forecasting.

### Metrik Evaluasi
Kami telah menghitung metrik evaluasi prediksi, yaitu Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE), untuk memberikan gambaran tentang seberapa baik predik
