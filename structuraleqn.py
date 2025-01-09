import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import statsmodels.api as sm
import requests
import pyreadstat
import tempfile
import json
from sklearn.preprocessing import MinMaxScaler
from SurveyAnalyzer import SPSS_file_manager, survey_interface, data_analyzer



# Selettore per scegliere tra survey o dati YouGov nella barra laterale
st.title("Structural Equation Creator from Survey")

# Condizionale che cambia l'interfaccia in base alla selezione
# Inizializziamo le classi
file_manager = SPSS_file_manager()
interface = survey_interface()
analyzer = data_analyzer()

dataset = st.file_uploader("Upload File", type=["sav"])

if dataset is not None:
    # Carica il file SPSS
    df, meta = file_manager.load_spss_file(dataset)
    df.index = df["record"]

    # Seleziona le domande
    (
        selected_questions,
        df_currentcustomer,
        brands,
        dict_rename_cols,
    ) = interface.select_questions(df, meta)

    if all([selected_questions, brands, dict_rename_cols]):
        if all(selected_questions.values()):
            # Analisi e creazione dei grafici
            (
                coefficients_table,
                detailed_coefficients,
                fig,
            ) = analyzer.create_coefficients_and_plots(
                df,
                selected_questions,
                df_currentcustomer,
                brands,
                dict_rename_cols,
                interface.Fs,
            )

            coefficients_df = pd.DataFrame(
                coefficients_table,
                columns=["F", "Coefficiente", "Peso Coefficiente"],
            )
            coefficients_df["Peso Coefficiente"] = (
                coefficients_df["Coefficiente"]
                / coefficients_df["Coefficiente"].sum()
                * 100
            )
            st.write("Tabella Coefficienti:", coefficients_df)

            detailed_df = pd.DataFrame(
                detailed_coefficients,
                columns=["F", "Coefficiente", "Intercetta", "p-value"],
            )
            with st.expander("Tabella Dettagliata"):
                st.write(detailed_df)
                st.write('Se il p-value è pià piccolo di 0.05 siamo a posto.')

            st.pyplot(fig)


exit()