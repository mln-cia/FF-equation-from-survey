import streamlit as st
import tempfile
import pyreadstat
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# Classe per la gestione dei file SPSS
class SPSS_file_manager:
    @staticmethod
    @st.cache_data()
    def load_spss_file(dataset):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(dataset.getvalue())
            tmp_file_path = tmp_file.name
        df, meta = pyreadstat.read_sav(tmp_file_path)
        return df, meta


# Classe per la logica dell'interfaccia utente
class survey_interface:
    def __init__(self):
        self.Fs = [
            "Familiarity",
            "Feeling",
            "Favourability",
            "Fervor",
            "Findability",
            "Facilitation",
            "Fascination",
            "Following",
        ]

    @staticmethod
    def create_options_dict(meta):
        options = set(
            i.split(":")[0].split("r")[0] + i.split("-")[-1].split(":")[-1]
            for i in meta.column_names_to_labels.values()
        )
        dict_options = {
            i.split(":")[0].split("r")[0] + i.split("-")[-1].split(":")[-1]: i
            for i in meta.column_names_to_labels.values()
        }
        return options, dict_options

    def select_questions(self, df, meta):
        options, dict_options = self.create_options_dict(meta)

        question_currentcustomer = st.selectbox(
            "Seleziona la domanda relativa alla " +
            "variabile Target - **Current Customer**",
            options=options,
            index=None,
        )

        if question_currentcustomer:
            dict_rename_cols = {
                key: val.split(":")[-1].split("-")[0].strip()
                for key, val in meta.column_names_to_labels.items()
            }
            df_currentcustomer = (
                df[
                    [
                        col
                        for col in df.columns
                        if question_currentcustomer.split(" ", 1)[0] +
                        "r" in col
                    ]
                ]
                .rename(columns=dict_rename_cols)
                .fillna(0)
            )
            brands = st.multiselect(
                "Select Relevant Brands",
                options=df_currentcustomer.columns,
                default=df_currentcustomer.columns.to_list(),
            )

            selected_questions = {}
            col1, col2 = st.columns(2)
            with col1:
                for f in self.Fs[:4]:
                    selected_questions[f] = st.selectbox(
                        "Seleziona la domanda relativa alla " +
                        f"variabile **{f}**",
                        options=options,
                        key=f"{f}_selectbox_1",
                        index=None,
                    )
            with col2:
                for f in self.Fs[4:]:
                    selected_questions[f] = st.selectbox(
                        "Seleziona la domanda relativa alla " +
                        f"variabile **{f}**",
                        options=options,
                        key=f"{f}_selectbox_2",
                        index=None,
                    )
            return (selected_questions, df_currentcustomer,
                    brands, dict_rename_cols)
        
            st.write(selected_questions)

        else:
            return None, None, None, None


# Classe per gestire l'analisi e la visualizzazione dei dati
class data_analyzer:
    @staticmethod
    def create_coefficients_and_plots(
        df, selected_questions, df_currentcustomer, brands,
        dict_rename_cols, Fs
    ):
        coefficients_table = []
        detailed_coefficients = []
        fig, axes = plt.subplots(4, 2, figsize=(18, 15))
        axes = axes.flatten()

        for i, f in enumerate(Fs):
            df_F = (
                df[
                    [
                        col
                        for col in df.columns
                        if selected_questions[f].split(" ", 1)[0] + "r" in col and col.startswith(selected_questions[f].split(" ", 1)[0])
                    ]
                ]
                .rename(columns=dict_rename_cols)
                .fillna(0)
            )

            final_df = pd.concat(
                [
                    df_F.sum() / df_F.count(),
                    df_currentcustomer.sum() / df_currentcustomer.count(),
                ],
                axis=1,
            )
            final_df.columns = [f, "CurrentCustomer"]
            final_df = final_df.loc[brands]
            # final_df[f] = final_df[f]/final_df[f].max()

            X_fit = final_df[[f]]
            X_with_intercept = sm.add_constant(X_fit)
            y = final_df["CurrentCustomer"]

            linear_model = sm.OLS(y, X_with_intercept)
            result = linear_model.fit()

            coefficients_table.append([f, result.params[f], 0])
            detailed_coefficients.append(
                [f, result.params[f], result.params[0], result.pvalues[f]]
            )

            sns.scatterplot(
                data=final_df, x=f, y="CurrentCustomer", ax=axes[i]
            )
            for k in range(len(final_df)):
                axes[i].text(
                    final_df[f].iloc[k],
                    final_df["CurrentCustomer"].iloc[k],
                    final_df.index[k],
                    fontsize=11,
                )

            predicted_values = result.predict(X_with_intercept)
            axes[i].plot(
                final_df[f], predicted_values, color="red",
                label="Linea di Regressione"
            )
            axes[i].legend()

        return coefficients_table, detailed_coefficients, fig
