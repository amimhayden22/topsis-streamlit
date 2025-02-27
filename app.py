import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Analisis Iklan TikTok Terbaik - Bharata Internasional",
    page_icon="🔥",
)

# Fungsi TOPSIS (sama seperti sebelumnya)
def topsis(decision_matrix, weights, criteria_types):
    norm = np.sqrt(np.sum(decision_matrix**2, axis=0))
    normalized_matrix = decision_matrix / norm
    weighted_normalized_matrix = normalized_matrix * weights

    ideal_best = np.zeros(len(weights))
    ideal_worst = np.zeros(len(weights))
    for i in range(len(weights)):
        if criteria_types[i] == 'max':  # Benefit criteria: semakin tinggi semakin baik
            ideal_best[i] = np.max(weighted_normalized_matrix[:, i])
            ideal_worst[i] = np.min(weighted_normalized_matrix[:, i])
        else:  # Cost criteria: semakin rendah semakin baik
            ideal_best[i] = np.min(weighted_normalized_matrix[:, i])
            ideal_worst[i] = np.max(weighted_normalized_matrix[:, i])

    separation_best = np.sqrt(np.sum((weighted_normalized_matrix - ideal_best)**2, axis=1))
    separation_worst = np.sqrt(np.sum((weighted_normalized_matrix - ideal_worst)**2, axis=1))
    performance_score = separation_worst / (separation_best + separation_worst)
    return performance_score

# Fungsi untuk memuat data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error saat memuat file: {e}")
        return None

# Main function aplikasi Streamlit
def main():
    st.title("Dashboard Analitik TOPSIS")
    st.write("Unggah file CSV, pilih kolom kriteria, tentukan tipe dan bobot, lalu lihat hasil TOPSIS dan visualisasinya.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            return

        st.subheader("Preview Data")
        st.dataframe(df.head())

        # Pilih kriteria yang akan digunakan
        all_columns = df.columns.tolist()
        criteria_columns = st.multiselect("Pilih kolom-kolom kriteria:", options=all_columns)
        
        if criteria_columns:
            st.markdown("### Tentukan Tipe Kriteria dan Bobot")
            st.markdown("Pilih tipe kriteria min atau max, min artinya nilai rendah maka bagus sedangkan max nilai tinggi maka bagus.")
            criteria_types = []
            weights = []
            default_weight = 1.0 / len(criteria_columns)
            for col in criteria_columns:
                ctype = st.radio(f"Tipe kriteria untuk **{col}**:", options=['min', 'max'], index=1, key=col)
                criteria_types.append(ctype)
                weight = st.number_input(f"Masukkan bobot untuk **{col}** (0-1):", min_value=0.0, max_value=1.0, value=default_weight, step=0.01, key=f"weight_{col}")
                weights.append(weight)
            
            # Normalisasi bobot sehingga total = 1
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                st.error("Total bobot harus lebih dari 0.")
                return

            st.write("Bobot yang digunakan (setelah normalisasi):")
            for col, wt in zip(criteria_columns, weights):
                st.write(f"- **{col}**: {wt:.4f}")
            
            # Pastikan kolom yang dipilih berupa numerik
            for col in criteria_columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')
            df_cleaned = df.dropna(subset=criteria_columns).copy()
            if df_cleaned.empty:
                st.error("Data tidak tersedia setelah pembersihan (NA).")
                return

            # Buat matriks keputusan dan jalankan TOPSIS
            decision_matrix = df_cleaned[criteria_columns].values.astype(float)
            performance_scores = topsis(decision_matrix, weights, criteria_types)
            df_cleaned["TOPSIS Score"] = performance_scores

            # Urutkan alternatif berdasarkan TOPSIS Score (nilai tertinggi lebih baik)
            df_sorted = df_cleaned.sort_values(by="TOPSIS Score", ascending=False)
            
            st.markdown("### Hasil Analisis TOPSIS")
            st.dataframe(df_sorted)

            # Bagian Dashboard Analitik
            st.markdown("## Dashboard Analitik")
            
            # Ringkasan KPI
            st.subheader("Ringkasan KPI")
            total_ads = len(df_sorted)
            best_ad = df_sorted.iloc[0]
            st.write(f"**Jumlah Iklan:** {total_ads}")
            st.write(f"**Iklan Terbaik:** {best_ad['Ad name']} dengan TOPSIS Score {best_ad['TOPSIS Score']:.4f}")
            
            # Grafik Bar 10 Iklan Teratas
            st.subheader("10 Iklan Teratas Berdasarkan TOPSIS Score")
            top10 = df_sorted.head(10)
            fig_bar = px.bar(top10, x="Ad name", y="TOPSIS Score", 
                             title="10 Iklan Teratas", 
                             labels={"TOPSIS Score": "Skor TOPSIS", "Ad name": "Nama Iklan"})
            st.plotly_chart(fig_bar)
            
            # Grafik Scatter: Contoh hubungan antara salah satu kriteria (misal, Impressions) dan TOPSIS Score
            if "Impressions" in criteria_columns:
                st.subheader("Hubungan Impressions dan TOPSIS Score")
                fig_scatter = px.scatter(df_sorted, x="Impressions", y="TOPSIS Score", 
                                         title="Scatter Plot Impressions vs TOPSIS Score",
                                         labels={"Impressions": "Impressions", "TOPSIS Score": "Skor TOPSIS"},
                                         hover_data=["Ad name"])
                st.plotly_chart(fig_scatter)
            
            st.markdown("#### 5 Alternatif Teratas")
            st.dataframe(df_sorted.head(5))

            st.markdown("#### 5 Alternatif Terbawah")
            st.dataframe(df_sorted.tail(5))
            
            # Opsi untuk menyimpan hasil
            if st.button("Simpan Hasil Analisis ke CSV"):
                result_file = "topsis_results_dashboard.csv"
                df_sorted.to_csv(result_file, index=False)
                st.success(f"Hasil analisis telah disimpan ke {result_file}")
    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

if __name__ == '__main__':
    main()