import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import pickle
import os
import anthropic
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
import re
import time
from datetime import datetime

if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""

# =============================================
# KONFIGURASI AWAL
# =============================================
try:
    api_key = st.secrets["anthropic"]["api_key"]
except KeyError:
    st.error("""
    ❌ **API Key Anthropic tidak ditemukan!**
    
    Cara memperbaiki di Streamlit Cloud:
    1. Klik **Manage app** (pojok kanan bawah)
    2. Buka **Settings → Secrets**
    3. Tambahkan konfigurasi berikut:
    
    ```toml
    [anthropic]
    api_key = "sk-ant-api03-XXXXXXXXXXXXXXXX"
    ```
    
    Dapatkan API key di: https://console.anthropic.com
    """)
    st.stop()

client = anthropic.Anthropic(api_key=api_key)


# =============================================
# FUNGSI UTAMA
# =============================================

def ensure_persistent_data():
    if 'persistent_data' not in st.session_state:
        try:
            if os.path.exists("uploaded_data.pkl"):
                with open("uploaded_data.pkl", "rb") as f:
                    saved_data = pickle.load(f)
                    st.session_state.persistent_data = saved_data
            else:
                st.session_state.persistent_data = None
        except Exception as e:
            st.error(f"Gagal memuat data tersimpan: {str(e)}")
            st.session_state.persistent_data = None


def get_ai_insight(parameter, data_series, chart_type, chart_data=None):
    prompt = f"""
Kamu adalah seorang ahli analisis data pembangkit listrik yang menggunakan boiler CFB dan Steam turbin dengan kapasitas 25 MW (Power Plant Performance Analyst).
Analisis berikut berasal dari parameter operasional pada PLTU (Pembangkit Listrik Tenaga Uap).

Tugasmu adalah:
1. Jelaskan tren utama dari parameter {parameter} pada grafik {chart_type} dalam 1-2 kalimat.
2. Identifikasi 1-2 fluktuasi besar, anomali, atau outlier yang tidak biasa (jika ada).
3. Berikan analisa singkat (maksimal 2 kalimat) dampaknya terhadap efisiensi dan stabilitas pembangkit.
4. Rekomendasikan 1-2 tindakan teknis untuk operator atau pemantauan jika ada deviasi signifikan.

Format jawaban HARUS mengikuti template berikut:
[TREN] <jelasan tren>
[ANOMALI] <penjelasan anomali>
[DAMPAK] <analisis dampak>
[REKOMENDASI] <saran tindakan>
"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_content = response.content[0].text
        sections = {
            "TREN": "🔍 **Tren Utama**",
            "ANOMALI": "⚠️ **Anomali yang Teridentifikasi**",
            "DAMPAK": "⚡ **Dampak Operasional**",
            "REKOMENDASI": "🛠️ **Rekomendasi Tindakan**"
        }
        formatted_result = ""
        for line in ai_content.splitlines():
            line = line.strip()
            if line.startswith("[TREN]"):
                formatted_result += f"\n{sections['TREN']}\n{line[6:].strip()}\n"
            elif line.startswith("[ANOMALI]"):
                formatted_result += f"\n{sections['ANOMALI']}\n{line[9:].strip()}\n"
            elif line.startswith("[DAMPAK]"):
                formatted_result += f"\n{sections['DAMPAK']}\n{line[8:].strip()}\n"
            elif line.startswith("[REKOMENDASI]"):
                formatted_result += f"\n{sections['REKOMENDASI']}\n{line[13:].strip()}\n"
        return formatted_result if formatted_result else ai_content
    except Exception as e:
        return f"Error dalam mendapatkan insight AI: {str(e)}"


def get_anomaly_insight(parameter, anomaly_data, method, metrics, correlation_data=None):
    correlation_info = ""
    if correlation_data:
        correlation_info = "\n\nINFORMASI KORELASI PARAMETER:\n"
        for i, (corr_param, corr_value, direction, strength, explanation) in enumerate(correlation_data[:3]):
            correlation_info += f"{i+1}. {corr_param}: {corr_value:.3f} ({direction}, {strength})\n"
            correlation_info += f"   - {explanation}\n"

    prompt = f"""
Kamu adalah seorang ahli analisis data pembangkit listrik PLTU dengan boiler CFB dan Steam turbin kapasitas 25 MW (Power Plant Performance Analyst). Referensimu adalah buku Boiler Operation & Design, Standar EPRI, ASME, dan standar industri lainnya.

Data anomali berikut berasal dari parameter {{parameter}} pada PLTU menggunakan metode deteksi {{method}}.

STATISTIK ANOMALI:
- Jumlah anomali terdeteksi: {{metrics['count']}}
- Persentase anomali: {{metrics['percent']}}
- Anomali nilai TINGGI (di atas normal): {{metrics['high_count']}} kejadian
- Anomali nilai RENDAH (di bawah normal): {{metrics['low_count']}} kejadian
- Rata-rata nilai saat anomali: {{metrics['mean']:.2f}}
- Rata-rata nilai historis normal: {{metrics['hist_mean']:.2f}}
- Deviasi terhadap nilai normal: {{metrics['deviation']:.2f}}%
{{correlation_info}}

INSTRUKSI: Berikan analisis teknis mendalam dan SPESIFIK untuk parameter {{parameter}} pada sistem PLTU. JANGAN berikan jawaban generik.

Format jawaban WAJIB mengikuti template ini PERSIS:

[ANALISIS]
Tulis 3-4 kalimat penjelasan teknis mendalam tentang anomali {{parameter}}, karakteristik deviasi dan implikasinya pada sistem PLTU.

[PENYEBAB]
1. Penyebab pertama yang paling relevan dengan {{parameter}}
   - Detail teknis sub-poin pertama
   - Detail teknis sub-poin kedua
   - Detail teknis sub-poin ketiga
2. Penyebab kedua yang relevan dengan {{parameter}}
   - Detail teknis sub-poin pertama
   - Detail teknis sub-poin kedua
   - Detail teknis sub-poin ketiga
3. Penyebab ketiga yang relevan dengan {{parameter}}
   - Detail teknis sub-poin pertama
   - Detail teknis sub-poin kedua
   - Detail teknis sub-poin ketiga
4. Penyebab keempat yang relevan dengan {{parameter}}
   - Detail teknis sub-poin pertama
   - Detail teknis sub-poin kedua
   - Detail teknis sub-poin ketiga

[REKOMENDASI]
1. Tindakan pertama SPESIFIK untuk {{parameter}}
   - Langkah teknis detail pertama
   - Langkah teknis detail kedua
   - Target nilai atau parameter yang diharapkan
2. Tindakan kedua SPESIFIK untuk {{parameter}}
   - Langkah teknis detail pertama
   - Langkah teknis detail kedua
   - Alat atau metode yang digunakan
3. Tindakan ketiga SPESIFIK untuk {{parameter}}
   - Langkah teknis detail pertama
   - Langkah teknis detail kedua
   - Frekuensi atau jadwal pelaksanaan
4. Tindakan keempat untuk monitoring lanjutan
   - Parameter yang perlu dipantau bersama
   - Threshold batas aman yang direkomendasikan
   - Eskalasi jika anomali berlanjut

[DAMPAK]
Tulis 3-4 kalimat dampak spesifik jika anomali {{parameter}} dibiarkan: efisiensi termal, keandalan peralatan, konsumsi bahan bakar, dan keselamatan operasi PLTU.
"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            temperature=0.7,
            messages=[{{"role": "user", "content": prompt}}]
        )
        ai_content = response.content[0].text

        sections = {{
            "ANALISIS": "\U0001f52c **Analisis Teknis Anomali**",
            "PENYEBAB": "\u26a0\ufe0f **Kemungkinan Penyebab**",
            "REKOMENDASI": "\U0001f6e0\ufe0f **Rekomendasi Tindak Lanjut**",
            "DAMPAK": "\U0001f4a5 **Potensi Dampak**"
        }}

        formatted_result = ""
        for section, header in sections.items():
            section_tag = f"[{{section}}]"
            if section_tag in ai_content:
                start_pos = ai_content.find(section_tag) + len(section_tag)
                next_pos = len(ai_content)
                for next_tag in [f"[{{s}}]" for s in sections.keys()]:
                    tag_pos = ai_content.find(next_tag, start_pos)
                    if tag_pos > -1 and tag_pos < next_pos:
                        next_pos = tag_pos
                section_content = ai_content[start_pos:next_pos].strip()

                if section in ["PENYEBAB", "REKOMENDASI"]:
                    lines_raw = section_content.split("\n")
                    formatted_lines = []
                    current_point = None
                    for line in lines_raw:
                        line = line.strip()
                        if not line:
                            if current_point is not None and formatted_lines and formatted_lines[-1] != "":
                                formatted_lines.append("")
                            continue
                        number_match = re.match(r"^(\d+)\.(\s|$)", line)
                        if number_match:
                            number = number_match.group(1)
                            rest_of_line = line[len(number_match.group(0)):].strip()
                            if current_point is not None and formatted_lines and formatted_lines[-1] != "":
                                formatted_lines.append("")
                            current_point = number
                            formatted_lines.append(f"**{{number}}.** {{rest_of_line}}")
                        elif line.startswith(("*", "\u2022", "-")):
                            sub_content = line[1:].strip()
                            formatted_lines.append(f"   - {{sub_content}}")
                        elif current_point is not None:
                            if formatted_lines and formatted_lines[-1].startswith(f"**{{current_point}}.**"):
                                formatted_lines[-1] = f"{{formatted_lines[-1]}} {{line}}"
                            else:
                                formatted_lines.append(f"   - {{line}}")
                        else:
                            current_point = "1" if not formatted_lines else str(int(current_point or "0") + 1)
                            formatted_lines.append(f"**{{current_point}}.** {{line}}")
                    section_content = "\n".join(formatted_lines)

                formatted_result += f"\n{{header}}\n{{section_content}}\n"

        return formatted_result if formatted_result else ai_content

    except Exception as e:
        dev = metrics["deviation"]
        high = metrics["high_count"]
        low = metrics["low_count"]
        mean_val = metrics["mean"]
        hist_mean = metrics["hist_mean"]

        return f"""\U0001f52c **Analisis Teknis Anomali**
Parameter {{parameter}} menunjukkan {{metrics['count']}} anomali ({{metrics['percent']}}) dengan deviasi {{dev:.1f}}% terhadap nilai normal historis ({{hist_mean:.2f}}). Terdapat {{high}} kejadian nilai tinggi dan {{low}} kejadian nilai rendah yang menyimpang dari pola operasional normal. Deviasi sebesar {{abs(dev):.1f}}% ini mengindikasikan adanya gangguan signifikan yang memerlukan investigasi segera oleh tim operasi PLTU.

\u26a0\ufe0f **Kemungkinan Penyebab**

**1.** Gangguan pada sistem instrumentasi dan sensor pengukuran {{parameter}}
   - Sensor mengalami drift kalibrasi akibat paparan suhu tinggi jangka panjang
   - Koneksi wiring transmitter longgar atau terkorosi sehingga sinyal tidak stabil
   - Impulse line tersumbat atau bocor menyebabkan pembacaan tidak akurat
   - Zero/span adjustment pada transmitter bergeser dari nilai referensi

**2.** Perubahan kondisi operasional sistem yang mempengaruhi {{parameter}}
   - Fluktuasi beban pembangkit melebihi kemampuan respon sistem kontrol otomatis
   - Perubahan kualitas bahan bakar (ukuran partikel, kadar air, nilai kalor) yang tidak terduga
   - Gangguan pada sistem kontrol DCS/PLC terkait loop kontrol parameter ini
   - Interaksi antar parameter operasional yang memperburuk kondisi sistem

**3.** Degradasi atau kerusakan mekanis pada peralatan terkait {{parameter}}
   - Fouling atau deposisi abu pada komponen yang mempengaruhi aliran/tekanan
   - Keausan komponen bergerak (bearing, seal, impeller) mendekati batas usia pakai
   - Kebocoran internal atau eksternal yang mengurangi efisiensi sistem secara bertahap
   - Vibrasi berlebihan yang menyebabkan ketidakstabilan pembacaan sensor

**4.** Faktor eksternal dan kondisi lingkungan operasi
   - Perubahan kondisi ambient (suhu udara, kelembaban) yang mempengaruhi performa sistem
   - Gangguan pada sistem pendingin atau pelumasan komponen terkait
   - Variasi kualitas air umpan atau steam yang mempengaruhi proses termal
   - Interaksi dengan sistem auxiliary lain yang tidak terdeteksi sebelumnya

\U0001f6e0\ufe0f **Rekomendasi Tindak Lanjut**

**1.** Verifikasi dan kalibrasi ulang instrumentasi pengukuran {{parameter}}
   - Lakukan cross-check pembacaan dengan portable instrument standar tersertifikasi
   - Periksa kondisi fisik sensor, transmitter, dan impulse line secara visual menyeluruh
   - Kalibrasi ulang menggunakan alat standar sesuai prosedur pabrikan
   - Dokumentasikan hasil kalibrasi dan bandingkan dengan baseline historis sebelumnya

**2.** Inspeksi visual dan pemeriksaan kondisi peralatan terkait {{parameter}}
   - Lakukan walkthrough inspection pada semua komponen terkait parameter ini
   - Periksa kondisi isolasi termal, gasket, dan flange connection di area terkait
   - Ukur vibrasi dan temperatur bearing menggunakan thermal camera dan vibration meter
   - Catat semua temuan dalam maintenance logbook untuk analisis tren jangka panjang

**3.** Review dan optimasi parameter setting sistem kontrol
   - Evaluasi setpoint dan tuning parameter PID controller yang terkait {{parameter}}
   - Bandingkan trend data historis 30, 60, dan 90 hari terakhir untuk pola anomali
   - Konsultasikan dengan vendor OEM jika diperlukan adjustment parameter kontrol
   - Lakukan simulasi operasi pada beban berbeda untuk validasi respons sistem

**4.** Implementasi monitoring intensif dan rencana tindak lanjut
   - Tingkatkan frekuensi pembacaan manual dari 1x/shift menjadi setiap 2 jam
   - Set alarm batas atas/bawah yang lebih ketat di DCS untuk deteksi dini anomali
   - Siapkan prosedur contingency jika anomali berlanjut atau kondisi memburuk
   - Jadwalkan inspeksi mendalam pada planned maintenance berikutnya sesuai WO

\U0001f4a5 **Potensi Dampak**
Anomali berkelanjutan pada {{parameter}} dengan deviasi {{abs(dev):.1f}}% berpotensi menurunkan efisiensi termal pembangkit secara signifikan dan meningkatkan konsumsi bahan bakar spesifik (heat rate). Jika tidak ditangani, kondisi ini dapat mempercepat keausan komponen terkait dan berujung pada forced outage yang tidak terencana. Dalam jangka panjang, operasi di luar batas normal parameter ini berisiko menyebabkan kerusakan permanen pada peralatan utama dengan biaya perbaikan besar dan downtime produksi signifikan.

*\u26a0\ufe0f Catatan: Analisis fallback ditampilkan - koneksi AI terputus. Error: {{str(e)[:80]}}*"""



def get_prediction_insight(input_param, input_value, results, correlations, method):
    insight = f"""
### 📈 Interpretasi Hasil Prediksi
- Parameter input: **{input_param}** = {input_value:.2f}
- Metode: {method}
"""
    if results:
        best_param = max(results.items(), key=lambda x: x[1]['r_squared'])
        insight += f"\n- Prediksi terbaik: **{best_param[0]}** = {best_param[1]['prediction']:.2f} (R²={best_param[1]['r_squared']:.3f})"
    if correlations:
        insight += "\n\n### 🔗 Korelasi Utama\n"
        for param, corr_value, direction, strength in correlations:
            insight += f"- {input_param} vs {param}: {direction}, {strength} (r={corr_value:.3f})\n"
    return insight


def save_data(sheet_dict):
    try:
        with open("uploaded_data.pkl", "wb") as f:
            pickle.dump(sheet_dict, f)
        st.session_state.persistent_data = sheet_dict
    except Exception as e:
        st.error(f"Gagal menyimpan data: {str(e)}")


def load_data():
    if 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        return st.session_state.persistent_data
    try:
        if os.path.exists("uploaded_data.pkl"):
            with open("uploaded_data.pkl", "rb") as f:
                data = pickle.load(f)
                st.session_state.persistent_data = data
                return data
        return None
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None


def clear_saved_data():
    try:
        if os.path.exists("uploaded_data.pkl"):
            os.remove("uploaded_data.pkl")
        if 'persistent_data' in st.session_state:
            del st.session_state.persistent_data
        for key in ["df", "sheet_names", "sheet_dict", "processed_df"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Data berhasil dihapus")
    except Exception as e:
        st.error(f"Gagal menghapus data: {str(e)}")


# =============================================
# TEMA DARK MODE
# =============================================

st.markdown("""
<style>
.stApp { background-color: #121212; color: white; }
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #FFA500;
}
.metric-value { color: #FFA500; font-size: 1.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

dark_template = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': 'white'},
    }
}

ensure_persistent_data()

# =============================================
# NAVIGASI SIDEBAR
# =============================================

with st.sidebar:
    st.image(
        "https://gajiloker.com/wp-content/uploads/2024/02/Gaji-PT-PLN-Nusantara-Power-Services.jpg",
        width=250
    )
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Home", "Data Collecting and Visualitation", "Machine Learning"],
        icons=["house", "bar-chart", "cpu"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1E1E1E"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#2E2E2E"},
            "nav-link-selected": {"background-color": "#FFA500"},
        }
    )
    if st.button("🗑️ Hapus Semua Data", help="Hapus semua data yang telah diunggah", type="primary"):
        clear_saved_data()

# =============================================
# HALAMAN HOME
# =============================================
if selected == "Home":
    st.markdown("""
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }
    .title-container {
        background: linear-gradient(135deg, #1a2a6c, #2a4858, #003366);
        padding: 30px 20px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border-left: 5px solid #4facfe;
    }
    .main-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ffffff, #4facfe, #ffffff);
        background-size: 200% auto;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s linear infinite;
    }
    .subtitle {
        color: #e6e6e6;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 15px;
    }
    .info-container {
        background: rgba(30,30,30,0.9);
        padding: 25px;
        border-radius: 12px;
        border-left: 4px solid #4facfe;
        margin-top: 20px;
    }
    </style>

    <div class="title-container">
        <h1 class="main-title">📈 AI - PLTU ANGGREK</h1>
        <p class="subtitle">DIGIT-OPS: Deteksi Anomali & Optimasi Kinerja PLTU Anggrek</p>
    </div>
    """, unsafe_allow_html=True)

    placeholder = st.empty()
    full_text = "Selamat Datang Di Artificial Intelligence PLTU Anggrek "
    for i in range(len(full_text) + 1):
        placeholder.markdown(f"### {full_text[:i]}_")
        time.sleep(0.03)
    placeholder.markdown(f"### {full_text}")

    time.sleep(0.5)
    st.markdown("""
    <div class="info-container">
        <h3>OPERATION PLTU ANGGREK</h3>
        <p>Sistem ini menyediakan:</p>
        <ul>
            <li>Visualisasi parameter operasional</li>
            <li>Analisis performa pembangkit</li>
            <li>Deteksi anomali menggunakan machine learning</li>
            <li>Monitoring kesiapan peralatan</li>
        </ul>
        <p>Silakan pilih menu di sidebar untuk mengakses fitur yang tersedia.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================
# HALAMAN DATA COLLECTING AND VISUALITATION
# =============================================

elif selected == "Data Collecting and Visualitation":
    st.title("📊 Data Collecting and Visualitation")

    if 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        st.session_state.sheet_dict = st.session_state.persistent_data
        st.session_state.sheet_names = list(st.session_state.sheet_dict.keys())
        st.session_state.df = list(st.session_state.sheet_dict.values())[0]
        st.success("✅ Data berhasil dimuat dari sesi sebelumnya!")

    uploaded_file = st.file_uploader("📄 Upload file data (CSV atau Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.session_state.sheet_dict = {"Sheet1": df}
            else:
                xls = pd.ExcelFile(uploaded_file)
                sheet_dict = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
                st.session_state.sheet_dict = sheet_dict

            st.session_state.sheet_names = list(st.session_state.sheet_dict.keys())
            st.session_state.df = list(st.session_state.sheet_dict.values())[0]
            save_data(st.session_state.sheet_dict)
            st.success("✅ Data berhasil diunggah!")
        except Exception as e:
            st.error(f"Gagal memuat file: {str(e)}")

    if "sheet_dict" not in st.session_state:
        st.info("Silakan upload file terlebih dahulu.")
        st.stop()

    selected_sheet = st.radio("📄 Pilih Sheet:", st.session_state.sheet_names, horizontal=True)
    df = st.session_state.sheet_dict[selected_sheet]
    st.session_state.df = df

    date_column = None
    for col in df.columns:
        if pd.to_datetime(df[col], errors='coerce').notna().all():
            date_column = col
            df[date_column] = pd.to_datetime(df[date_column])
            df['date_display'] = df[date_column].dt.strftime('%Y-%m-%d %H:%M')
            break

    columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_param = st.selectbox("Pilih Parameter:", options=columns, index=0, key="param_selectbox")
    st.markdown(f"**Parameter yang dipilih**: <span style='color:#FFA500'>{selected_param}</span>", unsafe_allow_html=True)

    st.markdown("### 📊 Statistik Deskriptif")
    stats = df[selected_param].describe()
    cols = st.columns(6)
    metrics = [
        ('Minimum', stats['min']),
        ('Q1', stats['25%']),
        ('Median', stats['50%']),
        ('Mean', stats['mean']),
        ('Q3', stats['75%']),
        ('Maximum', stats['max'])
    ]
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div>{label}</div>
                <div class="metric-value">{value:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📉 Grafik Tren")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        line_color = st.color_picker("Pilih Warna Garis:", value='#FFA500')
    with col2:
        show_ma = st.checkbox("Tampilkan Moving Average", value=True)

    if date_column:
        df_sorted = df.sort_values(by=date_column).copy()
        x_axis = date_column
    else:
        x_axis = df.index
        df_sorted = df.copy()

    fig_line = px.line(
        df_sorted, x=x_axis, y=selected_param,
        title=f"<b>Tren {selected_param}</b>",
        template='plotly_dark',
        color_discrete_sequence=[line_color]
    )

    if show_ma:
        df_sorted['MA_7'] = df_sorted[selected_param].rolling(window=7).mean()
        fig_line.add_scatter(
            x=df_sorted[x_axis], y=df_sorted['MA_7'],
            name='Moving Avg (7)', line=dict(color='#00FFFF', width=2, dash='dot')
        )

    fig_line.update_layout(**dark_template['layout'])
    st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")

    st.markdown("### 📊 Histogram")
    fig_hist = px.histogram(df, x=selected_param, template='plotly_dark', color_discrete_sequence=['#FFA500'])
    st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

    st.markdown("### 📦 Box Plot")
    fig_box = px.box(df, y=selected_param, template='plotly_dark', color_discrete_sequence=['#FFA500'])
    st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")

    if st.button("🤖 Analisa AI PLTU ANGGREK", key="ai_analysis_button"):
        with st.spinner("⏳ Menganalisis data dengan AI..."):
            st.markdown("#### 🤖 Insight PLTU Anggrek")
            st.markdown("##### 📈 Analisis Grafik Tren")
            ai_result = get_ai_insight(selected_param, df[selected_param].dropna(), "Grafik Tren")
            st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>{ai_result}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### 📊 Analisis Histogram")
            ai_result_hist = get_ai_insight(selected_param, df[selected_param].dropna(), "Histogram")
            st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #00FFFF;'>{ai_result_hist}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("##### 📦 Analisis Box Plot")
            ai_result_box = get_ai_insight(selected_param, df[selected_param].dropna(), "Box Plot")
            st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FF69B4;'>{ai_result_box}</div>", unsafe_allow_html=True)


# =============================================
# HALAMAN MACHINE LEARNING
# =============================================

elif selected == "Machine Learning":
    st.title("🤖 Machine Learning PLTU Anggrek")

    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = {}
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""

    if "df" not in st.session_state and 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        sheet_dict = st.session_state.persistent_data
        st.session_state.df = list(sheet_dict.values())[0]

    if "df" not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu Data Collecting.")
        st.stop()

    df = st.session_state.df.copy()

    date_column = None
    for col in df.columns:
        if pd.to_datetime(df[col], errors='coerce').notna().all():
            date_column = col
            df[date_column] = pd.to_datetime(df[date_column])
            break

    if not date_column:
        st.error("Data tidak memiliki kolom tanggal yang valid.")
        st.stop()

    df = df.sort_values(date_column)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Data Preparation",
        "📊 Exploration Data Analysis",
        "🔍 Anomaly Detection And AI Evaluation",
        "🎯 Prediction Analysis",
        "📈 Forecasting"
    ])

    # ===== TAB 1: DATA PREPARATION =====
    with tab1:
        st.header("Persiapan Data")

        st.subheader("1. Pilih Parameter Target untuk Pemfilteran Data")
        filter_param = st.selectbox(
            "Pilih Parameter untuk Pemfilteran Data:",
            options=df.select_dtypes(include=['number']).columns.tolist(),
            key="filter_param_select"
        )

        filter_threshold = st.number_input(
            "Nilai Minimum Data yang Valid:", value=5.0,
            help="Data dengan nilai di bawah threshold ini akan ditandai untuk preprocessing"
        )

        filtered_rows = df[df[filter_param] < filter_threshold].shape[0]
        missing_rows = df[filter_param].isna().sum()
        total_rows_to_process = filtered_rows + missing_rows
        st.info(f"Terdapat {filtered_rows} baris dengan nilai {filter_param} < {filter_threshold} dan {missing_rows} baris kosong. Total {total_rows_to_process} baris perlu diproses.")

        st.subheader("Distribusi Nilai Parameter Filter")
        fig_filter = px.histogram(df, x=filter_param, template='plotly_dark',
                                   color_discrete_sequence=['#FFA500'], title=f"Distribusi Nilai {filter_param}")
        fig_filter.add_vline(x=filter_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_filter, use_container_width=True)

        st.subheader("2. Pilih Parameter yang Akan Diproses")
        process_method = st.radio(
            "Metode Pemilihan Parameter:",
            options=["Proses Semua Parameter Numerik", "Pilih Parameter Spesifik"]
        )

        num_columns = df.select_dtypes(include=['number']).columns.tolist()
        if process_method == "Pilih Parameter Spesifik":
            selected_params = st.multiselect("Pilih Parameter:", options=num_columns, default=[filter_param])
        else:
            selected_params = num_columns
            st.info(f"Semua {len(num_columns)} parameter numerik akan diproses.")

        st.subheader("3. Tren Data Asli Parameter Utama")
        target_param = st.selectbox("Pilih Parameter Utama:", options=selected_params, index=0, key="target_param_select")

        fig_original = px.line(df, x=date_column, y=target_param,
                                title=f"Tren Original {target_param}", template='plotly_dark',
                                color_discrete_sequence=['#FFA500'])
        mask_below_threshold = df[filter_param] < filter_threshold
        if filter_param == target_param:
            fig_original.add_scatter(
                x=df[mask_below_threshold][date_column], y=df[mask_below_threshold][target_param],
                mode='markers', marker=dict(color='red', size=10, symbol='x'), name=f'Nilai < {filter_threshold}'
            )
        fig_original.update_layout(height=400)
        st.plotly_chart(fig_original, use_container_width=True)

        st.subheader("4. Pilih Metode Preprocessing")
        preprocessing_options = st.multiselect(
            "Teknik yang akan diterapkan:",
            options=["Imputasi Missing Values", "Smoothing (Moving Average)", "Normalisasi (Min-Max)", "Standarisasi (Z-Score)", "Detrending"],
            default=["Imputasi Missing Values"]
        )

        processing_params = {}
        if "Imputasi Missing Values" in preprocessing_options:
            imputation_method = st.radio(
                "Metode Imputasi:",
                options=["Mean", "Median", "Forward Fill", "Backward Fill", "Linear Interpolation"],
                horizontal=True
            )
            processing_params['imputation_method'] = imputation_method
        if "Smoothing (Moving Average)" in preprocessing_options:
            processing_params['window_size'] = st.slider("Window Size Smoothing:", 3, 14, 7)
        if "Detrending" in preprocessing_options:
            processing_params['trend_window'] = st.slider("Window Size Detrending:", 7, 30, 14)

        if st.button("🛠️ Proses Data", key="process_button"):
            with st.spinner("Memproses data..."):
                processed_data = df.copy()
                mask_to_process = (processed_data[filter_param] < filter_threshold) | (processed_data[filter_param].isna())
                total_rows_processed = mask_to_process.sum()
                total_params_processed = len(selected_params)
                st.info(f"Memproses {total_rows_processed} baris × {total_params_processed} parameter")

                progress_bar = st.progress(0)
                method = processing_params.get('imputation_method', 'Mean')

                for i, param in enumerate(selected_params):
                    progress_bar.progress(int((i + 1) / len(selected_params) * 100))

                    if "Imputasi Missing Values" in preprocessing_options:
                        if method == "Mean":
                            fill_value = processed_data[param].mean()
                        elif method == "Median":
                            fill_value = processed_data[param].median()
                        else:
                            fill_value = processed_data[param].mean()

                        if method in ["Mean", "Median"]:
                            if param == filter_param:
                                mask_below = (processed_data[param] < filter_threshold) & ~processed_data[param].isna()
                                processed_data.loc[mask_below, param] = fill_value
                            processed_data.loc[mask_to_process, param] = processed_data.loc[mask_to_process, param].fillna(fill_value)
                        elif method == "Forward Fill":
                            temp_series = processed_data[param].copy()
                            processed_data.loc[mask_to_process, param] = temp_series.ffill().loc[mask_to_process]
                        elif method == "Backward Fill":
                            temp_series = processed_data[param].copy()
                            processed_data.loc[mask_to_process, param] = temp_series.bfill().loc[mask_to_process]
                        elif method == "Linear Interpolation":
                            temp_df = processed_data[[date_column, param]].copy()
                            if param == filter_param:
                                mask_below = (temp_df[param] < filter_threshold) & ~temp_df[param].isna()
                                temp_df.loc[mask_below, param] = np.nan
                            temp_df[param] = temp_df[param].interpolate(method='linear')
                            for idx in processed_data[mask_to_process].index:
                                if idx in temp_df.index:
                                    processed_data.loc[idx, param] = temp_df.loc[idx, param]

                    if param == target_param:
                        if "Smoothing (Moving Average)" in preprocessing_options:
                            window = processing_params.get('window_size', 7)
                            processed_data[f'{param}_smooth'] = processed_data[param].rolling(window=window, min_periods=1).mean()
                        if "Normalisasi (Min-Max)" in preprocessing_options:
                            scaler = MinMaxScaler()
                            values = processed_data[param].values.reshape(-1, 1)
                            processed_data[f'{param}_norm'] = scaler.fit_transform(values)
                        if "Standarisasi (Z-Score)" in preprocessing_options:
                            scaler_std = StandardScaler()
                            values = processed_data[param].values.reshape(-1, 1)
                            processed_data[f'{param}_std'] = scaler_std.fit_transform(values)
                        if "Detrending" in preprocessing_options:
                            window = processing_params.get('trend_window', 14)
                            trend = processed_data[param].rolling(window=window, min_periods=1).mean()
                            processed_data[f'{param}_detrend'] = processed_data[param] - trend

                st.session_state.processed_data = processed_data
                st.session_state.target_param = target_param
                st.session_state.date_column = date_column
                st.session_state.full_data = processed_data.copy()
                st.session_state.filter_param = filter_param
                st.session_state.filter_threshold = filter_threshold
                st.session_state.mask_to_process = mask_to_process

                st.success(f"✅ Preprocessing selesai! {total_rows_processed} baris diproses.")

                viz_columns = [target_param] + [col for col in processed_data.columns if col.startswith(f"{target_param}_")]
                fig_processed = px.line(processed_data, x=date_column, y=viz_columns,
                                         title=f"Perbandingan Hasil Preprocessing {target_param}",
                                         template='plotly_dark')
                fig_processed.update_layout(height=500)
                st.plotly_chart(fig_processed, use_container_width=True)

                with st.expander("Lihat Data Hasil Preprocessing"):
                    st.dataframe(processed_data[viz_columns + [date_column]])

    # ===== TAB 2: EDA =====
    with tab2:
        st.header("📊 Exploration Data Analysis")

        if "processed_data" not in st.session_state:
            st.warning("Lakukan data preparation terlebih dahulu.")
            st.stop()

        try:
            processed_data = st.session_state.processed_data.copy()
            target_param = st.session_state.target_param
            date_column = st.session_state.date_column
        except Exception as e:
            st.error(f"Error mengakses data: {str(e)}")
            st.stop()

        cols_with_processed = [c for c in processed_data.columns if c != date_column]
        processed_cols = [c for c in cols_with_processed if c.startswith(f"{target_param}_")]
        default_col = processed_cols[0] if processed_cols else target_param

        analysis_col = st.selectbox(
            "Pilih Kolom Utama untuk Analisis:",
            options=cols_with_processed,
            index=cols_with_processed.index(default_col) if default_col in cols_with_processed else 0,
            key="eda_col_select"
        )

        analysis_col2 = st.selectbox(
            "Pilih Parameter Kedua (Opsional):",
            options=["None"] + cols_with_processed, index=0, key="eda_col_select2"
        )

        st.markdown("### 📊 Statistik Deskriptif")
        try:
            stats_desc = processed_data[analysis_col].describe()
            cols = st.columns(6)
            metrics = [
                ('Minimum', stats_desc['min']), ('Q1', stats_desc['25%']),
                ('Median', stats_desc['50%']), ('Mean', stats_desc['mean']),
                ('Q3', stats_desc['75%']), ('Maximum', stats_desc['max'])
            ]
            for i, (label, value) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div style='background-color:#1E1E1E; padding:10px; border-radius:5px; text-align:center;'>
                        <div style='font-size:12px;'>{label}</div>
                        <div style='font-size:16px; font-weight:bold; color:#FFA500;'>{value:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error statistik: {str(e)}")

        st.markdown("### 📈 Grafik Tren")
        try:
            y_cols = [analysis_col] if analysis_col2 == "None" else [analysis_col, analysis_col2]
            fig_line = px.line(processed_data, x=date_column, y=y_cols,
                                title=f"Tren {' & '.join(y_cols)}", template='plotly_dark',
                                color_discrete_sequence=['#FFA500', '#00FF7F'])
            show_ma_eda = st.checkbox("Tampilkan Moving Average", value=True, key="eda_show_ma")
            if show_ma_eda and analysis_col2 == "None":
                temp_data = processed_data.copy()
                temp_data['MA_7'] = temp_data[analysis_col].rolling(window=7).mean()
                fig_line.add_scatter(x=temp_data[date_column], y=temp_data['MA_7'],
                                     name='Moving Avg (7)', line=dict(color='#00FFFF', width=2, dash='dot'))
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.error(f"Error grafik tren: {str(e)}")

        st.markdown("### 📊 Histogram")
        try:
            fig_hist = px.histogram(processed_data, x=analysis_col, template='plotly_dark',
                                     color_discrete_sequence=['#FFA500'])
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Error histogram: {str(e)}")

        st.markdown("### 📦 Box Plot")
        try:
            fig_box = px.box(processed_data, y=analysis_col, template='plotly_dark',
                              color_discrete_sequence=['#FFA500'])
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"Error box plot: {str(e)}")

        st.markdown("### 🔗 Correlation Matrix")
        try:
            numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
            if date_column in numeric_cols:
                numeric_cols.remove(date_column)
            if len(numeric_cols) > 1:
                corr_matrix = processed_data[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                      color_continuous_scale='RdBu_r', template='plotly_dark',
                                      title='Korelasi Antar Parameter')
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"Error correlation matrix: {str(e)}")

        st.markdown("### 🌊 Density Plot (KDE)")
        try:
            fig_kde = px.histogram(processed_data, x=analysis_col, nbins=40, histnorm='density',
                                    marginal="box", template='plotly_dark', color_discrete_sequence=['#FF4500'])
            st.plotly_chart(fig_kde, use_container_width=True)
        except Exception as e:
            st.error(f"Error density plot: {str(e)}")

        st.markdown("### 🎻 Violin Plot")
        try:
            fig_violin = px.violin(processed_data, y=analysis_col, box=True, points="all",
                                    template="plotly_dark", color_discrete_sequence=['#32CD32'])
            st.plotly_chart(fig_violin, use_container_width=True)
        except Exception as e:
            st.error(f"Error violin plot: {str(e)}")

        st.markdown("### 📊 Rolling Statistics")
        try:
            temp_df = processed_data.copy()
            temp_df['Rolling_Mean'] = temp_df[analysis_col].rolling(window=7).mean()
            temp_df['Rolling_Std'] = temp_df[analysis_col].rolling(window=7).std()
            fig_roll = px.line(temp_df, x=date_column,
                                y=[analysis_col, 'Rolling_Mean', 'Rolling_Std'],
                                template="plotly_dark", title=f"Rolling Statistics - {analysis_col}")
            st.plotly_chart(fig_roll, use_container_width=True)
        except Exception as e:
            st.error(f"Error rolling statistics: {str(e)}")

        if analysis_col2 != "None":
            st.markdown("### 🔎 Scatter Plot Antar 2 Parameter")
            try:
                fig_scatter = px.scatter(processed_data, x=analysis_col, y=analysis_col2,
                                          template="plotly_dark", color_discrete_sequence=['#FF69B4'],
                                          title=f"{analysis_col} vs {analysis_col2}")
                corr_coef = processed_data[[analysis_col, analysis_col2]].corr().iloc[0, 1]
                fig_scatter.add_annotation(text=f"Correlation: {corr_coef:.3f}",
                                            xref="paper", yref="paper", x=0.02, y=0.98,
                                            showarrow=False, font=dict(size=14, color="white"),
                                            bgcolor="rgba(0,0,0,0.7)", bordercolor="#FF69B4", borderwidth=1)
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Error scatter plot: {str(e)}")

        st.markdown("---")
        if st.button("🤖 Analisa AI PLTU ANGGREK", key="eda_ai_analysis"):
            with st.spinner("⏳ Menganalisis data dengan AI..."):
                clean_data = processed_data[analysis_col].dropna()
                ai_result = get_ai_insight(analysis_col, clean_data, "EDA Analysis")
                st.markdown(f"""
                <div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>
                    {ai_result}
                </div>
                """, unsafe_allow_html=True)

    # ===== TAB 3: ANOMALY DETECTION =====
    with tab3:
        st.header("Deteksi Anomali (30 Hari Terakhir vs Pola Historis)")

        if "processed_data" not in st.session_state:
            st.warning("Lakukan data preparation terlebih dahulu.")
            st.stop()

        full_data = st.session_state.full_data
        target_param = st.session_state.target_param
        date_column = st.session_state.date_column

        last_date = full_data[date_column].max()
        start_date = last_date - pd.Timedelta(days=30)
        df_last30 = full_data[full_data[date_column] >= start_date].copy()

        st.markdown(f"### Analisis 30 Hari Terakhir ({start_date.strftime('%Y-%m-%d')} hingga {last_date.strftime('%Y-%m-%d')})")

        cols_with_processed = [c for c in full_data.columns if c != date_column]
        processed_cols = [c for c in cols_with_processed if c.startswith(f"{target_param}_")]
        default_col = processed_cols[0] if processed_cols else target_param

        analysis_col = st.selectbox(
            "Pilih Kolom untuk Analisis Anomali:",
            options=cols_with_processed,
            index=cols_with_processed.index(default_col) if default_col in cols_with_processed else 0,
            key="anomaly_col_select"
        )

        method = st.selectbox(
            "Metode Deteksi Anomali:",
            options=[
                "Threshold-based (IQR Historis)",
                "Isolation Forest (Pola Historis)",
                "Support Vector Machine (SVM)",
                "Adaptive Z-Score (Bulan Sebelumnya)",
            ]
        )

        params = {}
        if method == "Threshold-based (IQR Historis)":
            params['threshold'] = st.slider("Threshold IQR:", 1.0, 3.0, 1.5, step=0.1)
        elif method == "Isolation Forest (Pola Historis)":
            params['contamination'] = st.slider("Estimasi Kontaminasi:", 0.01, 0.2, 0.05, step=0.01)
        elif method == "Support Vector Machine (SVM)":
            params['nu'] = st.slider("Nu:", 0.01, 0.2, 0.05, step=0.01)
            params['kernel'] = st.selectbox("Kernel:", options=["rbf", "linear", "poly", "sigmoid"], index=0)
        elif method == "Adaptive Z-Score (Bulan Sebelumnya)":
            params['z_threshold'] = st.slider("Threshold Z-Score:", 2.0, 5.0, 3.0)
            params['window'] = st.slider("Window Size (hari):", 7, 30, 14)

        if st.button("🔍 Deteksi Anomali", key="detect_button"):
            with st.spinner("Mendeteksi anomali..."):
                try:
                    historical_data = full_data[full_data[date_column] < start_date]

                    if method == "Threshold-based (IQR Historis)":
                        Q1 = full_data[analysis_col].quantile(0.25)
                        Q3 = full_data[analysis_col].quantile(0.75)
                        IQR = Q3 - Q1
                        threshold_val = params['threshold']
                        lower_bound = Q1 - (threshold_val * IQR)
                        upper_bound = Q3 + (threshold_val * IQR)
                        df_last30['Anomaly'] = ((df_last30[analysis_col] < lower_bound) |
                                                 (df_last30[analysis_col] > upper_bound)).astype(int)

                    elif method == "Isolation Forest (Pola Historis)":
                        model = IsolationForest(contamination=params['contamination'], random_state=42)
                        model.fit(full_data[[analysis_col]])
                        df_last30['Anomaly'] = model.predict(df_last30[[analysis_col]])
                        df_last30['Anomaly'] = df_last30['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

                    elif method == "Support Vector Machine (SVM)":
                        from sklearn.svm import OneClassSVM
                        scaler = StandardScaler()
                        scaler.fit(historical_data[[analysis_col]])
                        X_train_scaled = scaler.transform(historical_data[[analysis_col]])
                        X_test_scaled = scaler.transform(df_last30[[analysis_col]])
                        svm_model = OneClassSVM(nu=params['nu'], kernel=params['kernel'], gamma='scale')
                        svm_model.fit(X_train_scaled)
                        df_last30['Anomaly'] = svm_model.predict(X_test_scaled)
                        df_last30['Anomaly'] = df_last30['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

                    elif method == "Adaptive Z-Score (Bulan Sebelumnya)":
                        threshold_val = params['z_threshold']
                        if len(historical_data) > 0:
                            prev_month_mean = historical_data[analysis_col].mean()
                            prev_month_std = historical_data[analysis_col].std()
                            df_last30['z_score'] = (df_last30[analysis_col] - prev_month_mean) / prev_month_std
                            df_last30['Anomaly'] = (df_last30['z_score'].abs() > threshold_val).astype(int)
                        else:
                            st.warning("Tidak cukup data historis")
                            df_last30['Anomaly'] = 0

                    st.session_state.anomaly_results = df_last30

                    fig = px.line(df_last30, x=date_column, y=analysis_col,
                                   title=f"Anomali Terdeteksi ({method}) - 30 Hari Terakhir",
                                   template='plotly_dark')

                    anomalies = df_last30[df_last30['Anomaly'] == 1]
                    if len(anomalies) > 0:
                        fig.add_scatter(x=anomalies[date_column], y=anomalies[analysis_col],
                                        mode='markers', name='Anomaly',
                                        marker=dict(color='red', size=8))

                    if method == "Threshold-based (IQR Historis)":
                        fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                      annotation_text=f"Upper Bound")
                        fig.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                                      annotation_text=f"Lower Bound")

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Total Anomali Terdeteksi:** {df_last30['Anomaly'].sum()}")

                    if len(anomalies) > 0:
                        st.markdown("### Detail Anomali")
                        st.dataframe(anomalies[[date_column, analysis_col]].sort_values(date_column), height=300)

                        with st.expander("🤖 Insight Deteksi Anomali"):
                            with st.spinner("Menganalisis anomali..."):
                                historical_stats = {
                                    'mean': historical_data[analysis_col].mean(),
                                    'std': historical_data[analysis_col].std(),
                                    'min': historical_data[analysis_col].min(),
                                    'max': historical_data[analysis_col].max(),
                                }

                                high_anomalies = anomalies[anomalies[analysis_col] > historical_stats['mean']].shape[0]
                                low_anomalies = anomalies[anomalies[analysis_col] < historical_stats['mean']].shape[0]

                                anomaly_metrics = {
                                    'count': len(anomalies),
                                    'percent': f"{(len(anomalies) / len(df_last30) * 100):.2f}%",
                                    'high_count': high_anomalies,
                                    'low_count': low_anomalies,
                                    'mean': anomalies[analysis_col].mean(),
                                    'hist_mean': historical_stats['mean'],
                                    'deviation': ((anomalies[analysis_col].mean() - historical_stats['mean']) / historical_stats['mean'] * 100)
                                    if historical_stats['mean'] != 0 else 0
                                }

                                numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
                                correlation_data = []
                                if len(numeric_cols) > 1 and analysis_col in numeric_cols:
                                    corr_matrix = full_data[numeric_cols].corr()
                                    if analysis_col in corr_matrix.columns:
                                        target_corrs = corr_matrix[analysis_col].drop(analysis_col)
                                        sorted_corrs = target_corrs.abs().sort_values(ascending=False)
                                        for param, corr_value in target_corrs[sorted_corrs.head(3).index].items():
                                            strength = "sangat kuat" if abs(corr_value) > 0.7 else "kuat" if abs(corr_value) > 0.5 else "sedang" if abs(corr_value) > 0.3 else "lemah"
                                            direction = "positif" if corr_value > 0 else "negatif"
                                            explanation = f"Peningkatan {param} diikuti peningkatan {analysis_col}" if corr_value > 0 else f"Peningkatan {param} diikuti penurunan {analysis_col}"
                                            correlation_data.append((param, corr_value, direction, strength, explanation))

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Anomali", anomaly_metrics['count'])
                                with col2:
                                    st.metric("Persentase", anomaly_metrics['percent'])
                                with col3:
                                    dev = anomaly_metrics['deviation']
                                    st.metric("Deviasi Rata-rata", f"{dev:.1f}%")

                                ai_analysis = get_anomaly_insight(
                                    parameter=analysis_col,
                                    anomaly_data=anomalies,
                                    method=method,
                                    metrics=anomaly_metrics,
                                    correlation_data=correlation_data
                                )
                                st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>{ai_analysis}</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error dalam deteksi anomali: {str(e)}")

        if st.session_state.get('anomaly_results') is not None:
            st.markdown("---")
            st.markdown("### 💬 Chat Analisis AI Lanjutan")

            user_prompt_input = st.text_area(
                "Ajukan pertanyaan spesifik tentang anomali:",
                placeholder="Contoh: 'Analisis penyebab anomali pada tanggal 15 Januari'",
                height=100, key="ai_chat_input"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apa penyebab utama anomali?", key="q1_button"):
                    st.session_state.user_prompt = "Apa penyebab utama anomali yang terdeteksi?"
                if st.button("Rekomendasi perbaikan?", key="q2_button"):
                    st.session_state.user_prompt = "Beri rekomendasi teknis untuk menangani anomali."
            with col2:
                if st.button("Dampak terhadap efisiensi?", key="q3_button"):
                    st.session_state.user_prompt = "Apa dampak anomali ini terhadap efisiensi pembangkit?"
                if st.button("Korelasi dengan parameter lain?", key="q4_button"):
                    st.session_state.user_prompt = "Analisis korelasi anomali ini dengan parameter lainnya."

            prompt_to_use = user_prompt_input or st.session_state.get('user_prompt', '')

            if st.button("🤖 Analisis dengan AI", key="chat_analysis_button") and prompt_to_use:
                with st.spinner("Menganalisis dengan AI..."):
                    try:
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=2000,
                            temperature=0.7,
                            messages=[{"role": "user", "content": f"""
Kamu adalah ahli analisis PLTU. Jawab pertanyaan berikut:

PERTANYAAN: {prompt_to_use}

KONTEKS: Parameter {analysis_col}, metode {method}, 30 hari terakhir.
"""}]
                        )
                        ai_response = response.content[0].text
                        st.markdown("### 🔍 Hasil Analisis AI")
                        st.markdown(f"<div style='background-color:#1E1E1E; padding:20px; border-radius:10px; border-left:4px solid #4facfe;'>{ai_response}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error AI: {str(e)}")

    # ===== TAB 4: PREDICTION ANALYSIS =====
    with tab4:
        st.header("🔮 Prediksi Parameter Berdasarkan Input")

        if "processed_data" not in st.session_state:
            st.warning("Lakukan data preparation terlebih dahulu.")
            st.stop()

        try:
            full_data = st.session_state.full_data
            target_param = st.session_state.target_param

            numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Diperlukan setidaknya 2 parameter numerik.")
                st.stop()

            default_input_param = target_param if target_param in numeric_cols else numeric_cols[0]
            input_param = st.selectbox(
                "Pilih Parameter Input:",
                options=numeric_cols,
                index=numeric_cols.index(default_input_param) if default_input_param in numeric_cols else 0,
                key="pred_input_param"
            )

            param_min = float(full_data[input_param].min())
            param_max = float(full_data[input_param].max())
            param_mean = float(full_data[input_param].mean())

            input_value = st.slider(
                f"Nilai {input_param}:",
                min_value=param_min, max_value=param_max, value=param_mean,
                step=0.1 if (param_max - param_min) < 10 else 1.0,
                key="pred_input_value"
            )

            prediction_method = st.selectbox(
                "Metode Prediksi:",
                options=["Korelasi Linear Sederhana", "Random Forest Regression", "Gradient Boosting", "Neural Network"],
                key="pred_method"
            )

            n_estimators = 100
            if prediction_method in ["Random Forest Regression", "Gradient Boosting"]:
                n_estimators = st.slider("Jumlah Estimator:", 10, 200, 100, step=10, key="n_estimators_slider")

            if st.button("🎯 Lakukan Prediksi", key="predict_button"):
                with st.spinner("Melakukan prediksi..."):
                    try:
                        X = full_data[[input_param]]
                        results = {}

                        for output_param in numeric_cols:
                            if output_param == input_param:
                                continue
                            y = full_data[output_param]
                            valid_indices = ~(X[input_param].isna() | y.isna())
                            X_clean = X[valid_indices]
                            y_clean = y[valid_indices]

                            if len(X_clean) < 10:
                                continue

                            if prediction_method == "Korelasi Linear Sederhana":
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                                model.fit(X_clean, y_clean)
                                prediction = model.predict([[input_value]])[0]
                                r_squared = model.score(X_clean, y_clean)
                                results[output_param] = {'prediction': prediction, 'r_squared': r_squared, 'method': 'Linear'}

                            elif prediction_method == "Random Forest Regression":
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                model.fit(X_clean, y_clean)
                                prediction = model.predict([[input_value]])[0]
                                r_squared = model.score(X_clean, y_clean)
                                results[output_param] = {'prediction': prediction, 'r_squared': r_squared, 'method': 'Random Forest'}

                            elif prediction_method == "Gradient Boosting":
                                from sklearn.ensemble import GradientBoostingRegressor
                                model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
                                model.fit(X_clean, y_clean)
                                prediction = model.predict([[input_value]])[0]
                                r_squared = model.score(X_clean, y_clean)
                                results[output_param] = {'prediction': prediction, 'r_squared': r_squared, 'method': 'Gradient Boosting'}

                            elif prediction_method == "Neural Network":
                                from sklearn.neural_network import MLPRegressor
                                scaler_X = StandardScaler()
                                scaler_y = StandardScaler()
                                X_scaled = scaler_X.fit_transform(X_clean)
                                y_scaled = scaler_y.fit_transform(y_clean.values.reshape(-1, 1))
                                model = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu',
                                                     solver='adam', max_iter=500, random_state=42)
                                model.fit(X_scaled, y_scaled.ravel())
                                input_scaled = scaler_X.transform([[input_value]])
                                pred_scaled = model.predict(input_scaled)
                                prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                                r_squared = model.score(X_scaled, y_scaled)
                                results[output_param] = {'prediction': prediction, 'r_squared': r_squared, 'method': 'Neural Network'}

                        if results:
                            st.success("✅ Prediksi Berhasil!")
                            st.session_state["prediction_results"] = results
                            st.session_state["prediction_input"] = {
                                "input_param": input_param,
                                "input_value": input_value,
                                "method": prediction_method,
                                "numeric_cols": numeric_cols,
                                "full_data": full_data
                            }
                        else:
                            st.error("Tidak ada hasil prediksi.")

                    except Exception as e:
                        st.error(f"Error prediksi: {str(e)}")

        except Exception as e:
            st.error(f"Gagal menyiapkan data: {str(e)}")

        if "prediction_results" in st.session_state:
            results = st.session_state["prediction_results"]
            input_param = st.session_state["prediction_input"]["input_param"]
            input_value = st.session_state["prediction_input"]["input_value"]
            prediction_method = st.session_state["prediction_input"]["method"]
            numeric_cols = st.session_state["prediction_input"]["numeric_cols"]
            full_data = st.session_state["prediction_input"]["full_data"]

            result_data = [
                {'Parameter': param, 'Prediksi': f"{v['prediction']:.2f}", 'Akurasi (R²)': f"{v['r_squared']:.3f}", 'Metode': v['method']}
                for param, v in results.items()
            ]
            st.markdown(f"### Hasil Prediksi untuk {input_param} = {input_value:.2f}")
            st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

            st.markdown("### 📊 Visualisasi Prediksi")
            param_to_plot = st.selectbox("Pilih parameter:", options=list(results.keys()), key="viz_param")
            fig = px.scatter(full_data, x=input_param, y=param_to_plot, trendline="ols",
                              title=f"{input_param} vs {param_to_plot} (R²: {results[param_to_plot]['r_squared']:.3f})",
                              template="plotly_dark")
            fig.add_vline(x=input_value, line_dash="dash", line_color="red",
                          annotation_text=f"Input: {input_value:.2f}")
            fig.add_scatter(x=[input_value], y=[results[param_to_plot]['prediction']],
                            mode='markers', marker=dict(color='red', size=10), name='Prediksi')
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🤖 Interpretasi AI untuk Hasil Prediksi"):
                try:
                    corr_matrix = full_data[numeric_cols].corr()
                    top_correlations = []
                    if input_param in corr_matrix.columns:
                        correlations = corr_matrix[input_param].drop(input_param)
                        sorted_corrs = correlations.abs().sort_values(ascending=False)
                        for param, corr_value in correlations[sorted_corrs.head(3).index].items():
                            direction = "positif" if corr_value > 0 else "negatif"
                            strength = "sangat kuat" if abs(corr_value) > 0.7 else "kuat" if abs(corr_value) > 0.5 else "sedang" if abs(corr_value) > 0.3 else "lemah"
                            top_correlations.append((param, corr_value, direction, strength))
                    ai_insight = get_prediction_insight(input_param, input_value, results, top_correlations, prediction_method)
                    st.markdown(ai_insight)
                except Exception as e:
                    st.error(f"Gagal membuat insight: {str(e)}")

    # ===== TAB 5: FORECASTING =====
    with tab5:
        st.header("📈 Forecasting Parameter")

        if "processed_data" not in st.session_state:
            st.warning("⚠️ Lakukan data preparation terlebih dahulu.")
            st.stop()

        try:
            full_data = st.session_state.full_data
            date_column = st.session_state.date_column
            full_data[date_column] = pd.to_datetime(full_data[date_column])

            numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                st.error("❌ Tidak ada parameter numerik untuk forecasting")
                st.stop()

            st.markdown("### 🔍 Preview Data")
            st.dataframe(full_data[[date_column] + numeric_cols].tail(10), use_container_width=True)

            target = st.selectbox("🎯 Pilih parameter untuk forecasting:", numeric_cols, key="forecast_param")
            horizon = st.slider("⏳ Jumlah langkah ke depan:", 5, 200, 20, key="forecast_horizon")
            method_fc = st.selectbox(
                "⚙️ Metode forecasting:",
                ["ARIMA", "Prophet", "LSTM (Deep Learning)"],
                key="forecast_method"
            )

            if st.button("🚀 Jalankan Forecasting", key="forecast_button"):
                with st.spinner("⏳ Sedang melakukan forecasting..."):
                    try:
                        series = full_data[[date_column, target]].dropna()
                        series.columns = ["ds", "y"]

                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt

                        if method_fc == "Prophet":
                            try:
                                from prophet import Prophet
                            except ImportError:
                                st.error("Prophet tidak tersedia. Install dengan: pip install prophet")
                                st.stop()

                            model = Prophet()
                            model.fit(series)
                            future = model.make_future_dataframe(periods=horizon, freq="30min")
                            forecast = model.predict(future)

                            plt.style.use("dark_background")
                            fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
                            ax.set_facecolor("black")
                            ax.plot(series['ds'], series['y'], color="orange", linewidth=1.5, alpha=0.7, label="Actual")
                            ax.plot(forecast['ds'], forecast['yhat'], color="cyan", linewidth=2.5, label="Forecast")
                            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                            color="deepskyblue", alpha=0.2, label="CI")
                            ax.set_title("📈 Forecasting (Prophet)", fontsize=16, color="white")
                            ax.legend(loc="lower right", frameon=False)
                            ax.tick_params(colors="white")
                            st.pyplot(fig)
                            plt.close(fig)
                            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))

                        elif method_fc == "ARIMA":
                            from statsmodels.tsa.arima.model import ARIMA
                            model = ARIMA(series["y"], order=(5, 1, 0))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=horizon)

                            forecast_df = pd.DataFrame({
                                "ds": pd.date_range(start=series["ds"].iloc[-1], periods=horizon + 1, freq="30min")[1:],
                                "forecast": forecast
                            })

                            plt.style.use("dark_background")
                            fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
                            ax.set_facecolor("black")
                            ax.plot(series["ds"], series["y"], color="orange", label="Actual")
                            ax.plot(forecast_df["ds"], forecast_df["forecast"], color="cyan", label="Forecast")
                            ax.set_title("📈 Forecasting (ARIMA)", fontsize=16, color="white")
                            ax.legend()
                            ax.tick_params(colors="white")
                            st.pyplot(fig)
                            plt.close(fig)
                            st.dataframe(forecast_df)

                        elif method_fc == "LSTM (Deep Learning)":
                            try:
                                import tensorflow as tf
                                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                            except ImportError:
                                st.error("TensorFlow tidak tersedia. Install dengan: pip install tensorflow")
                                st.stop()

                            data = series["y"].values.reshape(-1, 1)
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            data_scaled = scaler.fit_transform(data)

                            seq_len = 20
                            X_seq, y_seq = [], []
                            for i in range(len(data_scaled) - seq_len):
                                X_seq.append(data_scaled[i:i + seq_len])
                                y_seq.append(data_scaled[i + seq_len])
                            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

                            split = int(len(X_seq) * 0.8)
                            X_train, X_test = X_seq[:split], X_seq[split:]
                            y_train, y_test = y_seq[:split], y_seq[split:]

                            model = tf.keras.Sequential([
                                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
                                tf.keras.layers.Dropout(0.2),
                                tf.keras.layers.LSTM(32),
                                tf.keras.layers.Dense(1)
                            ])
                            model.compile(optimizer="adam", loss="mse")
                            model.fit(X_train, y_train, epochs=20, batch_size=16,
                                      validation_split=0.1, verbose=0)

                            y_pred = model.predict(X_test)
                            y_pred_rescaled = scaler.inverse_transform(y_pred)
                            y_test_rescaled = scaler.inverse_transform(y_test)

                            last_seq = data_scaled[-seq_len:]
                            preds = []
                            current_seq = last_seq.reshape(1, seq_len, 1)
                            for _ in range(horizon):
                                next_val = model.predict(current_seq, verbose=0)[0]
                                preds.append(next_val)
                                current_seq = np.append(current_seq[:, 1:, :], [[next_val]], axis=1)

                            preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
                            forecast_df = pd.DataFrame({
                                "ds": pd.date_range(start=series["ds"].iloc[-1], periods=horizon + 1, freq="30min")[1:],
                                "forecast": preds_rescaled.flatten()
                            })

                            plt.style.use("dark_background")
                            fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
                            ax.set_facecolor("black")
                            ax.plot(series["ds"], series["y"], color="orange", label="Actual")
                            ax.plot(forecast_df["ds"], forecast_df["forecast"], color="cyan", label="Forecast")
                            ax.set_title("🤖 Forecasting (LSTM)", fontsize=16, color="white")
                            ax.legend()
                            ax.tick_params(colors="white")
                            st.pyplot(fig)
                            plt.close(fig)
                            st.dataframe(forecast_df)

                            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                            rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                            r2 = r2_score(y_test_rescaled, y_pred_rescaled)
                            st.success(f"✅ Evaluasi LSTM: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")

                    except Exception as e:
                        st.error(f"❌ Kesalahan forecasting: {str(e)}")

        except Exception as e:
            st.error(f"❌ Gagal menyiapkan data: {str(e)}")
