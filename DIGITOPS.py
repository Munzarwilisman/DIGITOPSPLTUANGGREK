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
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from fpdf import FPDF
import base64
import io
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import time
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from pathlib import Path
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""


# =============================================
# KONFIGURASI AWAL
# =============================================

# Konfigurasi API KEY Anthropic
import anthropic
client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])


# =============================================
# FUNGSI UTAMA
# =============================================

def ensure_persistent_data():
    """Memastikan data tetap tersimpan meskipun aplikasi di-refresh"""
    # Cek apakah data sudah ada di session state
    if 'persistent_data' not in st.session_state:
        # Coba muat dari file jika ada
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
    """Fungsi untuk menghasilkan insight analisis dari DeepSeek AI"""
    prompt = f"""
Kamu adalah seorang ahli analisis data pembangkit listrik yang menggunakan boiler CFB dan Steam turbin dengan kapasitas 25 MW (Power Plant Performance Analyst) refrensi mu adalah buku seperti boiler operation dan Design, Standart EPRI, ASME dan standart lainnya. 
Analisis berikut berasal dari parameter operasional pada PLTU (Pembangkit Listrik Tenaga Uap) dan kamu seorang expert di peralatan condenser, steam turbin, generator, boiler, water wall, Air Preheater khusus type tubular, Fan Boiler, Motor Valve dan lain-lain di PLTU dan gunakan deep research dengan semua referensi yang kamu miliki.

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

Data visualisasi: {chart_data if chart_data else "Tidak tersedia"}
"""
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Gunakan model Claude yang sesuai
            max_tokens=2000,
            temperature=0.8,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Respons Claude ada di response.content[0].text
        ai_content = response.content[0].text

        # Format hasil menjadi lebih terstruktur
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

# Pastikan import 're' sudah ditambahkan di bagian atas file
import re  # Tambahkan ini di bagian atas file bersama import lainnya

def get_anomaly_insight(parameter, anomaly_data, method, metrics, correlation_data=None):
    """Fungsi untuk menghasilkan insight khusus anomali dari Claude AI dengan mempertimbangkan korelasi"""
    
    # Siapkan informasi korelasi untuk dimasukkan dalam prompt
    correlation_info = ""
    if correlation_data:
        correlation_info = f"\n\nINFORMASI KORELASI PARAMETER:\n"
        for i, (corr_param, corr_value, direction, strength, explanation) in enumerate(correlation_data[:3]):  # Ambil 3 korelasi terkuat
            correlation_info += f"{i+1}. {corr_param}: {corr_value:.3f} ({direction}, {strength})\n"
            correlation_info += f"   - {explanation}\n"
    
    prompt = f"""
Kamu adalah seorang ahli analisis data pembangkit listrik yang menggunakan boiler CFB dan Steam turbin dengan kapasitas 25 MW (Power Plant Performance Analyst) refrensi mu adalah buku seperti boiler operation dan Design, Standart EPRI, ASME dan standart lainnya. 
Analisis berikut berasal dari parameter operasional pada PLTU (Pembangkit Listrik Tenaga Uap) dan kamu seorang expert di peralatan condenser, steam turbin, generator, boiler, water wall, Air Preheater, Fan Boiler, Motor Valve dan lain-lain di PLTU dan gunakan deep research dengan semua referensi yang kamu miliki.
Data berikut berasal dari analisis parameter {parameter} pada PLTU menggunakan metode {method}.

Detail analisis:
- Jumlah anomali terdeteksi: {metrics['count']}
- Persentase anomali: {metrics['percent']}
- Anomali nilai tinggi: {metrics['high_count']}
- Anomali nilai rendah: {metrics['low_count']}
- Rata-rata nilai anomali: {metrics['mean']:.2f}
- Rata-rata nilai normal historis: {metrics['hist_mean']:.2f}
- Deviasi rata-rata: {metrics['deviation']:.2f}%
{correlation_info}

Tugasmu adalah:
1. Berikan penjelasan teknis yang jelas tentang apa arti dari anomali yang ditemukan pada parameter {parameter}.
2. Analisis korelasi dengan parameter lain yang terkait dan jelaskan implikasi teknisnya.
3. Identifikasi 4-6 kemungkinan penyebab anomali tersebut pada sistem PLTU secara spesifik dengan detail teknis, DENGAN MEMPERTIMBANGKAN korelasi dengan parameter lain.
4. Berikan 3-4 rekomendasi bagi operator untuk tindak lanjut yang SANGAT SPESIFIK untuk mengatasi anomali pada parameter {parameter}.
5. Jelaskan potensi dampak jika anomali dibiarkan.

Format jawaban HARUS mengikuti template berikut:
[ANALISIS] <jelasan teknis anomali>
[PENYEBAB] <penyebab teknis dalam format berikut>
1. Penyebab utama pertama
   - Sub-detail pertama
   - Sub-detail kedua
   - Sub-detail ketiga
   - Sub-detail keempat
2. Penyebab utama kedua
   - Sub-detail pertama
   - Sub-detail kedua
   - Sub-detail ketiga
   - Sub-detail keempat
3. Penyebab utama ketiga
   - Sub-detail pertama
   - Sub-detail kedua
   - Sub-detail ketiga
   - Sub-detail keempat


[REKOMENDASI] <rekomendasi tindak lanjut spesifik dalam format yang sama dengan penyebab>
[DAMPAK] <potensi dampak jika diabaikan>

PENTING:
- Jawaban HARUS sangat spesifik dan teknis untuk parameter {parameter} (JANGAN memberikan jawaban generik)
- Jelaskan dampak SPESIFIK dari parameter {parameter} pada sistem PLTU, bukan dampak umum
- Berikan rekomendasi yang HANYA relevan dengan parameter {parameter}
- Gunakan format penomoran yang konsisten: angka diikuti titik untuk poin utama (1., 2., 3.) dan dash (-) untuk sub-detail
- JANGAN gunakan simbol bullet (* atau •) - gunakan HANYA dash (-) untuk sub-detail
- Format konten dengan sangat rapi dan konsisten
- Gunakan terminologi yang akurat untuk sistem PLTU
- Analisis HARUS mempertimbangkan korelasi dengan parameter lain yang terkait dengan {parameter} 
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.8,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        ai_content = response.content[0].text

        # Definisikan emojis dan labels untuk setiap bagian
        sections = {
            "ANALISIS": "🔬 **Analisis Teknis Anomali**",
            "PENYEBAB": "⚠️ **Kemungkinan Penyebab**",
            "REKOMENDASI": "🛠️ **Rekomendasi Tindak Lanjut**",
            "DAMPAK": "💥 **Potensi Dampak**"
        }

        # Default contents sebagai fallback
        default_contents = {
            "ANALISIS": f"Anomali pada parameter {parameter} menunjukkan penyimpangan dari pola operasional normal yang memerlukan investigasi lebih lanjut.",
            "PENYEBAB": f"**1.** Kemungkinan disebabkan oleh perubahan kondisi operasional\n   - Fluktuasi beban operasi\n   - Perubahan kualitas bahan bakar\n\n**2.** Masalah pada instrumentasi pengukuran parameter {parameter}\n   - Kalibrasi sensor tidak akurat\n   - Drift pada transmitter\n\n**3.** Gangguan pada komponen yang mempengaruhi parameter ini\n   - Degradasi peralatan terkait\n   - Fouling atau plugging",
            "REKOMENDASI": f"**1.** Lakukan inspeksi fisik pada sensor {parameter}\n   - Periksa kondisi fisik sensor\n   - Pastikan tidak ada kerusakan\n\n**2.** Verifikasi kalibrasi instrumen pengukuran\n   - Gunakan alat standar kalibrasi\n   - Dokumentasikan hasil kalibrasi\n\n**3.** Analisis pola operasi sebelum terjadinya anomali\n   - Review data historis\n   - Identifikasi perubahan operasional",
            "DAMPAK": f"Jika anomali pada {parameter} tidak ditangani, dapat memengaruhi efisiensi dan keandalan pembangkit listrik secara keseluruhan."
        }

        formatted_result = ""
        for section, header in sections.items():
            section_tag = f"[{section}]"
            if section_tag in ai_content:
                start_pos = ai_content.find(section_tag) + len(section_tag)
                next_pos = len(ai_content)
                for next_tag in [f"[{s}]" for s in sections.keys()]:
                    tag_pos = ai_content.find(next_tag, start_pos)
                    if tag_pos > -1 and tag_pos < next_pos:
                        next_pos = tag_pos
                section_content = ai_content[start_pos:next_pos].strip()

                # Format khusus untuk Penyebab dan Rekomendasi
                if section in ["PENYEBAB", "REKOMENDASI"]:
                    lines = section_content.split('\n')
                    formatted_lines = []
                    current_point = None
                    for line in lines:
                        line = line.strip()
                        if not line:
                            if current_point is not None and formatted_lines and not formatted_lines[-1] == "":
                                formatted_lines.append("")
                            continue
                        number_match = re.match(r'^(\d+)\.(\s|$)', line)
                        if number_match:
                            number = number_match.group(1)
                            rest_of_line = line[len(number_match.group(0)):].strip()
                            if current_point is not None and formatted_lines and not formatted_lines[-1] == "":
                                formatted_lines.append("")
                            current_point = number
                            formatted_lines.append(f"**{number}.** {rest_of_line}")
                        elif line.startswith('*') or line.startswith('•') or line.startswith('-'):
                            sub_content = line[1:].strip()
                            formatted_lines.append(f"   - {sub_content}")
                        elif current_point is not None:
                            if formatted_lines and formatted_lines[-1].startswith(f"**{current_point}.**"):
                                last_line = formatted_lines[-1]
                                formatted_lines[-1] = f"{last_line} {line}"
                            else:
                                formatted_lines.append(f"   - {line}")
                        else:
                            current_point = "1" if not formatted_lines else str(int(current_point) + 1)
                            formatted_lines.append(f"**{current_point}.** {line}")
                    section_content = '\n'.join(formatted_lines)

                formatted_result += f"\n{header}\n{section_content}\n"
            else:
                formatted_result += f"\n{header}\n{default_contents[section]}\n"

        return formatted_result

    except Exception as e:
        # Fallback jika error
        fallback_result = f"""
🔬 **Analisis Teknis Anomali**
Anomali pada parameter {parameter} menunjukkan penyimpangan dari pola operasional normal yang memerlukan investigasi lebih lanjut.

⚠️ **Kemungkinan Penyebab**
**1.** Potensi masalah pada sensor atau instrumentasi pengukuran
   - Kalibrasi sensor memerlukan pemeriksaan
   - Kemungkinan drift pada pengukuran

**2.** Perubahan kondisi operasional yang tidak terdokumentasi
   - Perubahan pada beban atau parameter input lainnya
   - Modifikasi setting yang tidak tercatat

**3.** Degradasi komponen yang terkait dengan parameter ini
   - Umur komponen yang sudah mendekati batas operasional
   - Keausan normal pada peralatan terkait

🛠️ **Rekomendasi Tindak Lanjut**
**1.** Lakukan verifikasi sensor dan kalibrasi ulang instrumen pengukuran
   - Gunakan alat kalibrasi terstandar
   - Dokumentasikan hasil kalibrasi dengan baik

**2.** Periksa riwayat maintenance komponen terkait
   - Review jadwal pemeliharaan rutin
   - Bandingkan dengan data historis

**3.** Analisis tren parameter terkait dalam periode yang lebih panjang
   - Cari pola perubahan bertahap
   - Identifikasi korelasi dengan parameter lain

**4.** Konsultasikan dengan tim teknis untuk analisis lanjutan
   - Libatkan spesialis instrumentasi
   - Pertimbangkan pemeriksaan mendalam

💥 **Potensi Dampak**
Jika dibiarkan, anomali pada parameter ini dapat memengaruhi efisiensi operasional, meningkatkan konsumsi bahan bakar, atau menyebabkan kerusakan komponen terkait dalam jangka panjang.
"""
        return fallback_result + f"\n\n*Catatan: Terjadi error dalam analisis AI: {str(e)}*"

# === Fungsi Insight AI untuk hasil prediksi ===
def get_prediction_insight(input_param, input_value, results, correlations, method):
    """
    Generate AI insight untuk hasil prediksi
    """
    insight = f"""
# 📈 Interpretasi Hasil Prediksi

## Parameter Input
- **{input_param}** = **{input_value:.2f}**

## Metode Prediksi
{method}

## Hasil Prediksi Utama
"""

    # Tambahkan hasil prediksi terbaik (dengan R² tertinggi)
    if results:
        best_param = max(results.items(), key=lambda x: x[1]['r_squared'])
        insight += f"- **{best_param[0]}** diprediksi sebesar **{best_param[1]['prediction']:.2f}** (Akurasi: {best_param[1]['r_squared']:.3f})\n"

    # Tambahkan informasi korelasi
    if correlations:
        insight += "\n## 🔗 Hubungan Korelasi\n"
        for param, corr_value, direction, strength in correlations:
            insight += f"- **{input_param}** dan **{param}** memiliki korelasi {direction} yang {strength} ({corr_value:.3f})\n"

    # Tambahkan rekomendasi
    insight += """
## 🎯 Rekomendasi Operasional

Berdasarkan pola historis, dengan nilai input yang diberikan:
"""

    if "temperature" in input_param.lower() or "suhu" in input_param.lower():
        if input_value > np.mean([v['prediction'] for v in results.values()]):
            insight += "- ⚠️ **Nilai temperatur relatif tinggi**, perhatikan kemungkinan overheating\n"
            insight += "- ✅ Pertimbangkan untuk meningkatkan cooling capacity jika diperlukan\n"
        else:
            insight += "- ✅ **Nilai temperatur dalam rentang normal**\n"
            insight += "- ℹ️ Monitor terus untuk menjaga stabilitas operasi\n"

    elif "pressure" in input_param.lower() or "tekanan" in input_param.lower():
        insight += "- 🔧 **Parameter tekanan mempengaruhi banyak aspek operasional**\n"
        insight += "- 📊 Pastikan tekanan dalam rentang aman sesuai design specification\n"

    insight += """
## ⚠️ Batasan Prediksi

1. Prediksi berdasarkan pola historis dan mungkin tidak memperhitungkan kondisi khusus
2. Akurasi prediksi bergantung pada kualitas dan konsistensi data historis
3. Selalu konfirmasi dengan monitoring real-time sebelum mengambil keputusan operasional
"""

    return insight

def save_data(sheet_dict):
    """Menyimpan data ke file pickle dan session state"""
    try:
        # Simpan ke file
        with open("uploaded_data.pkl", "wb") as f:
            pickle.dump(sheet_dict, f)
        
        # Simpan ke session state untuk persistensi
        st.session_state.persistent_data = sheet_dict
    except Exception as e:
        st.error(f"Gagal menyimpan data: {str(e)}")

def load_data():
    """Memuat data dari session state atau file pickle"""
    # Cek dari session state dulu
    if 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        return st.session_state.persistent_data
    
    # Jika tidak ada di session state, coba muat dari file
    try:
        if os.path.exists("uploaded_data.pkl"):
            with open("uploaded_data.pkl", "rb") as f:
                data = pickle.load(f)
                # Simpan ke session state untuk penggunaan berikutnya
                st.session_state.persistent_data = data
                return data
        return None
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return None

def clear_saved_data():
    """Menghapus data yang tersimpan"""
    try:
        if os.path.exists("uploaded_data.pkl"):
            os.remove("uploaded_data.pkl")
        
        # Hapus dari session state
        if 'persistent_data' in st.session_state:
            del st.session_state.persistent_data
            
        # Hapus variable session state lainnya
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
.stApp {
    background-color: #121212;
    color: white;
}
.st-b8 {
    background-color: #1E1E1E;
}
.css-1d391kg {
    background-color: #1E1E1E;
}
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #FFA500;
}
.metric-value {
    color: #FFA500;
    font-size: 1.2em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

dark_template = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': 'white'},
        'xaxis': {
            'gridcolor': 'rgba(100,100,100,0.5)',
            'linecolor': 'rgba(255,255,255,0.5)',
            'zerolinecolor': 'rgba(100,100,100,0.5)'
        },
        'yaxis': {
            'gridcolor': 'rgba(100,100,100,0.5)',
            'linecolor': 'rgba(255,255,255,0.5)',
            'zerolinecolor': 'rgba(100,100,100,0.5)'
        }
    }
}
# Panggil fungsi untuk memastikan data persisten
ensure_persistent_data()

# =============================================
# NAVIGASI SIDEBAR
# =============================================

# Add the logo to the sidebar
with st.sidebar:
    st.image("https://gajiloker.com/wp-content/uploads/2024/02/Gaji-PT-PLN-Nusantara-Power-Services.jpg", 
             width=250)  # Adjust width as needed
    
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Home", "Data Collecting and Visualitation", "Machine Learning"],
        icons=["house", "bar-chart", "cpu", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1E1E1E"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#2E2E2E"},
            "nav-link-selected": {"background-color": "#FFA500"},
        }
    )
        
    if st.button("\U0001F5D1️ Hapus Semua Data", 
                help="Hapus semua data yang telah diunggah",
                type="primary"):
        clear_saved_data()

# =============================================
# HALAMAN HOME
# =============================================
if selected == "Home":
    # CSS for modern design
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes wave {
        0% { transform: translateY(0); }
        25% { transform: translateY(-5px); }
        50% { transform: translateY(0); }
        75% { transform: translateY(5px); }
        100% { transform: translateY(0); }
    }
    
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
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        animation: fadeIn 1.5s ease-out;
        border-left: 5px solid #4facfe;
        position: relative;
        overflow: hidden;
    }
    
    .title-container:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 16px;
        pointer-events: none;
    }
    
    .title-container:after {
        content: '';
        position: absolute;
        top: -20%;
        right: -5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(79, 172, 254, 0.15) 0%, rgba(0, 51, 102, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
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
        font-weight: 400;
        margin-top: 15px;
        text-align: center;
        max-width: 90%;
        z-index: 1;
        position: relative;
        letter-spacing: 0.5px;
        line-height: 1.4;
    }
    
    .ai-animation-container {
        margin: 30px auto;
        width: 100%;
        display: flex;
        justify-content: center;
        padding: 15px;
        position: relative;
    }
    
    .ai-brain {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        border-radius: 50%;
        position: relative;
        animation: pulse 3s infinite ease-in-out;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
    }
    
    .ai-circuit {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 120px;
        height: 120px;
        margin-left: -60px;
        margin-top: -60px;
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 50%;
        border-top-color: rgba(255, 255, 255, 0.8);
        animation: rotate 4s linear infinite;
    }
    
    .ai-circuit:before {
        content: '';
        position: absolute;
        top: 8px;
        left: 8px;
        right: 8px;
        bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        border-right-color: rgba(255, 255, 255, 0.6);
        animation: rotate 3s linear infinite reverse;
    }
    
    .ai-data {
        position: absolute;
        width: 4px;
        height: 4px;
        background-color: white;
        border-radius: 50%;
        animation: float 2s infinite ease-in-out;
        box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
    }
    
    .ai-data:nth-child(1) { top: 20%; left: 80%; animation-delay: 0.2s; }
    .ai-data:nth-child(2) { top: 80%; left: 20%; animation-delay: 0.5s; }
    .ai-data:nth-child(3) { top: 40%; left: 10%; animation-delay: 0.8s; }
    .ai-data:nth-child(4) { top: 60%; left: 90%; animation-delay: 1.1s; }
    
    .ai-wave {
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 100%;
        display: flex;
        justify-content: center;
    }
    
    .ai-wave-bar {
        width: 3px;
        height: 10px;
        margin: 0 2px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 3px;
        animation: wave 1.2s infinite ease-in-out;
    }
    
    .ai-wave-bar:nth-child(1) { animation-delay: 0.0s; height: 8px; }
    .ai-wave-bar:nth-child(2) { animation-delay: 0.1s; height: 10px; }
    .ai-wave-bar:nth-child(3) { animation-delay: 0.2s; height: 14px; }
    .ai-wave-bar:nth-child(4) { animation-delay: 0.3s; height: 18px; }
    .ai-wave-bar:nth-child(5) { animation-delay: 0.4s; height: 14px; }
    .ai-wave-bar:nth-child(6) { animation-delay: 0.5s; height: 10px; }
    .ai-wave-bar:nth-child(7) { animation-delay: 0.6s; height: 8px; }
    
    .stButton button {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    
    .info-container {
        background: rgba(30, 30, 30, 0.9);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 12px;
        border-left: 4px solid #4facfe;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        margin-top: 20px;
    }
    </style>
    
    <div class="title-container">
        <h1 class="main-title">📈 AI - PLTU ANGGREK</h1>
        <p class="subtitle">DIGIT-OPS (Digital Intelligent Operation System): Deteksi Anomali & Optimasi Kinerja PLTU Anggrek</p>
    </div>
    
    <div class="ai-animation-container">
        <div class="ai-brain">
            <div class="ai-circuit"></div>
            <div class="ai-data"></div>
            <div class="ai-data"></div>
            <div class="ai-data"></div>
            <div class="ai-data"></div>
            <div class="ai-wave">
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
                <div class="ai-wave-bar"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Animasi teks menggunakan st.empty() dan loop
    placeholder = st.empty()
    
    full_text = "Selamat Datang Di Artificial Intelligence PLTU Anggrek "
    
    # Efek typing animation
    for i in range(len(full_text) + 1):
        placeholder.markdown(f"### {full_text[:i]}_")
        time.sleep(0.03)  # Kecepatan ketikan
    
    # Menampilkan teks final tanpa underscore
    placeholder.markdown(f"### {full_text}")
    
    # Tampilkan konten utama dengan sedikit delay untuk efek animasi berurutan
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
# HALAMAN PERFORMANCE INDIKATOR
# =============================================

elif selected == "Data Collecting and Visualitation":
    st.title("\U0001F4CA Data Collecting and Visualitation")
    
    # Cek apakah data sudah tersedia dari session state
    if 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        st.session_state.sheet_dict = st.session_state.persistent_data
        st.session_state.sheet_names = list(st.session_state.sheet_dict.keys())
        st.session_state.df = list(st.session_state.sheet_dict.values())[0]
        st.success("✅ Data berhasil dimuat dari sesi sebelumnya!")
    
    # File uploader tetap ada untuk mengunggah data baru
    uploaded_file = st.file_uploader("\U0001F4C4 Upload file data (CSV atau Excel)", type=["csv", "xlsx"])

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

    selected_sheet = st.radio("\U0001F4C4 Pilih Sheet:", st.session_state.sheet_names, horizontal=True)
    df = st.session_state.sheet_dict[selected_sheet]
    st.session_state.df = df

    # Deteksi kolom tanggal
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

    # Statistik Deskriptif
    st.markdown("### \U0001F4CA Statistik Deskriptif")
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

    # Grafik Tren
    st.markdown("### \U0001F4C9 Grafik Tren (Resolusi Penuh)")
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
        df_sorted, 
        x=x_axis,
        y=selected_param,
        title=f"<b>Tren {selected_param}</b>",
        labels={'value': selected_param},
        template='plotly_dark',
        color_discrete_sequence=[line_color]
    )

    if show_ma:
        df_sorted['MA_7'] = df_sorted[selected_param].rolling(window=7).mean()
        fig_line.add_scatter(
            x=df_sorted[x_axis],
            y=df_sorted['MA_7'],
            name='Moving Avg (7)',
            line=dict(color='#00FFFF', width=2, dash='dot')
        )

    fig_line.update_traces(
        line=dict(width=3),
        marker=dict(size=8, opacity=0.9, line=dict(width=1, color='white'))
    )

    if date_column:
        fig_line.update_xaxes(
            tickformat='%Y-%m-%d %H:%M',
            tickangle=45,
            nticks=10,
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='rgba(150,150,150,0.2)'
            )
        )
        fig_line.update_traces(
            hovertemplate="<b>Tanggal:</b> %{x|%Y-%m-%d %H:%M}<br><b>Nilai:</b> %{y:.2f}<extra></extra>"
        )

    fig_line.update_layout(
        **dark_template['layout'],
        margin=dict(l=40, r=40, t=80, b=80),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='white')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='white')
        )
    )

    mean_value = df_sorted[selected_param].mean()
    last_value = df_sorted[selected_param].iloc[-1] if len(df_sorted) > 0 else 0
    
    fig_line.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        text=f"<b>Nilai Terakhir:</b> <span style='color:#FFA500'>{last_value:.2f}</span>",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(30,30,30,0.7)",
        bordercolor="#FFA500",
        borderwidth=1
    )
    
    fig_line.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.88,
        text=f"<b>Rata-rata:</b> <span style='color:#00FFFF'>{mean_value:.2f}</span>",
        showarrow=False,
        font=dict(size=14),
        bgcolor="rgba(30,30,30,0.7)",
        bordercolor="#00FFFF",
        borderwidth=1
    )

    st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")
    
    # Histogram
    st.markdown("### \U0001F4CA Histogram")
    fig_hist = px.histogram(
        df, 
        x=selected_param, 
        template='plotly_dark',
        color_discrete_sequence=['#FFA500']
    )
    fig_hist.update_layout(
        bargap=0.1,
        xaxis_title=selected_param,
        yaxis_title='Frekuensi'
    )
    st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

    # Box Plot
    st.markdown("### \U0001F4E6 Box Plot")
    fig_box = px.box(
        df, 
        y=selected_param, 
        template='plotly_dark',
        color_discrete_sequence=['#FFA500']
    )
    fig_box.update_layout(
        yaxis_title=selected_param,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")

    # Tombol Analisa AI
    if st.button("🤖 Analisa AI PLTU ANGGREK", key="ai_analysis_button"):
        with st.spinner("⏳ Menganalisis data dengan AI..."):
            st.markdown("#### 🤖 Insight PLTU Anggrek")
            
            with st.container():
                st.markdown("##### 📈 Analisis Grafik Tren")
                ai_result = get_ai_insight(selected_param, df[selected_param].dropna(), "Grafik Tren", chart_data=fig_line.to_dict())
                st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>{ai_result}</div>", unsafe_allow_html=True)
                st.markdown("---")
                
                st.markdown("##### 📊 Analisis Histogram")
                ai_result_hist = get_ai_insight(selected_param, df[selected_param].dropna(), "Histogram", chart_data=fig_hist.to_dict())
                st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #00FFFF;'>{ai_result_hist}</div>", unsafe_allow_html=True)
                st.markdown("---")
                
                st.markdown("##### 📦 Analisis Box Plot")
                ai_result_box = get_ai_insight(selected_param, df[selected_param].dropna(), "Box Plot", chart_data=fig_box.to_dict())
                st.markdown(f"<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FF69B4;'>{ai_result_box}</div>", unsafe_allow_html=True)

# ====# =============================================
# HALAMAN MACHINE LEARNING (MODIFIED)
# =============================================

elif selected == "Machine Learning":
    st.title("🤖 Machine Learning PLTU Anggrek")
    st.markdown("### Fitur Analisis Lanjutan dengan Machine Learning")
    
    # INISIALISASI STATE UNTUK ANALISIS AI - TEMPATKAN DI SINI
    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = {}
    
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    
    # Cek apakah data tersedia di session state
    if "df" not in st.session_state and 'persistent_data' in st.session_state and st.session_state.persistent_data is not None:
        # Muat data dari persistent_data
        sheet_dict = st.session_state.persistent_data
        st.session_state.df = list(sheet_dict.values())[0]
    
    if "df" not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu Performance Indikator.")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # Deteksi kolom tanggal
    date_column = None
    for col in df.columns:
        if pd.to_datetime(df[col], errors='coerce').notna().all():
            date_column = col
            df[date_column] = pd.to_datetime(df[date_column])
            break
    
    if not date_column:
        st.error("Data tidak memiliki kolom tanggal yang valid. Tidak dapat melakukan analisis time series.")
        st.stop()
    
    # Sort by date
    df = df.sort_values(date_column)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 Data Preparation", "📊 Exploration Data Analysis", "🔍 Anomaly Detection And AI Evaluation", "🎯Prediction Analysis", "📈 Forecasting"])
    
    with tab1:
        st.header("Persiapan Data (Seluruh Periode)")
        
        # Step 1: Pilih parameter target untuk pemfilteran
        st.subheader("1. Pilih Parameter Target untuk Pemfilteran Data")
        filter_param = st.selectbox(
            "Pilih Parameter untuk Pemfilteran Data:",
            options=df.select_dtypes(include=['number']).columns.tolist(),
            key="filter_param_select"
        )
        
        # Tetapkan threshold untuk filter data
        filter_threshold = st.number_input(
            "Nilai Minimum Data yang Valid:",
            value=5.0,
            help="Data dengan nilai di bawah threshold ini akan ditandai untuk preprocessing"
        )
        
        # Tampilkan data yang di bawah threshold
        filtered_rows = df[df[filter_param] < filter_threshold].shape[0]
        missing_rows = df[filter_param].isna().sum()
        total_rows_to_process = filtered_rows + missing_rows
        
        st.info(f"Terdapat {filtered_rows} baris dengan nilai {filter_param} < {filter_threshold} dan {missing_rows} baris dengan nilai kosong. Total {total_rows_to_process} baris yang perlu diproses.")
        
        # Tunjukkan distribusi data pada parameter filter
        st.subheader("Distribusi Nilai Parameter Filter")
        fig_filter = px.histogram(
            df,
            x=filter_param,
            template='plotly_dark',
            color_discrete_sequence=['#FFA500'],
            title=f"Distribusi Nilai {filter_param}"
        )
        fig_filter.add_vline(x=filter_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_filter, use_container_width=True)
        
        # Step 2: Pilih parameter yang akan diproses
        st.subheader("2. Pilih Parameter yang Akan Diproses")
        
        # Metode pemilihan parameter yang akan diproses
        process_method = st.radio(
            "Metode Pemilihan Parameter:",
            options=[
                "Proses Semua Parameter Numerik",
                "Pilih Parameter Spesifik"
            ]
        )
        
        num_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if process_method == "Pilih Parameter Spesifik":
            selected_params = st.multiselect(
                "Pilih Parameter yang Akan Diproses:",
                options=num_columns,
                default=[filter_param]
            )
        else:
            selected_params = num_columns
            st.info(f"Semua parameter numerik ({len(num_columns)} parameter) akan diproses.")
        
        # Tampilkan grafik trend parameter utama yang akan dianalisis
        st.subheader("3. Tren Data Asli Parameter Utama")
        target_param = st.selectbox(
            "Pilih Parameter Utama untuk Analisis:",
            options=selected_params,
            index=0,
            key="target_param_select"
        )
        
        fig_original = px.line(
            df,
            x=date_column,
            y=target_param,
            title=f"Tren Original {target_param}",
            template='plotly_dark',
            color_discrete_sequence=['#FFA500']
        )
        
        # Tandai data yang akan diproses pada grafik
        if filter_param == target_param:
            mask_below_threshold = df[filter_param] < filter_threshold
            mask_na = df[filter_param].isna()
            
            # Data di bawah threshold
            fig_original.add_scatter(
                x=df[mask_below_threshold][date_column],
                y=df[mask_below_threshold][target_param],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name=f'Nilai < {filter_threshold}'
            )
            
            # Data yang hilang (jika ada dan jika parameter sama)
            if mask_na.any():
                # Untuk data NA, kita perlu membuat titik dengan nilai y=0 atau nilai lain yang jelas
                fig_original.add_scatter(
                    x=df[mask_na][date_column],
                    y=[0] * mask_na.sum(),  # Gunakan nilai 0 untuk visualisasi
                    mode='markers',
                    marker=dict(color='purple', size=10, symbol='circle-open'),
                    name='Missing Values'
                )
        
        fig_original.update_layout(height=400)
        st.plotly_chart(fig_original, use_container_width=True)
        
        # Step 3: Pilihan teknik preprocessing
        st.subheader("4. Pilih Metode Preprocessing")
        preprocessing_options = st.multiselect(
            "Teknik yang akan diterapkan:",
            options=[
                "Imputasi Missing Values",
                "Smoothing (Moving Average)",
                "Normalisasi (Min-Max)",
                "Standarisasi (Z-Score)", 
                "Detrending"
            ],
            default=["Imputasi Missing Values"]
        )
        
        # Parameter preprocessing
        processing_params = {}
        
        # Opsi imputasi
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
        
        # Proses data
        if st.button("🛠️ Proses Data", key="process_button"):
            with st.spinner("Memproses data..."):
                # Salin data asli
                processed_data = df.copy()
                
                # Identifikasi baris yang perlu diproses
                mask_to_process = (processed_data[filter_param] < filter_threshold) | (processed_data[filter_param].isna())
                st.info(f"Memproses {mask_to_process.sum()} dari {len(processed_data)} baris data berdasarkan filter.")
                
                # Pastikan semua parameter di baris yang sama dengan beban di bawah threshold juga diproses
                st.subheader("Pemrosesan Data Otomatis")
                
                # Hitung jumlah baris dan parameter yang akan diproses
                total_rows_processed = mask_to_process.sum()
                total_params_processed = len(selected_params)
                total_cells_processed = total_rows_processed * total_params_processed
                
                st.info(f"Memproses otomatis {total_rows_processed} baris data × {total_params_processed} parameter = {total_cells_processed} nilai data")
                
                progress_bar = st.progress(0)
                processed_params_summary = []
                
                # Proses semua parameter yang dipilih
                for i, param in enumerate(selected_params):
                    # Update progress bar
                    progress = int((i+1) / len(selected_params) * 100)
                    progress_bar.progress(progress)
                    
                    # 1. Imputasi Missing Values untuk semua data di baris yang sama
                    if "Imputasi Missing Values" in preprocessing_options:
                        method = processing_params.get('imputation_method', 'Mean')
                        
                        # Selalu proses seluruh baris yang terfilter (termasuk nilai 0 dan di bawah threshold)
                        if method == "Mean":
                            fill_value = processed_data[param].mean()
                            # Untuk nilai di bawah threshold pada parameter filter, ganti dengan nilai rata-rata
                            if param == filter_param:
                                # Ganti semua nilai di bawah threshold dengan rata-rata
                                mask_below = (processed_data[param] < filter_threshold) & ~processed_data[param].isna()
                                processed_data.loc[mask_below, param] = fill_value
                            
                            # Proses data yang ada di bawah threshold dan juga yang kosong di parameter lain
                            processed_data.loc[mask_to_process, param] = processed_data.loc[mask_to_process, param].fillna(fill_value)
                            
                            # Jika parameter bukan parameter filter, tetap lakukan imputasi untuk nilai yang rendah
                            if param != filter_param:
                                # Cari nilai yang sangat rendah atau 0 yang mungkin merupakan error
                                suspicious_values_mask = (processed_data.loc[mask_to_process, param] < filter_threshold/10) | (processed_data.loc[mask_to_process, param] == 0)
                                if suspicious_values_mask.any():
                                    processed_data.loc[mask_to_process & suspicious_values_mask, param] = fill_value
                                    
                        elif method == "Median":
                            fill_value = processed_data[param].median()
                            # Untuk nilai di bawah threshold pada parameter filter, ganti dengan nilai median
                            if param == filter_param:
                                # Ganti semua nilai di bawah threshold dengan median
                                mask_below = (processed_data[param] < filter_threshold) & ~processed_data[param].isna()
                                processed_data.loc[mask_below, param] = fill_value
                            
                            # Proses data yang kosong
                            processed_data.loc[mask_to_process, param] = processed_data.loc[mask_to_process, param].fillna(fill_value)
                            
                            # Jika parameter bukan parameter filter, tetap lakukan imputasi untuk nilai yang rendah
                            if param != filter_param:
                                # Cari nilai yang sangat rendah atau 0 yang mungkin merupakan error
                                suspicious_values_mask = (processed_data.loc[mask_to_process, param] < filter_threshold/10) | (processed_data.loc[mask_to_process, param] == 0)
                                if suspicious_values_mask.any():
                                    processed_data.loc[mask_to_process & suspicious_values_mask, param] = fill_value
                                    
                        elif method == "Forward Fill":
                            # Lakukan ffill pada baris yang terfilter
                            temp_series = processed_data[param].copy()
                            processed_data.loc[mask_to_process, param] = temp_series.fillna(method='ffill').loc[mask_to_process]
                            
                            # Untuk nilai di bawah threshold, ganti dengan nilai sebelumnya (forward fill)
                            if param == filter_param:
                                mask_below = (processed_data[param] < filter_threshold) & ~processed_data[param].isna()
                                # Jika ada nilai di bawah threshold, coba gunakan nilai sebelumnya
                                for idx in processed_data[mask_below].index:
                                    if idx > 0 and idx-1 in processed_data.index:
                                        processed_data.loc[idx, param] = processed_data.loc[idx-1, param]
                            
                            # Sama untuk parameter lain
                            if param != filter_param:
                                suspicious_values_mask = (processed_data.loc[mask_to_process, param] < filter_threshold/10) | (processed_data.loc[mask_to_process, param] == 0)
                                # Gunakan forward fill untuk nilai yang mencurigakan
                                for idx in processed_data[mask_to_process & suspicious_values_mask].index:
                                    if idx > 0 and idx-1 in processed_data.index:
                                        processed_data.loc[idx, param] = processed_data.loc[idx-1, param]
                            
                        elif method == "Backward Fill":
                            # Lakukan bfill pada baris yang terfilter
                            temp_series = processed_data[param].copy()
                            processed_data.loc[mask_to_process, param] = temp_series.fillna(method='bfill').loc[mask_to_process]
                            
                            # Untuk nilai di bawah threshold, ganti dengan nilai sesudahnya (backward fill)
                            if param == filter_param:
                                mask_below = (processed_data[param] < filter_threshold) & ~processed_data[param].isna()
                                # Jika ada nilai di bawah threshold, coba gunakan nilai sesudahnya
                                for idx in processed_data[mask_below].index:
                                    if idx < len(processed_data)-1 and idx+1 in processed_data.index:
                                        processed_data.loc[idx, param] = processed_data.loc[idx+1, param]
                            
                            # Sama untuk parameter lain
                            if param != filter_param:
                                suspicious_values_mask = (processed_data.loc[mask_to_process, param] < filter_threshold/10) | (processed_data.loc[mask_to_process, param] == 0)
                                # Gunakan backward fill untuk nilai yang mencurigakan
                                for idx in processed_data[mask_to_process & suspicious_values_mask].index:
                                    if idx < len(processed_data)-1 and idx+1 in processed_data.index:
                                        processed_data.loc[idx, param] = processed_data.loc[idx+1, param]
                            
                        elif method == "Linear Interpolation":
                            # Siapkan data untuk interpolasi
                            temp_df = processed_data[[date_column, param]].copy()
                            
                            # Tandai nilai di bawah threshold dan nilai 0 sebagai NaN untuk diinterpolasi
                            if param == filter_param:
                                mask_below = (temp_df[param] < filter_threshold) & ~temp_df[param].isna()
                                temp_df.loc[mask_below, param] = np.nan
                            
                            # Untuk parameter lain, tandai nilai yang mencurigakan
                            if param != filter_param:
                                suspicious_values_mask = (temp_df.loc[mask_to_process, param] < filter_threshold/10) | (temp_df.loc[mask_to_process, param] == 0)
                                if suspicious_values_mask.any():
                                    temp_df.loc[mask_to_process & suspicious_values_mask, param] = np.nan
                            
                            # Lakukan interpolasi pada data
                            temp_df[param] = temp_df[param].interpolate(method='linear')
                            
                            # Terapkan hasil interpolasi kembali ke data yang diproses
                            for idx in processed_data[mask_to_process].index:
                                if idx in temp_df.index:
                                    processed_data.loc[idx, param] = temp_df.loc[idx, param]
                        
                        # Tambahkan informasi ke ringkasan daripada menampilkan banyak notifikasi
                        processed_params_summary.append(f"'{param}'")
                    
                    # Buat kolom baru untuk hasil preprocessing (hanya untuk parameter target)
                    if param == target_param:
                        # 2. Smoothing
                        if "Smoothing (Moving Average)" in preprocessing_options:
                            window = processing_params.get('window_size', 7)
                            # Simpan hasil smoothing untuk semua data
                            processed_data[f'{param}_smooth'] = processed_data[param].rolling(window=window, min_periods=1).mean()
                            st.success(f"Data {param} di-smoothing dengan moving average (window={window})")
                        
                        # 3. Normalisasi
                        if "Normalisasi (Min-Max)" in preprocessing_options:
                            scaler = MinMaxScaler()
                            # Reshape diperlukan untuk scaler
                            values = processed_data[param].values.reshape(-1, 1)
                            processed_data[f'{param}_norm'] = scaler.fit_transform(values)
                            st.success(f"Data {param} dinormalisasi (Min-Max scaling)")
                        
                        # 4. Standarisasi
                        if "Standarisasi (Z-Score)" in preprocessing_options:
                            scaler = StandardScaler()
                            # Reshape diperlukan untuk scaler
                            values = processed_data[param].values.reshape(-1, 1)
                            processed_data[f'{param}_std'] = scaler.fit_transform(values)
                            st.success(f"Data {param} distandardisasi (Z-Score)")
                        
                        # 5. Detrending
                        if "Detrending" in preprocessing_options:
                            window = processing_params.get('trend_window', 14)
                            trend = processed_data[param].rolling(window=window, min_periods=1).mean()
                            processed_data[f'{param}_detrend'] = processed_data[param] - trend
                            st.success(f"Data {param} di-detrend (window={window})")
                
                # Tampilkan ringkasan pemrosesan satu kali saja, bukan untuk setiap parameter
                if len(processed_params_summary) > 0:
                    # Format ringkasan yang lebih rapi
                    if len(processed_params_summary) <= 3:
                        param_list = ", ".join(processed_params_summary)
                    else:
                        param_list = ", ".join(processed_params_summary[:3]) + f", dan {len(processed_params_summary)-3} parameter lainnya"
                    
                    st.success(f"Berhasil memproses parameter {param_list} pada {total_rows_processed} baris dengan metode {method}")
                
                # Simpan ke session state
                st.session_state.processed_data = processed_data
                st.session_state.target_param = target_param
                st.session_state.date_column = date_column
                st.session_state.full_data = processed_data.copy()  # Simpan semua data
                
                # Tampilkan hasil untuk parameter target
                st.subheader("5. Hasil Preprocessing")
                
                # Buat daftar kolom yang akan divisualisasikan
                viz_columns = [target_param] + [col for col in processed_data.columns if col.startswith(f"{target_param}_")]
                
                # Visualisasi perbandingan
                fig_processed = px.line(
                    processed_data,
                    x=date_column,
                    y=viz_columns,
                    title=f"Perbandingan Hasil Preprocessing {target_param}",
                    template='plotly_dark',
                    labels={'value': 'Nilai', 'variable': 'Metode'}
                )
                
                # Tandai data yang telah diproses
                for col in viz_columns:
                    if col == target_param:
                        # Tandai data yang diproses pada grafik utama
                        fig_processed.add_scatter(
                            x=processed_data[mask_to_process][date_column],
                            y=processed_data[mask_to_process][col],
                            mode='markers',
                            marker=dict(color='lime', size=8, symbol='circle'),
                            name='Data yang Diproses',
                            showlegend=True
                        )
                        break
                
                fig_processed.update_layout(height=500)
                st.plotly_chart(fig_processed, use_container_width=True)
                
                # Tambahkan statistik perbandingan
                st.subheader("6. Statistik Perbandingan")
                
                comparison_cols = st.columns(len(viz_columns))
                
                for i, col in enumerate(viz_columns):
                    with comparison_cols[i]:
                        col_mean = processed_data[col].mean()
                        col_std = processed_data[col].std()
                        col_min = processed_data[col].min()
                        col_max = processed_data[col].max()
                        
                        st.markdown(f"""
                        <div style='background-color:#1E1E1E; padding:10px; border-radius:10px; border-left:4px solid #FFA500; text-align:center;'>
                            <div style='font-weight:bold;'>{col}</div>
                            <div style='color:#00FFFF;'>Mean: {col_mean:.2f}</div>
                            <div style='color:#00FFFF;'>Std: {col_std:.2f}</div>
                            <div style='color:#00FFFF;'>Min: {col_min:.2f}</div>
                            <div style='color:#00FFFF;'>Max: {col_max:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tampilkan dataframe
                with st.expander("Lihat Data Hasil Preprocessing"):
                    st.dataframe(processed_data[viz_columns + [date_column]])
                
                # Simpan informasi filter untuk digunakan di tab lain
                st.session_state.filter_param = filter_param
                st.session_state.filter_threshold = filter_threshold
                st.session_state.mask_to_process = mask_to_process
    
        with tab2:
            st.header("📊 Exploration Data Analysis")
    
    # Import required libraries dalam scope tab
    try:
        import plotly.graph_objects as go
        import numpy as np
        from scipy import stats
        from statsmodels.graphics.tsaplots import plot_acf
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')  # Set dark theme for matplotlib
    except ImportError as e:
        st.warning(f"Beberapa library tidak tersedia: {str(e)}")
        st.warning("Install dengan: pip install scipy statsmodels matplotlib")
        # Set default values untuk library yang tidak ada
        go = None
        stats = None
        plot_acf = None
    
    if "processed_data" not in st.session_state:
        st.warning("Lakukan data preparation terlebih dahulu di tab Data Preparation")
        st.stop()
    
    # ⛔️ jangan ubah langsung session_state → bikin copy lokal
    try:
        processed_data = st.session_state.processed_data.copy()
        target_param = st.session_state.target_param
        date_column = st.session_state.date_column
    except Exception as e:
        st.error(f"Error mengakses data: {str(e)}")
        st.stop()
    
    # Pilih kolom yang akan dianalisis
    cols_with_processed = [c for c in processed_data.columns if c != date_column]
    default_col = target_param
    
    processed_cols = [c for c in cols_with_processed if c.startswith(f"{target_param}_")]
    if processed_cols:
        default_col = processed_cols[0]
    
    analysis_col = st.selectbox(
        "Pilih Kolom Utama untuk Analisis:",
        options=cols_with_processed,
        index=cols_with_processed.index(default_col) if default_col in cols_with_processed else 0,
        key="eda_col_select"
    )

    # Pilih parameter kedua (opsional)
    st.markdown("### 🔀 Pilih Parameter Kedua (Opsional)")
    analysis_col2 = st.selectbox(
        "Pilih Parameter Kedua:",
        options=["None"] + cols_with_processed,
        index=0,
        key="eda_col_select2"
    )
    
    # Statistik Deskriptif
    st.markdown("### 📊 Statistik Deskriptif")
    try:
        stats_desc = processed_data[analysis_col].describe()
        
        cols = st.columns(6)
        metrics = [
            ('Minimum', stats_desc['min']),
            ('Q1', stats_desc['25%']),
            ('Median', stats_desc['50%']),
            ('Mean', stats_desc['mean']),
            ('Q3', stats_desc['75%']),
            ('Maximum', stats_desc['max'])
        ]
        
        for i, (label, value) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div>{label}</div>
                    <div class="metric-value">{value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error menampilkan statistik: {str(e)}")
    
    # Grafik Tren
    st.markdown("### 📈 Grafik Tren")
    try:
        col1, col2 = st.columns([1, 1])
        with col1:
            line_color = st.color_picker("Pilih Warna Garis:", value='#FFA500', key="eda_line_color")
        with col2:
            show_ma = st.checkbox("Tampilkan Moving Average", value=True, key="eda_show_ma")
        
        y_cols = [analysis_col] if analysis_col2 == "None" else [analysis_col, analysis_col2]
        
        fig_line = px.line(
            processed_data,
            x=date_column,
            y=y_cols,
            title=f"<b>Tren {' & '.join(y_cols)}</b>",
            template='plotly_dark',
            color_discrete_sequence=['#FFA500', '#00FF7F']
        )

        if show_ma and analysis_col2 == "None":
            # Buat copy local untuk moving average agar tidak mengubah session_state
            temp_data = processed_data.copy()
            temp_data['MA_7'] = temp_data[analysis_col].rolling(window=7).mean()
            fig_line.add_scatter(
                x=temp_data[date_column],
                y=temp_data['MA_7'],
                name='Moving Avg (7)',
                line=dict(color='#00FFFF', width=2, dash='dot')
            )

        st.plotly_chart(fig_line, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat grafik tren: {str(e)}")
    
    # Histogram
    st.markdown("### 📊 Histogram")
    try:
        fig_hist = px.histogram(
            processed_data, 
            x=analysis_col, 
            template='plotly_dark',
            color_discrete_sequence=['#FFA500']
        )
        fig_hist.update_layout(
            bargap=0.1,
            xaxis_title=analysis_col,
            yaxis_title='Frekuensi'
        )
        st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat histogram: {str(e)}")

    # Box Plot
    st.markdown("### 📦 Box Plot")
    try:
        fig_box = px.box(
            processed_data, 
            y=analysis_col, 
            template='plotly_dark',
            color_discrete_sequence=['#FFA500']
        )
        fig_box.update_layout(
            yaxis_title=analysis_col,
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat box plot: {str(e)}")

    # Correlation Matrix
    st.markdown("### 🔗 Correlation Matrix")
    try:
        numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
        if date_column in numeric_cols:
            numeric_cols.remove(date_column)
            
        if len(numeric_cols) > 1:
            corr_matrix = processed_data[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                template='plotly_dark',
                title='Korelasi Antar Parameter'
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Tidak cukup parameter numerik untuk menampilkan matriks korelasi")
    except Exception as e:
        st.error(f"Error membuat correlation matrix: {str(e)}")
    
    # Density Plot
    st.markdown("### 🌊 Density Plot (KDE)")
    try:
        fig_kde = px.histogram(
            processed_data,
            x=analysis_col,
            nbins=40,
            histnorm='density',
            marginal="box",
            template='plotly_dark',
            color_discrete_sequence=['#FF4500']
        )
        st.plotly_chart(fig_kde, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat density plot: {str(e)}")

    # Violin Plot
    st.markdown("### 🎻 Violin Plot")
    try:
        fig_violin = px.violin(
            processed_data,
            y=analysis_col,
            box=True,
            points="all",
            template="plotly_dark",
            color_discrete_sequence=['#32CD32']
        )
        st.plotly_chart(fig_violin, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat violin plot: {str(e)}")

    # Lag Plot
    st.markdown("### 🔁 Lag Plot")
    try:
        if len(processed_data) > 1:
            fig_lag = px.scatter(
                x=processed_data[analysis_col][:-1].values,
                y=processed_data[analysis_col][1:].values,
                labels={'x': f'{analysis_col} (t)', 'y': f'{analysis_col} (t+1)'},
                template="plotly_dark",
                title=f"Lag Plot - {analysis_col}"
            )
            fig_lag.update_traces(marker=dict(size=6, color="#FFD700", opacity=0.8))
            st.plotly_chart(fig_lag, use_container_width=True, theme="streamlit")
        else:
            st.warning("Data tidak cukup untuk lag plot")
    except Exception as e:
        st.error(f"Error membuat lag plot: {str(e)}")

    # Autocorrelation (ACF)
    st.markdown("### 📉 Autocorrelation (ACF)")
    try:
        if plot_acf is not None:
            # Buat figure matplotlib dengan dark theme
            fig_acf, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
            ax.set_facecolor('#0E1117')
            
            # Clean data untuk ACF
            clean_data = processed_data[analysis_col].dropna()
            if len(clean_data) > 30:
                plot_acf(clean_data, ax=ax, lags=min(30, len(clean_data)//4), color='#FFA500')
                ax.set_title(f'Autocorrelation Function - {analysis_col}', color='white')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_acf)
                plt.close(fig_acf)  # Clean up matplotlib figure
            else:
                st.warning("Data tidak cukup untuk analisis autocorrelation (minimal 30 data points)")
        else:
            st.warning("Statsmodels tidak tersedia untuk ACF plot")
    except Exception as e:
        st.error(f"Error membuat ACF plot: {str(e)}")

    # Rolling Statistics
    st.markdown("### 📊 Rolling Statistics (Mean & Std)")
    try:
        temp_df = processed_data.copy()  # Local copy
        temp_df['Rolling_Mean'] = temp_df[analysis_col].rolling(window=7).mean()
        temp_df['Rolling_Std'] = temp_df[analysis_col].rolling(window=7).std()
        
        fig_roll = px.line(
            temp_df,
            x=date_column,
            y=[analysis_col, 'Rolling_Mean', 'Rolling_Std'],
            template="plotly_dark",
            title=f"Rolling Statistics - {analysis_col}"
        )
        fig_roll.update_layout(
            yaxis_title='Value',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roll, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.error(f"Error membuat rolling statistics: {str(e)}")

    # Scatter Matrix (Pair Plot)
    if analysis_col2 != "None":
        st.markdown("### 🔎 Scatter Matrix (Pair Plot)")
        try:
            scatter_cols = [analysis_col, analysis_col2]
            fig_scatter_matrix = px.scatter_matrix(
                processed_data[scatter_cols],
                dimensions=scatter_cols,
                template="plotly_dark",
                color_discrete_sequence=['#FF7F50'],
                title=f"Scatter Matrix: {analysis_col} vs {analysis_col2}"
            )
            fig_scatter_matrix.update_layout(height=600)
            st.plotly_chart(fig_scatter_matrix, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Error membuat scatter matrix: {str(e)}")

        # Scatter Plot Antar Parameter
        st.markdown("### 🔎 Scatter Plot Antar 2 Parameter")
        try:
            fig_scatter = px.scatter(
                processed_data,
                x=analysis_col,
                y=analysis_col2,
                template="plotly_dark",
                color_discrete_sequence=['#FF69B4'],
                title=f"Correlation: {analysis_col} vs {analysis_col2}"
            )
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
            
            # Add correlation coefficient
            corr_coef = processed_data[[analysis_col, analysis_col2]].corr().iloc[0, 1]
            fig_scatter.add_annotation(
                text=f"Correlation: {corr_coef:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color="white"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="#FF69B4",
                borderwidth=1
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit")
        except Exception as e:
            st.error(f"Error membuat scatter plot: {str(e)}")
    
    # Tombol Analisa AI - di akhir tab
    st.markdown("---")
    if st.button("🤖 Analisa AI PLTU ANGGREK", key="eda_ai_analysis"):
        try:
            with st.spinner("⏳ Menganalisis data dengan AI..."):
                st.markdown("#### 🤖 Insight PLTU Anggrek")
                
                # Pastikan fungsi get_ai_insight ada
                if 'get_ai_insight' in globals():
                    clean_data = processed_data[analysis_col].dropna()
                    ai_result = get_ai_insight(analysis_col, clean_data, "EDA Analysis")
                    st.markdown(f"""
                    <div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>
                        {ai_result}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Fitur AI analysis belum tersedia. Pastikan fungsi get_ai_insight sudah didefinisikan.")
        except Exception as e:
            st.error(f"Error dalam AI analysis: {str(e)}")

# ⚠️ IMPORTANT: Semua kode di atas berada dalam scope 'with tab2:'
# Tidak ada kode yang keluar dari indentasi tab
    
    with tab3:
        st.header("Deteksi Anomali (30 Hari Terakhir vs Pola Historis)")
        
        if "processed_data" not in st.session_state:
            st.warning("Lakukan data preparation terlebih dahulu di tab sebelumnya")
            st.stop()
            
        full_data = st.session_state.full_data
        target_param = st.session_state.target_param
        date_column = st.session_state.date_column
        
        # Ambil 30 hari terakhir
        last_date = full_data[date_column].max()
        start_date = last_date - pd.Timedelta(days=30)
        df_last30 = full_data[full_data[date_column] >= start_date].copy()
        
        # Tunjukkan info tentang data yang diproses pada tab ini
        if "filter_param" in st.session_state and "filter_threshold" in st.session_state:
            filter_param = st.session_state.filter_param
            filter_threshold = st.session_state.filter_threshold
            
            # Hitung berapa banyak data yang diproses dalam 30 hari terakhir
            if "mask_to_process" in st.session_state:
                mask_30day = st.session_state.mask_to_process & (full_data[date_column] >= start_date)
                processed_count_30day = mask_30day.sum()
                total_count_30day = df_last30.shape[0]
                
                # Tampilkan informasi
                st.info(f"Dari {total_count_30day} data dalam 30 hari terakhir, {processed_count_30day} baris telah diproses karena {filter_param} < {filter_threshold} atau nilainya kosong.")
        
        st.markdown(f"### Analisis 30 Hari Terakhir ({start_date.strftime('%Y-%m-%d')} hingga {last_date.strftime('%Y-%m-%d')})")
        
        # Pilih kolom yang akan dianalisis
        cols_with_processed = [c for c in full_data.columns if c != date_column]
        default_col = target_param
        
        # Pilih kolom yang sudah diproses sebagai default jika ada
        processed_cols = [c for c in cols_with_processed if c.startswith(f"{target_param}_")]
        if processed_cols:
            default_col = processed_cols[0]
        
        analysis_col = st.selectbox(
            "Pilih Kolom untuk Analisis Anomali:",
            options=cols_with_processed,
            index=cols_with_processed.index(default_col) if default_col in cols_with_processed else 0,
            help="Pilih kolom yang sudah diproses sesuai kebutuhan"
        )
        
        # Pilih metode deteksi
        method = st.selectbox(
            "Metode Deteksi Anomali:",
            options=[
                "Threshold-based (IQR Historis)",
                "Isolation Forest (Pola Historis)",
                "Support Vector Machine (SVM)",  # Metode baru
                "Autoencoder (Deep Learning)",   # Metode baru
                "Adaptive Z-Score (Bulan Sebelumnya)",
                "Time Series Decomposition"
            ]
        )
        
        # Parameter metode
        params = {}
        if method == "Threshold-based (IQR Historis)":
            params['threshold'] = st.slider("Threshold IQR:", 1.0, 3.0, 1.5, step=0.1)
        elif method == "Isolation Forest (Pola Historis)":
            params['contamination'] = st.slider("Estimasi Kontaminasi:", 0.01, 0.2, 0.05, step=0.01)
        elif method == "Support Vector Machine (SVM)":
            params['nu'] = st.slider("Nu (Batas Atas Fraksi Outlier):", 0.01, 0.2, 0.05, step=0.01)
            params['kernel'] = st.selectbox("Kernel:", options=["rbf", "linear", "poly", "sigmoid"], index=0)
        elif method == "Autoencoder (Deep Learning)":
            params['epochs'] = st.slider("Epochs:", 10, 100, 50, step=5)
            params['threshold_percentile'] = st.slider("Threshold Percentile:", 90, 99, 95, step=1)
            params['hidden_dim'] = st.slider("Hidden Layer Dimension:", 4, 32, 8, step=4)
        elif method == "Adaptive Z-Score (Bulan Sebelumnya)":
            params['z_threshold'] = st.slider("Threshold Z-Score:", 2.0, 5.0, 3.0)
            params['window'] = st.slider("Window Size (hari):", 7, 30, 14)
        
        if st.button("🔍 Deteksi Anomali", key="detect_button"):
            with st.spinner("Mendeteksi anomali..."):
                if method == "Threshold-based (IQR Historis)":
                    # Hitung IQR dari data historis
                    Q1 = full_data[analysis_col].quantile(0.25)
                    Q3 = full_data[analysis_col].quantile(0.75)
                    IQR = Q3 - Q1
                    threshold = params['threshold']
                    
                    # Tentukan batas
                    lower_bound = Q1 - (threshold * IQR)
                    upper_bound = Q3 + (threshold * IQR)
                    
                    # Deteksi anomaly pada 30 hari terakhir
                    df_last30['Anomaly'] = ((df_last30[analysis_col] < lower_bound) | 
                                          (df_last30[analysis_col] > upper_bound)).astype(int)
                
                elif method == "Isolation Forest (Pola Historis)":
                    # Latih model menggunakan data historis
                    model = IsolationForest(
                        contamination=params['contamination'],
                        random_state=42
                    )
                    model.fit(full_data[[analysis_col]])
                    
                    # Prediksi pada 30 hari terakhir
                    df_last30['Anomaly'] = model.predict(df_last30[[analysis_col]])
                    df_last30['Anomaly'] = df_last30['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
                
                elif method == "Support Vector Machine (SVM)":
                    # Import yang diperlukan
                    from sklearn.svm import OneClassSVM
                    from sklearn.preprocessing import StandardScaler
                    
                    # Preprocessing - Standardize data
                    scaler = StandardScaler()
                    historical_data = full_data[full_data[date_column] < start_date]
                    
                    # Fit scaler pada data historis
                    scaler.fit(historical_data[[analysis_col]])
                    
                    # Transform data historis dan data 30 hari terakhir
                    X_train_scaled = scaler.transform(historical_data[[analysis_col]])
                    X_test_scaled = scaler.transform(df_last30[[analysis_col]])
                    
                    # Latih model
                    svm_model = OneClassSVM(
                        nu=params['nu'],
                        kernel=params['kernel'],
                        gamma='scale'
                    )
                    svm_model.fit(X_train_scaled)
                    
                    # Prediksi anomali pada 30 hari terakhir
                    df_last30['Anomaly'] = svm_model.predict(X_test_scaled)
                    df_last30['Anomaly'] = df_last30['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
                    
                    # Hitung skor anomali jika diperlukan
                    df_last30['anomaly_score'] = -svm_model.decision_function(X_test_scaled)
                
                elif method == "Autoencoder (Deep Learning)":
                    import numpy as np
                    import tensorflow as tf
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Input
                    from tensorflow.keras.callbacks import EarlyStopping
                    from sklearn.preprocessing import MinMaxScaler
                    
                    # Preprocessing - Scale data to [0,1]
                    scaler = MinMaxScaler()
                    historical_data = full_data[full_data[date_column] < start_date]
                    
                    # Fit scaler pada data historis
                    historical_values = historical_data[[analysis_col]].values
                    scaler.fit(historical_values)
                    
                    # Transform data historis dan data 30 hari terakhir
                    X_train_scaled = scaler.transform(historical_values)
                    X_test_scaled = scaler.transform(df_last30[[analysis_col]].values)
                    
                    # Buat model autoencoder
                    input_dim = X_train_scaled.shape[1]  # Jumlah fitur
                    encoding_dim = params['hidden_dim']  # Dimensi layer tersembunyi
                    
                    autoencoder = Sequential([
                        # Encoder
                        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
                        # Decoder
                        Dense(input_dim, activation='sigmoid')
                    ])
                    
                    # Kompilasi model
                    autoencoder.compile(optimizer='adam', loss='mse')
                    
                    # Callback untuk early stopping
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                    
                    # Latih model dengan data historis
                    with st.spinner("Melatih model Autoencoder..."):
                        history = autoencoder.fit(
                            X_train_scaled, X_train_scaled,  # Input = Output untuk autoencoder
                            epochs=params['epochs'],
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                    
                    # Plot loss history jika diperlukan
                    loss_fig = px.line(
                        x=range(1, len(history.history['loss'])+1),
                        y=history.history['loss'],
                        labels={'x': 'Epoch', 'y': 'Loss'},
                        title='Training Loss'
                    )
                    st.plotly_chart(loss_fig, use_container_width=True)
                    
                    # Rekonstruksi dan hitung error pada data historis
                    X_train_pred = autoencoder.predict(X_train_scaled)
                    train_mae = np.mean(np.abs(X_train_pred - X_train_scaled), axis=1)
                    
                    # Tentukan threshold berdasarkan persentil dari reconstruction error
                    threshold = np.percentile(train_mae, params['threshold_percentile'])
                    
                    # Rekonstruksi dan hitung error pada data 30 hari terakhir
                    X_test_pred = autoencoder.predict(X_test_scaled)
                    test_mae = np.mean(np.abs(X_test_pred - X_test_scaled), axis=1)
                    
                    # Deteksi anomali
                    df_last30['reconstruction_error'] = test_mae
                    df_last30['Anomaly'] = (test_mae > threshold).astype(int)
                
                elif method == "Adaptive Z-Score (Bulan Sebelumnya)":
                    window = params['window']
                    threshold = params['z_threshold']
                    
                    # Hitung rolling mean dan std dari data sebelum 30 hari terakhir
                    historical_data = full_data[full_data[date_column] < start_date]
                    
                    if len(historical_data) > 0:
                        # Hitung statistik dari bulan sebelumnya
                        prev_month_mean = historical_data[analysis_col].mean()
                        prev_month_std = historical_data[analysis_col].std()
                        
                        # Hitung z-score berdasarkan statistik bulan sebelumnya
                        df_last30['z_score'] = (df_last30[analysis_col] - prev_month_mean) / prev_month_std
                        df_last30['Anomaly'] = (df_last30['z_score'].abs() > threshold).astype(int)
                    else:
                        st.warning("Tidak cukup data historis untuk perbandingan")
                        df_last30['Anomaly'] = 0
                
                elif method == "Time Series Decomposition":
                    # Implementasi sederhana dari decomposition (hanya deteksi trend deviasi)
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Pastikan indeks adalah datetime untuk decomposition
                    temp_data = full_data.set_index(date_column)[[analysis_col]].copy()
                    
                    # Pastikan indeks berurutan untuk decomposition
                    temp_data = temp_data.asfreq('D', method='ffill')  # Gunakan frekuensi harian
                    
                    try:
                        # Decompose data
                        decomposition = seasonal_decompose(temp_data, model='additive', period=30)
                        
                        # Dapatkan residual
                        residual = decomposition.resid
                        
                        # Hitung batas anomali (mean ± 2*std dari residual)
                        residual_mean = residual.mean()[0]
                        residual_std = residual.std()[0]
                        
                        # Tentukan batas
                        upper_bound = residual_mean + 2 * residual_std
                        lower_bound = residual_mean - 2 * residual_std
                        
                        # Reset index untuk penggabungan
                        residual = residual.reset_index()
                        
                        # Gabungkan dengan data 30 hari terakhir
                        df_last30 = df_last30.merge(
                            residual,
                            on=date_column,
                            how='left',
                            suffixes=('', '_residual')
                        )
                        
                        # Deteksi anomali
                        df_last30['Anomaly'] = (
                            (df_last30['resid'] > upper_bound) | 
                            (df_last30['resid'] < lower_bound)
                        ).astype(int)
                        
                    except Exception as e:
                        st.error(f"Error dalam melakukan decomposition: {str(e)}")
                        st.info("Mencoba dengan metode alternatif...")
                        
                        # Metode alternatif jika decomposition gagal
                        rolling_mean = full_data[analysis_col].rolling(window=14, min_periods=1).mean()
                        rolling_std = full_data[analysis_col].rolling(window=14, min_periods=1).std()
                        
                        # Hitung z-score relatif terhadap rolling mean/std
                        df_last30['z_score'] = (df_last30[analysis_col] - rolling_mean.loc[df_last30.index]) / rolling_std.loc[df_last30.index]
                        df_last30['Anomaly'] = (df_last30['z_score'].abs() > 2.5).astype(int)
                
                # Simpan hasil
                st.session_state.anomaly_results = df_last30
                
                # Visualisasi
                fig = px.line(
                    df_last30,
                    x=date_column,
                    y=analysis_col,
                    title=f"Anomali Terdeteksi ({method}) - 30 Hari Terakhir",
                    template='plotly_dark'
                )
                
                # Tambahkan anomali
                anomalies = df_last30[df_last30['Anomaly'] == 1]
                fig.add_scatter(
                    x=anomalies[date_column],
                    y=anomalies[analysis_col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8)
                )
                
                # Tambahkan threshold jika ada
                if method == "Threshold-based (IQR Historis)":
                    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                                 annotation_text=f"Upper Bound (Q3 + {threshold}*IQR)")
                    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                                 annotation_text=f"Lower Bound (Q1 - {threshold}*IQR)")
                
                # Tambahkan visualization khusus untuk metode SVM dan Autoencoder
                if method == "Support Vector Machine (SVM)" and 'anomaly_score' in df_last30.columns:
                    score_fig = px.line(
                        df_last30,
                        x=date_column,
                        y='anomaly_score',
                        title="SVM Anomaly Score",
                        template='plotly_dark'
                    )
                    # Highlight threshold
                    decision_threshold = 0  # SVM decision boundary
                    score_fig.add_hline(y=decision_threshold, line_dash="dash", line_color="yellow",
                                      annotation_text="Decision Boundary")
                    st.plotly_chart(score_fig, use_container_width=True)
                    
                elif method == "Autoencoder (Deep Learning)" and 'reconstruction_error' in df_last30.columns:
                    error_fig = px.line(
                        df_last30,
                        x=date_column,
                        y='reconstruction_error',
                        title="Reconstruction Error",
                        template='plotly_dark'
                    )
                    # Highlight threshold
                    error_fig.add_hline(y=threshold, line_dash="dash", line_color="yellow",
                                      annotation_text=f"Error Threshold (percentil ke-{params['threshold_percentile']})")
                    st.plotly_chart(error_fig, use_container_width=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan hasil
                st.markdown(f"**Total Anomali Terdeteksi:** {df_last30['Anomaly'].sum()}")
                
                if df_last30['Anomaly'].sum() > 0:
                    st.markdown("### Detail Anomali")
                    st.dataframe(
                        anomalies[[date_column, analysis_col]].sort_values(date_column),
                        height=300
                    )
                    
                    # AI Insight dengan fokus pada model dan hasil deteksi anomali - lebih terstruktur
                    with st.expander("🤖 Insight Deteksi Anomali"):
                        with st.spinner("Menganalisis anomali..."):
                            # Hitung statistik historis
                            historical_data = full_data[full_data[date_column] < start_date]
                            historical_stats = {
                                'mean': historical_data[analysis_col].mean(),
                                'std': historical_data[analysis_col].std(),
                                'min': historical_data[analysis_col].min(),
                                'max': historical_data[analysis_col].max(),
                                'count': len(historical_data),
                                'period': f"{historical_data[date_column].min().strftime('%Y-%m-%d')} hingga {historical_data[date_column].max().strftime('%Y-%m-%d')}"
                            }
                            
                            # Statistik anomali
                            anomaly_stats = {
                                'count': len(anomalies),
                                'percent': f"{(len(anomalies) / len(df_last30) * 100):.2f}%",
                                'mean': anomalies[analysis_col].mean() if len(anomalies) > 0 else 0,
                                'std': anomalies[analysis_col].std() if len(anomalies) > 0 else 0,
                                'min': anomalies[analysis_col].min() if len(anomalies) > 0 else 0,
                                'max': anomalies[analysis_col].max() if len(anomalies) > 0 else 0,
                                'dates': anomalies[date_column].dt.strftime('%Y-%m-%d %H:%M').tolist()[:5] + (['...'] if len(anomalies) > 5 else []),
                                'values': anomalies[analysis_col].tolist()[:5] + (['...'] if len(anomalies) > 5 else [])
                            }
                            
                            # Informasi model
                            model_descriptions = {
                                "Isolation Forest (Pola Historis)": "Mendeteksi anomali dengan cara mengisolasi data yang paling mudah dipisahkan.",
                                "Support Vector Machine (SVM)": "Membuat batas keputusan di sekitar data normal dan mengidentifikasi data di luar batas.",
                                "Autoencoder (Deep Learning)": "Mempelajari pola normal, lalu mendeteksi anomali dari data yang sulit direkonstruksi.",
                                "Threshold-based (IQR Historis)": "Mendeteksi anomali berdasarkan batas atas dan bawah dari rentang antar-kuartil.",
                                "Adaptive Z-Score (Bulan Sebelumnya)": "Mendeteksi anomali berdasarkan seberapa jauh data dari rata-rata historis.",
                                "Time Series Decomposition": "Memisahkan data menjadi komponen trend, musiman, dan residual untuk deteksi anomali."
                            }
                            
                            # ====================
                        # 📊 EVALUASI MODEL
                        # ====================
                        st.markdown("## 📊 Evaluasi Model")

                        # Tambahkan CSS untuk gaya metrik modern
                        st.markdown("""
                            <style>
                            .metric-card {
                                background-color: #1e1e1e;
                                padding: 20px;
                                border-radius: 10px;
                                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                                margin-bottom: 15px;
                                color: white;
                                text-align: center;
                            }
                            .metric-title {
                                font-size: 14px;
                                color: #9ca3af;
                                margin-bottom: 5px;
                            }
                            .metric-value {
                                font-size: 28px;
                                font-weight: bold;
                                color: #10b981;
                            }
                            .metric-help {
                                font-size: 12px;
                                color: #d1d5db;
                                margin-top: 5px;
                            }
                            </style>
                        """, unsafe_allow_html=True)

                        # Hanya lanjut jika data historis tersedia
                        if len(historical_data) > 0:
                            # Prediksi untuk evaluasi
                            if method == "Threshold-based (IQR Historis)":
                                historical_anomalies = ((historical_data[analysis_col] < lower_bound) | 
                                                        (historical_data[analysis_col] > upper_bound)).sum()
                            elif method == "Isolation Forest (Poli Historis)":
                                historical_pred = model.predict(historical_data[[analysis_col]])
                                historical_anomalies = (historical_pred == -1).sum()
                            elif method == "Support Vector Machine (SVM)":
                                X_hist_scaled = scaler.transform(historical_data[[analysis_col]])
                                historical_pred = svm_model.predict(X_hist_scaled)
                                historical_anomalies = (historical_pred == -1).sum()
                            elif method == "Autoencoder (Deep Learning)":
                                X_hist_scaled = scaler.transform(historical_data[[analysis_col]].values)
                                X_hist_pred = autoencoder.predict(X_hist_scaled)
                                hist_mae = np.mean(np.abs(X_hist_pred - X_hist_scaled), axis=1)
                                historical_anomalies = (hist_mae > threshold).sum()
                            else:
                                historical_anomalies = 0

                            # Metrik
                            total_historical = len(historical_data)
                            total_recent = len(df_last30)
                            fpr = historical_anomalies / total_historical if total_historical > 0 else 0
                            detection_rate = len(anomalies) / total_recent if total_recent > 0 else 0

                            # Estimasi metrik
                            expected_anomaly_rate = 0.01
                            assumed_tp = len(anomalies) * (1 - expected_anomaly_rate)
                            assumed_fp = len(anomalies) * expected_anomaly_rate
                            precision = assumed_tp / (assumed_tp + historical_anomalies) if (assumed_tp + historical_anomalies) > 0 else 0

                            assumed_total_real_anomalies = total_historical * expected_anomaly_rate
                            recall = assumed_tp / assumed_total_real_anomalies if assumed_total_real_anomalies > 0 else 0

                            from statsmodels.stats.proportion import proportion_confint
                            ci_low, ci_high = proportion_confint(assumed_tp, len(anomalies), alpha=0.05)

                            # TAMPILKAN METRIK DALAM BENTUK KARTU
                            col1, col2, col3 = st.columns(3)
                            col4, col5, _ = st.columns([1, 1, 0.5])  # buat space kanan

                            with col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">🧪 False Positive Rate</div>
                                        <div class="metric-value">{fpr*100:.1f}%</div>
                                        <div class="metric-help">Idealnya &lt; 5%</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">📈 Detection Rate</div>
                                        <div class="metric-value">{detection_rate*100:.1f}%</div>
                                        <div class="metric-help">Deteksi anomali dalam 30 hari terakhir</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            with col3:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">🎯 Precision (Estimasi)</div>
                                        <div class="metric-value">{precision*100:.1f}%</div>
                                        <div class="metric-help">CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            with col4:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">🔍 Recall (Estimasi)</div>
                                        <div class="metric-value">{recall*100:.1f}%</div>
                                        <div class="metric-help">Kemampuan mendeteksi anomali</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            with col5:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-title">📚 Data Historis</div>
                                        <div class="metric-value">{total_historical:,}</div>
                                        <div class="metric-help">Total data evaluasi</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            # INTERPRETASI HASIL
                            st.markdown("### 📌 Interpretasi Hasil Evaluasi")

                            # FPR
                            if fpr < 0.05:
                                st.success("✅ **False Positive Rate Rendah**: Model jarang salah klasifikasi.")
                            elif fpr < 0.15:
                                st.warning("⚠️ **False Positive Rate Sedang**: Ada potensi kesalahan klasifikasi.")
                            else:
                                st.error("❌ **False Positive Rate Tinggi**: Model sering salah klasifikasi data normal.")

                            # Detection Rate
                            if detection_rate < 0.05:
                                st.info("ℹ️ **Detection Rate Rendah**: Sedikit data terdeteksi sebagai anomali.")
                            elif detection_rate < 0.15:
                                st.info("ℹ️ **Detection Rate Sedang**: Deteksi anomali moderat.")
                            else:
                                st.warning("⚠️ **Detection Rate Tinggi**: Banyak data dianggap anomali, evaluasi threshold diperlukan.")

                            # Precision
                            if precision > 0.8:
                                st.success(f"✅ **Precision Tinggi**: Mayoritas deteksi benar ({ci_low*100:.1f}%–{ci_high*100:.1f}%).")
                            elif precision > 0.5:
                                st.warning(f"⚠️ **Precision Sedang**: Beberapa deteksi mungkin salah ({ci_low*100:.1f}%–{ci_high*100:.1f}%).")
                            else:
                                st.error(f"❌ **Precision Rendah**: Banyak deteksi kemungkinan salah ({ci_low*100:.1f}%–{ci_high*100:.1f}%).")

                            # Recall
                            if recall > 0.8:
                                st.success("✅ **Recall Tinggi**: Sebagian besar anomali berhasil dideteksi.")
                            elif recall > 0.5:
                                st.warning("⚠️ **Recall Sedang**: Beberapa anomali mungkin terlewat.")
                            else:
                                st.error("❌ **Recall Rendah**: Banyak anomali tidak terdeteksi oleh model.")

                            # VISUALISASI TAMBAHAN
                            st.markdown("### 📉 Visualisasi Distribusi Error")

                            if method == "Autoencoder (Deep Learning)":
                                df_last30['reconstruction_error'] = np.mean(np.abs(autoencoder.predict(scaler.transform(df_last30[[analysis_col]])) - scaler.transform(df_last30[[analysis_col]])), axis=1)
                                error_col = 'reconstruction_error'
                            else:
                                df_last30['error'] = (df_last30[analysis_col] - df_last30[analysis_col].mean()).abs()
                                error_col = 'error'

                            fig_dist = px.histogram(
                                df_last30,
                                x=error_col,
                                color='Anomaly',
                                nbins=50,
                                title='Distribusi Error/Nilai dengan Anomali',
                                template='plotly_dark'
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                            st.markdown(f"""
                            **📝 Keterangan Metrik:**
                            - **False Positive Rate (FPR):** Persentase data normal yang salah diklasifikasikan. Idealnya <5%.
                            - **Detection Rate:** Seberapa banyak data dianggap anomali.
                            - **Precision:** Akurasi deteksi anomali (Estimasi).
                            - **Recall:** Cakupan deteksi anomali dari semua yang mungkin terjadi.
                            - *Asumsi: {expected_anomaly_rate*100}% dari data sebenarnya adalah anomali*
                            - *CI 95% menunjukkan ketidakpastian estimasi precision*
                            """)

                            # Penjelasan singkat model dalam bahasa sederhana
                            model_simple = model_descriptions.get(method, "Metode deteksi anomali berdasarkan pola data historis.")
                            
                            # Grafik tambahan jika diperlukan
                            if len(anomalies) > 0:
                                # Pengelompokan anomali (tinggi/rendah)
                                high_anomalies = anomalies[anomalies[analysis_col] > historical_stats['mean']].shape[0]
                                low_anomalies = anomalies[anomalies[analysis_col] < historical_stats['mean']].shape[0]
                                
                                # Buat data untuk chart tipe anomali
                                anomaly_types = {
                                    'Tipe': ['Nilai Tinggi', 'Nilai Rendah'],
                                    'Jumlah': [high_anomalies, low_anomalies]
                                }
                                
                                # Tampilkan chart tipe anomali
                                if high_anomalies > 0 or low_anomalies > 0:
                                    fig_type = px.bar(
                                        anomaly_types,
                                        x='Tipe',
                                        y='Jumlah',
                                        title="Tipe Anomali Terdeteksi",
                                        color='Tipe',
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_type, use_container_width=True)
                            
                            # Buat insight yang terstruktur dan mudah dibaca
                            st.markdown("""
                            # 🔍 Hasil Deteksi Anomali
                            """)
                            
                            # Tampilkan kartu rangkuman
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    label="Total Anomali", 
                                    value=anomaly_stats['count']
                                )
                            with col2:
                                st.metric(
                                    label="Persentase Data", 
                                    value=anomaly_stats['percent']
                                )
                            with col3:
                                if len(anomalies) > 0:
                                    deviation = ((anomaly_stats['mean'] - historical_stats['mean'])/historical_stats['mean']*100)
                                    st.metric(
                                        label="Deviasi Rata-rata", 
                                        value=f"{deviation:.1f}%",
                                        delta=f"{deviation:.1f}%"
                                    )
                                else:
                                    st.metric(label="Deviasi Rata-rata", value="0%")
                            
                            # Bagian 1: Penjelasan model yang digunakan
                            st.markdown("""
                            ## 💡 Penjelasan Model
                            """)
                            
                            st.info(f"""
                            **Model yang digunakan: {method}**
                            
                            {model_simple}
                            """)
                            
                            # Tampilkan parameter model yang digunakan dalam format yang mudah dibaca
                            st.markdown("### Parameter Model")
                            
                            # Format parameter berdasarkan tipe model
                            param_descriptions = []
                            for key, value in params.items():
                                if method == "Threshold-based (IQR Historis)" and key == "threshold":
                                    param_descriptions.append(f"**Faktor IQR:** {value} (semakin besar nilai, semakin sedikit anomali terdeteksi)")
                                elif method == "Isolation Forest (Pola Historis)" and key == "contamination":
                                    param_descriptions.append(f"**Perkiraan Persentase Anomali:** {value*100}%")
                                elif method == "Support Vector Machine (SVM)":
                                    if key == "nu":
                                        param_descriptions.append(f"**Perkiraan Persentase Anomali:** {value*100}%")
                                    elif key == "kernel":
                                        param_descriptions.append(f"**Tipe Kernel:** {value}")
                                elif method == "Autoencoder (Deep Learning)":
                                    if key == "epochs":
                                        param_descriptions.append(f"**Jumlah Pelatihan:** {value} kali")
                                    elif key == "threshold_percentile":
                                        param_descriptions.append(f"**Ambang Batas Persentil:** {value}%")
                                    elif key == "hidden_dim":
                                        param_descriptions.append(f"**Ukuran Lapisan Tersembunyi:** {value}")
                                elif method == "Adaptive Z-Score (Bulan Sebelumnya)":
                                    if key == "z_threshold":
                                        param_descriptions.append(f"**Ambang Batas Z-Score:** {value}")
                                    elif key == "window":
                                        param_descriptions.append(f"**Jendela Data (hari):** {value}")
                                else:
                                    param_descriptions.append(f"**{key}:** {value}")
                            
                            # Tampilkan parameter dalam bullet points
                            for desc in param_descriptions:
                                st.markdown(f"- {desc}")
                            
                            # Bagian 2: Hasil Analisis
                            if len(anomalies) > 0:
                                st.markdown("""
                                ## 📊 Analisis Anomali
                                """)
                                
                                # Tampilkan perbandingan data normal vs anomali
                                st.markdown("### Perbandingan Nilai")
                                compare_cols = st.columns(2)
                                with compare_cols[0]:
                                    st.markdown("**Data Normal:**")
                                    st.markdown(f"- Rata-rata: **{historical_stats['mean']:.2f}**")
                                    st.markdown(f"- Rentang: {historical_stats['min']:.2f} - {historical_stats['max']:.2f}")
                                
                                with compare_cols[1]:
                                    st.markdown("**Data Anomali:**")
                                    if len(anomalies) > 0:
                                        st.markdown(f"- Rata-rata: **{anomaly_stats['mean']:.2f}**")
                                        st.markdown(f"- Rentang: {anomaly_stats['min']:.2f} - {anomaly_stats['max']:.2f}")
                                    else:
                                        st.markdown("- Tidak ada anomali terdeteksi")
                                
                                # Tampilkan tanggal-tanggal anomali
                                st.markdown("### Waktu Anomali Terdeteksi")
                                if len(anomalies) > 0:
                                    # Dapatkan 5 tanggal pertama
                                    dates_to_show = anomalies.sort_values(date_column)[date_column].dt.strftime('%d %b %Y, %H:%M').tolist()[:5]
                                    for i, date in enumerate(dates_to_show):
                                        st.markdown(f"{i+1}. **{date}**")
                                    if len(anomalies) > 5:
                                        st.markdown(f"... dan {len(anomalies)-5} waktu lainnya")
                                else:
                                    st.markdown("Tidak ada anomali terdeteksi")
                                
                                # Bagian 3: Kesimpulan dan Rekomendasi
                                st.markdown("""
                                ## 🎯 Kesimpulan dan Rekomendasi
                                """)
                                # ANALISIS KORELASI BARU - TAMBAHKAN DI SINI
                                st.markdown("### 🔗 Analisis Korelasi Antar Parameter")

                                # Hitung matriks korelasi untuk semua parameter numerik
                                numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
                                correlation_data = []  # Inisialisasi list untuk menyimpan data korelasi

                                if len(numeric_cols) > 1:
                                    # Hitung korelasi
                                    corr_matrix = full_data[numeric_cols].corr()
                                    
                                    # Cari parameter dengan korelasi terkuat dengan parameter yang dianalisis
                                    if analysis_col in corr_matrix.columns:
                                        # Dapatkan korelasi dengan parameter yang sedang dianalisis
                                        target_correlations = corr_matrix[analysis_col].drop(analysis_col)
                                        
                                        # Urutkan berdasarkan nilai absolut korelasi (terkuat ke terlemah)
                                        sorted_correlations = target_correlations.abs().sort_values(ascending=False)
                                        
                                        # Ambil 5 parameter dengan korelasi terkuat
                                        top_correlations = sorted_correlations.head(5)
                                        
                                        st.markdown(f"**Parameter dengan Korelasi Terkuat terhadap {analysis_col}:**")
                                        
                                        # Simpan data korelasi untuk digunakan di AI insight
                                        for param, corr_value in target_correlations[top_correlations.index].items():
                                            strength = "sangat kuat" if abs(corr_value) > 0.7 else "kuat" if abs(corr_value) > 0.5 else "sedang" if abs(corr_value) > 0.3 else "lemah"
                                            direction = "positif" if corr_value > 0 else "negatif"
                                            
                                            # Simpan data korelasi
                                            explanation = f"Peningkatan {param} biasanya diikuti peningkatan {analysis_col}" if corr_value > 0 else f"Peningkatan {param} biasanya diikuti penurunan {analysis_col}"
                                            correlation_data.append((param, corr_value, direction, strength, explanation))
                                            
                                            st.markdown(f"- **{param}**: {corr_value:.3f} ({direction}, {strength})")
                                            
                                            # Berikan interpretasi singkat untuk korelasi yang signifikan
                                            if abs(corr_value) > 0.5:
                                                if corr_value > 0:
                                                    st.markdown(f"  → Peningkatan {param} biasanya diikuti peningkatan {analysis_col}")
                                                else:
                                                    st.markdown(f"  → Peningkatan {param} biasanya diikuti penurunan {analysis_col}")
                                    
                                    # Tampilkan heatmap korelasi
                                    fig_corr = px.imshow(
                                        corr_matrix,
                                        text_auto=True,
                                        aspect="auto",
                                        color_continuous_scale='RdBu_r',
                                        template='plotly_dark',
                                        title='Matriks Korelasi Antar Parameter'
                                    )
                                    fig_corr.update_layout(height=600)
                                    st.plotly_chart(fig_corr, use_container_width=True)
                                else:
                                    st.warning("Tidak cukup parameter numerik untuk analisis korelasi")

                                # KESIMPULAN AWAL (tetap pertahankan)
                                if len(anomalies) > 0:
                                    # Siapkan metrik untuk analisis
                                    anomaly_metrics = {
                                        'count': len(anomalies),
                                        'percent': anomaly_stats['percent'],
                                        'high_count': high_anomalies,
                                        'low_count': low_anomalies,
                                        'mean': anomaly_stats['mean'],
                                        'hist_mean': historical_stats['mean'],
                                        'deviation': ((anomaly_stats['mean'] - historical_stats['mean'])/historical_stats['mean']*100) if historical_stats['mean'] != 0 else 0
                                    }
                                    
                                    # Buat kesimpulan singkat berdasarkan hasil
                                    if high_anomalies > low_anomalies:
                                        conclusion = f"Mayoritas anomali ({high_anomalies} dari {len(anomalies)}) menunjukkan **nilai yang lebih tinggi** dari pola normal."
                                    elif low_anomalies > high_anomalies:
                                        conclusion = f"Mayoritas anomali ({low_anomalies} dari {len(anomalies)}) menunjukkan **nilai yang lebih rendah** dari pola normal."
                                    else:
                                        conclusion = "Anomali terdeteksi memiliki jumlah yang seimbang antara nilai tinggi dan rendah."
                                    
                                    st.markdown(f"""
                                    **Kesimpulan Umum:**
                                    - {conclusion}
                                    - Sebanyak {anomaly_stats['percent']} data dalam 30 hari terakhir terdeteksi sebagai anomali.
                                    """)
                                    
                                    # Tampilkan spinner selama mendapatkan analisis AI
                                    with st.spinner("Menganalisis anomali dengan AI..."):
                                        # Dapatkan insight dari AI dengan menyertakan data korelasi
                                        ai_analysis = get_anomaly_insight(
                                            parameter=analysis_col,
                                            anomaly_data=anomalies,
                                            method=method,
                                            metrics=anomaly_metrics,
                                            correlation_data=correlation_data  # Kirim data korelasi ke AI
                                        )
                                        
                                        # Tampilkan insight dengan styling yang bagus
                                        st.markdown("<div style='background-color:#1E1E1E; padding:15px; border-radius:10px; border-left:4px solid #FFA500;'>", unsafe_allow_html=True)
                                        st.markdown(ai_analysis, unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)

                                else:
                                    st.success("Tidak ada anomali yang terdeteksi dalam periode 30 hari terakhir dengan parameter model saat ini.")
                                    
                                    st.markdown("""
                                    ## 🎯 Rekomendasi
                                    
                                    Tidak ditemukan anomali dengan konfigurasi saat ini. Beberapa saran:
                                    
                                    1. Coba sesuaikan parameter model untuk threshold yang lebih sensitif
                                    2. Coba model deteksi anomali lainnya
                                    3. Periksa apakah periode 30 hari terakhir memang memiliki pola normal
                                    """)
                            
 # TAMBAHKAN INPUT CHAT UNTUK ANALISIS AI DI BAWAH TOMBOL DETEKSI ANOMALI
        if st.session_state.get('anomaly_results') is not None:
            st.markdown("---")
            st.markdown("### 💬 Chat Analisis AI Lanjutan")
            
            # Input chat untuk analisis tambahan
            user_prompt = st.text_area(
                "Ajukan pertanyaan spesifik tentang anomali yang terdeteksi:",
                placeholder="Contoh: 'Analisis penyebab anomali pada tanggal 15 Januari' atau 'Bagaimana korelasi antara parameter ini dengan tekanan boiler?'",
                height=100,
                key="ai_chat_input"
            )
            
            if st.button("🤖 Analisis dengan AI", key="chat_analysis_button") and user_prompt:
                with st.spinner("Menganalisis dengan AI..."):
                    # Dapatkan data anomali
                    anomalies = st.session_state.anomaly_results[st.session_state.anomaly_results['Anomaly'] == 1]
                    
                    # Siapkan data untuk analisis AI
                    historical_data = full_data[full_data[date_column] < start_date]
                    historical_stats = {
                        'mean': historical_data[analysis_col].mean(),
                        'std': historical_data[analysis_col].std(),
                        'min': historical_data[analysis_col].min(),
                        'max': historical_data[analysis_col].max(),
                    }
                    
                    anomaly_metrics = {
                        'count': len(anomalies),
                        'percent': f"{(len(anomalies) / len(st.session_state.anomaly_results) * 100):.2f}%",
                        'mean': anomalies[analysis_col].mean() if len(anomalies) > 0 else 0,
                    }
                    
                    # Hitung korelasi untuk analisis tambahan
                    numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
                    correlation_data = []
                    if len(numeric_cols) > 1 and analysis_col in numeric_cols:
                        corr_matrix = full_data[numeric_cols].corr()
                        if analysis_col in corr_matrix.columns:
                            target_correlations = corr_matrix[analysis_col].drop(analysis_col)
                            sorted_correlations = target_correlations.abs().sort_values(ascending=False)
                            top_correlations = sorted_correlations.head(3)
                            
                            for param, corr_value in target_correlations[top_correlations.index].items():
                                strength = "sangat kuat" if abs(corr_value) > 0.7 else "kuat" if abs(corr_value) > 0.5 else "sedang" if abs(corr_value) > 0.3 else "lemah"
                                direction = "positif" if corr_value > 0 else "negatif"
                                explanation = f"Peningkatan {param} biasanya diikuti peningkatan {analysis_col}" if corr_value > 0 else f"Peningkatan {param} biasanya diikuti penurunan {analysis_col}"
                                correlation_data.append((param, corr_value, direction, strength, explanation))
                    
                    # Buat prompt khusus berdasarkan input user
                    enhanced_prompt = f"""
{user_prompt}

Konteks analisis:
- Parameter yang dianalisis: {analysis_col}
- Metode deteksi: {method}
- Jumlah anomali terdeteksi: {anomaly_metrics['count']}
- Rata-rata nilai anomali: {anomaly_metrics['mean']:.2f}
- Rata-rata nilai historis: {historical_stats['mean']:.2f}

Data korelasi dengan parameter lain:
"""
                    
                    for i, (param, corr_value, direction, strength, explanation) in enumerate(correlation_data):
                        enhanced_prompt += f"{i+1}. {param}: {corr_value:.3f} ({direction}, {strength}) - {explanation}\n"
                    
                    enhanced_prompt += "\nBerikan analisis mendalam yang spesifik dan rekomendasi teknis."
                    
                    try:
                        # Panggil AI dengan prompt yang ditingkatkan
                        response = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=3000,
                            temperature=0.7,
                            messages=[
                                {
                                    "role": "user", 
                                    "content": f"""
Kamu adalah ahli analisis PLTU dengan spesialisasi deteksi anomali. Analisis berikut berdasarkan pertanyaan spesifik dari operator.

PERTANYAAN OPERATOR: {user_prompt}

INFORMASI TEKNIS:
- Parameter: {analysis_col}
- Metode Deteksi: {method}
- Periode Analisis: 30 hari terakhir
- Anomali Terdeteksi: {anomaly_metrics['count']}
- Rata-rata Anomali: {anomaly_metrics['mean']:.2f} (Normal: {historical_stats['mean']:.2f})

KORELASI PARAMETER:
"""
                                }
                            ]
                        )
                        
                        # Format respons AI
                        ai_response = response.content[0].text
                        
                        # Tampilkan hasil dengan formatting yang baik
                        st.markdown("### 🔍 Hasil Analisis AI")
                        st.markdown(f"<div style='background-color:#1E1E1E; padding:20px; border-radius:10px; border-left:4px solid #4facfe;'>", unsafe_allow_html=True)
                        st.markdown(ai_response)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error dalam analisis AI: {str(e)}")
            
            # TAMBAHKAN JUGA OPSI PRE-DEFINED QUESTIONS
            st.markdown("**Pertanyaan contoh yang bisa diajukan:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apa penyebab utama anomali?", key="q1_button"):
                    st.session_state.user_prompt = "Apa penyebab utama anomali yang terdeteksi dan bagaimana kaitannya dengan parameter operasional lainnya?"
                
                if st.button("Rekomendasi perbaikan?", key="q2_button"):
                    st.session_state.user_prompt = "Beri rekomendasi teknis spesifik untuk menangani anomali yang terdeteksi, termasuk langkah-langkah yang harus diambil operator."
            
            with col2:
                if st.button("Dampak terhadap efisiensi?", key="q3_button"):
                    st.session_state.user_prompt = "Apa dampak anomali ini terhadap efisiensi pembangkit dan konsumsi bahan bakar?"
                
                if st.button("Korelasi dengan parameter lain?", key="q4_button"):
                    st.session_state.user_prompt = "Analisis korelasi antara anomali ini dengan parameter operasional lainnya yang terkait."
            
            # Set nilai input jika tombol pre-defined ditekan
            if 'user_prompt' in st.session_state:
                st.text_area(
                    "Ajukan pertanyaan spesifik tentang anomali yang terdeteksi:",
                    value=st.session_state.user_prompt,
                    height=100,
                    key="ai_chat_input_filled"
                )
# Fungsi helper diletakkan di luar blok tab
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


# ===================== TAB 4 =====================
    with tab4:
        st.header("🔮 Prediksi Parameter Berdasarkan Input")

    # Cek jika data sudah diproses
    if "processed_data" not in st.session_state:
        st.warning("Lakukan data preparation terlebih dahulu di tab sebelumnya")
        st.stop()

    try:
        full_data = st.session_state.full_data
        target_param = st.session_state.target_param

        st.markdown("""
        ### Fitur Prediksi Multivariat
        Pilih nilai untuk satu parameter, dan sistem akan memprediksi nilai parameter lainnya berdasarkan pola historis.
        """)

        # Pilih parameter input
        numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Diperlukan setidaknya 2 parameter numerik untuk melakukan prediksi")
            st.stop()

        default_input_param = target_param if target_param in numeric_cols else numeric_cols[0]

        input_param = st.selectbox(
            "Pilih Parameter Input:",
            options=numeric_cols,
            index=numeric_cols.index(default_input_param) if default_input_param in numeric_cols else 0,
            help="Parameter yang nilainya akan Anda tentukan",
            key="pred_input_param"
        )

        # Range nilai parameter
        param_min = float(full_data[input_param].min())
        param_max = float(full_data[input_param].max())
        param_mean = float(full_data[input_param].mean())

        input_value = st.slider(
            f"Nilai {input_param}:",
            min_value=param_min,
            max_value=param_max,
            value=param_mean,
            step=0.1 if (param_max - param_min) < 10 else 1.0,
            help=f"Rentang historis: {param_min:.2f} hingga {param_max:.2f}",
            key="pred_input_value"
        )

        # Metode prediksi
        prediction_method = st.selectbox(
            "Metode Prediksi:",
            options=[
                "Korelasi Linear Sederhana",
                "Random Forest Regression",
                "Gradient Boosting",
                "Neural Network"
            ],
            key="pred_method"
        )

        # Tambahan parameter
        if prediction_method in ["Random Forest Regression", "Gradient Boosting"]:
            n_estimators = st.slider("Jumlah Estimator:", 10, 200, 100, step=10, key="n_estimators_slider")

        # Tombol prediksi
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
                            st.warning(f"Data tidak cukup untuk prediksi {output_param}")
                            continue

                        if prediction_method == "Korelasi Linear Sederhana":
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X_clean, y_clean)
                            prediction = model.predict([[input_value]])[0]
                            r_squared = model.score(X_clean, y_clean)
                            results[output_param] = {
                                'prediction': prediction,
                                'r_squared': r_squared,
                                'method': 'Linear Regression'
                            }

                        elif prediction_method == "Random Forest Regression":
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                            model.fit(X_clean, y_clean)
                            prediction = model.predict([[input_value]])[0]
                            r_squared = model.score(X_clean, y_clean)
                            results[output_param] = {
                                'prediction': prediction,
                                'r_squared': r_squared,
                                'method': 'Random Forest'
                            }

                        elif prediction_method == "Gradient Boosting":
                            from sklearn.ensemble import GradientBoostingRegressor
                            model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
                            model.fit(X_clean, y_clean)
                            prediction = model.predict([[input_value]])[0]
                            r_squared = model.score(X_clean, y_clean)
                            results[output_param] = {
                                'prediction': prediction,
                                'r_squared': r_squared,
                                'method': 'Gradient Boosting'
                            }

                        elif prediction_method == "Neural Network":
                            from sklearn.neural_network import MLPRegressor
                            from sklearn.preprocessing import StandardScaler
                            scaler_X = StandardScaler()
                            scaler_y = StandardScaler()
                            X_scaled = scaler_X.fit_transform(X_clean)
                            y_scaled = scaler_y.fit_transform(y_clean.values.reshape(-1, 1))
                            model = MLPRegressor(
                                hidden_layer_sizes=(50, 25),
                                activation='relu',
                                solver='adam',
                                max_iter=1000,
                                random_state=42
                            )
                            model.fit(X_scaled, y_scaled.ravel())
                            input_scaled = scaler_X.transform([[input_value]])
                            prediction_scaled = model.predict(input_scaled)
                            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
                            r_squared = model.score(X_scaled, y_scaled)
                            results[output_param] = {
                                'prediction': prediction,
                                'r_squared': r_squared,
                                'method': 'Neural Network'
                            }

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
                    st.error(f"Error dalam melakukan prediksi: {str(e)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memuat data: {str(e)}")

    # ==== Tampilkan hasil jika ada ====
    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]
        input_param = st.session_state["prediction_input"]["input_param"]
        input_value = st.session_state["prediction_input"]["input_value"]
        prediction_method = st.session_state["prediction_input"]["method"]
        numeric_cols = st.session_state["prediction_input"]["numeric_cols"]
        full_data = st.session_state["prediction_input"]["full_data"]

        # Dataframe hasil
        result_data = []
        for param, values in results.items():
            result_data.append({
                'Parameter': param,
                'Prediksi': f"{values['prediction']:.2f}",
                'Akurasi (R²)': f"{values['r_squared']:.3f}",
                'Metode': values['method']
            })
        result_df = pd.DataFrame(result_data)

        st.markdown(f"### Hasil Prediksi untuk {input_param} = {input_value:.2f}")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Visualisasi
        st.markdown("### 📊 Visualisasi Prediksi")
        param_to_plot = st.selectbox(
            "Pilih parameter hasil prediksi yang ingin dibandingkan dengan input:",
            options=list(results.keys()),
            key="viz_param"
        )
        fig = px.scatter(
            full_data,
            x=input_param,
            y=param_to_plot,
            trendline="ols",
            title=f"Hubungan {input_param} vs {param_to_plot} (R²: {results[param_to_plot]['r_squared']:.3f})",
            template="plotly_dark"
        )
        fig.add_vline(
            x=input_value, line_dash="dash", line_color="red",
            annotation_text=f"Input: {input_value:.2f}"
        )
        fig.add_scatter(
            x=[input_value],
            y=[results[param_to_plot]['prediction']],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Prediksi'
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI Insight
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

                ai_insight = get_prediction_insight(
                    input_param=input_param,
                    input_value=input_value,
                    results=results,
                    correlations=top_correlations,
                    method=prediction_method
                )
                st.markdown(ai_insight)
            except Exception as e:
                st.error(f"Gagal membuat insight: {str(e)}")
# ========================== TAB 5: FORECASTING ==========================
    with tab5:
        st.header("📈 Forecasting Parameter ")

    # Pastikan data ada
    if "processed_data" not in st.session_state:
        st.warning("⚠️ Lakukan data preparation terlebih dahulu di tab Data Preparation")
        st.stop()

    try:
        full_data = st.session_state.full_data
        date_column = st.session_state.date_column
        full_data[date_column] = pd.to_datetime(full_data[date_column])

        # Pilih parameter numerik
        numeric_cols = full_data.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.error("❌ Tidak ada parameter numerik untuk forecasting")
            st.stop()

        # Preview data
        st.markdown("### 🔍 Preview Data")
        st.dataframe(full_data[[date_column] + numeric_cols].tail(10), use_container_width=True)

        # Input user
        target = st.selectbox("🎯 Pilih parameter untuk forecasting:", numeric_cols, key="forecast_param")
        horizon = st.slider("⏳ Jumlah langkah ke depan (30 menit):", 5, 1000, 20, key="forecast_horizon")
        method = st.selectbox(
            "⚙️ Metode forecasting:",
            ["ARIMA", "Prophet", "LSTM (Deep Learning)"],
            key="forecast_method"
        )

        if st.button("🚀 Jalankan Forecasting", key="forecast_button"):
            with st.spinner("⏳ Sedang melakukan forecasting..."):
                try:
                    series = full_data[[date_column, target]].dropna()
                    series.columns = ["ds", "y"]

                    # ================= PROPHET =================
                    if method == "Prophet":
                        from prophet import Prophet
                        import matplotlib.pyplot as plt

                        model = Prophet()
                        model.fit(series)

                        future = model.make_future_dataframe(periods=horizon, freq="30min")
                        forecast = model.predict(future)

                        plt.style.use("dark_background")
                        fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
                        ax.set_facecolor("black")

                        ax.plot(series['ds'], series['y'], color="orange", linewidth=1.5, alpha=0.7, label="Actual Data")
                        series_ma = series.copy()
                        series_ma['y_ma7'] = series_ma['y'].rolling(window=7).mean()
                        ax.plot(series_ma['ds'], series_ma['y_ma7'], color="gold", linewidth=2.5, label="MA7")
                        ax.plot(forecast['ds'], forecast['yhat'], color="cyan", linewidth=2.5, label="Forecast")
                        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                        color="deepskyblue", alpha=0.2, label="Confidence Interval")

                        ax.set_title("📈 Forecasting", fontsize=16, fontweight="bold", color="white")
                        ax.set_xlabel("Tanggal & Waktu", fontsize=12, color="white")
                        ax.set_ylabel("Nilai", fontsize=12, color="white")
                        ax.tick_params(axis="x", colors="white")
                        ax.tick_params(axis="y", colors="white")
                        ax.legend(loc="lower right", frameon=False, fontsize=10)

                        st.pyplot(fig)
                        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon), use_container_width=True)

                    # ================= ARIMA =================
                    elif method == "ARIMA":
                        from statsmodels.tsa.arima.model import ARIMA
                        model = ARIMA(series["y"], order=(5, 1, 0))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=horizon)

                        forecast_df = pd.DataFrame({
                            "ds": pd.date_range(start=series["ds"].iloc[-1], periods=horizon+1, freq="30min")[1:],
                            "forecast": forecast
                        })

                        import matplotlib.pyplot as plt
                        plt.style.use("dark_background")
                        fig, ax = plt.subplots(figsize=(12,6), facecolor="black")
                        ax.set_facecolor("black")

                        ax.plot(series["ds"], series["y"], color="orange", label="Actual Data")
                        ax.plot(forecast_df["ds"], forecast_df["forecast"], color="cyan", label="Forecast")
                        ax.set_title("📈 Forecasting dengan ARIMA", fontsize=16, color="white")
                        ax.legend()
                        ax.tick_params(axis="x", colors="white")
                        ax.tick_params(axis="y", colors="white")

                        st.pyplot(fig)
                        st.dataframe(forecast_df, use_container_width=True)

                    # ================= LSTM (Deep Learning) =================
                    elif method == "LSTM (Deep Learning)":
                        import numpy as np
                        import tensorflow as tf
                        from sklearn.preprocessing import MinMaxScaler
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                        data = series["y"].values.reshape(-1, 1)
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        data_scaled = scaler.fit_transform(data)

                        seq_len = 20  # panjang input sequence
                        X, y = [], []
                        for i in range(len(data_scaled) - seq_len):
                            X.append(data_scaled[i:i+seq_len])
                            y.append(data_scaled[i+seq_len])
                        X, y = np.array(X), np.array(y)

                        # Train/test split
                        split = int(len(X) * 0.8)
                        X_train, X_test = X[:split], X[split:]
                        y_train, y_test = y[:split], y[split:]

                        # Build LSTM model
                        model = tf.keras.Sequential([
                            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.LSTM(32),
                            tf.keras.layers.Dense(1)
                        ])
                        model.compile(optimizer="adam", loss="mse")

                        # Train
                        history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=0)

                        # Prediction
                        y_pred = model.predict(X_test)
                        y_pred_rescaled = scaler.inverse_transform(y_pred)
                        y_test_rescaled = scaler.inverse_transform(y_test)

                        # Forecast ke depan
                        last_seq = data_scaled[-seq_len:]
                        preds = []
                        current_seq = last_seq.reshape(1, seq_len, 1)

                        for _ in range(horizon):
                            next_val = model.predict(current_seq)[0]
                            preds.append(next_val)
                            current_seq = np.append(current_seq[:, 1:, :], [[next_val]], axis=1)

                        preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

                        forecast_df = pd.DataFrame({
                            "ds": pd.date_range(start=series["ds"].iloc[-1], periods=horizon+1, freq="30min")[1:],
                            "forecast": preds_rescaled.flatten()
                        })

                        # Plot hasil
                        plt.style.use("dark_background")
                        fig, ax = plt.subplots(figsize=(12, 6), facecolor="black")
                        ax.set_facecolor("black")

                        ax.plot(series["ds"], series["y"], color="orange", label="Actual Data")
                        ax.plot(forecast_df["ds"], forecast_df["forecast"], color="cyan", label="Forecast")
                        ax.set_title("🤖 Forecasting dengan LSTM (Deep Learning)", fontsize=16, color="white")
                        ax.legend()
                        ax.tick_params(axis="x", colors="white")
                        ax.tick_params(axis="y", colors="white")

                        st.pyplot(fig)
                        st.dataframe(forecast_df, use_container_width=True)

                        # Evaluasi metrik
                        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

                        st.success(f"✅ Evaluasi Model LSTM: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat forecasting: {str(e)}")

    except Exception as e:
        st.error(f"❌ Gagal menyiapkan data: {str(e)}")
