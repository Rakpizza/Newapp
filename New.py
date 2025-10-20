import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
from datetime import datetime

# ========= OCR Fallback Handling ========= #
try:
    import easyocr
    OCR_MODE = "EasyOCR"
except ModuleNotFoundError:
    try:
        from paddleocr import PaddleOCR
        OCR_MODE = "PaddleOCR"
    except ModuleNotFoundError:
        OCR_MODE = None

# ========= Streamlit Setup ========= #
st.set_page_config(page_title="DualAsset Analyzer Pro", page_icon="📊", layout="wide")
st.title("📊 DualAsset Analyzer Pro")
st.subheader("Auto-detect best Bybit Dual Asset offers (Buy Low / Sell High)")

# ========= OCR Loader ========= #
@st.cache_resource
def load_ocr():
    if OCR_MODE == "EasyOCR":
        return easyocr.Reader(['en'], gpu=False)
    elif OCR_MODE == "PaddleOCR":
        return PaddleOCR(use_angle_cls=True, lang='en')
    else:
        return None

reader = load_ocr()

if not reader:
    st.error("❌ No OCR module found. Please install EasyOCR or PaddleOCR.")
    st.stop()
else:
    st.success(f"OCR Engine Loaded: {OCR_MODE}")

# ========= Helper Functions ========= #
def extract_text(image):
    img_array = np.array(image)
    if OCR_MODE == "EasyOCR":
        results = reader.readtext(img_array, detail=0)
    else:
        results = [r[1] for r in reader.ocr(img_array, cls=True)[0]]
    return results

def parse_numbers(text):
    nums = re.findall(r'[\d,]+\.?\d*', text)
    return [float(n.replace(',', '')) for n in nums if n.replace(',', '').replace('.', '').isdigit()]

def detect_type(text):
    if 'sell' in text.lower():
        return 'Sell High'
    elif 'buy' in text.lower():
        return 'Buy Low'
    return 'Unknown'

def analyze_ocr(ocr_list):
    text = "\n".join(ocr_list)
    index_price = None

    # Find Index Price
    for line in ocr_list:
        if 'index' in line.lower() or 'mark' in line.lower():
            nums = parse_numbers(line)
            if nums:
                index_price = nums[0]
                break
    if not index_price:
        nums = parse_numbers(text)
        if nums:
            index_price = np.median(nums)

    # Extract all numeric values
    all_nums = parse_numbers(text)
    apr_values = [n for n in all_nums if 50 <= n <= 800]
    price_values = [n for n in all_nums if n > 100 and abs(n - index_price) < index_price * 0.1]

    return index_price, price_values, apr_values, text

# ========= UI ========= #
st.divider()
uploaded_file = st.file_uploader("📷 Upload Bybit Screenshot", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", width=400)

    with st.spinner("Analyzing image... Please wait 5–10 seconds"):
        ocr_text = extract_text(image)

    if ocr_text:
        st.success("✅ OCR completed successfully!")
        index, prices, aprs, raw_text = analyze_ocr(ocr_text)
        table_type = detect_type(raw_text)

        if index and prices and aprs:
            offers = []
            for p in prices:
                for a in aprs:
                    delta = ((p - index) / index) * 100
                    daily = a / 365
                    score = a * abs(delta)
                    offers.append({
                        "Target": p,
                        "APR %": a,
                        "Δ %": round(delta, 3),
                        "Daily %": round(daily, 3),
                        "Score": round(score, 2)
                    })
            df = pd.DataFrame(sorted(offers, key=lambda x: x['Score'], reverse=True))

            st.subheader("📊 Detected Offers")
            st.dataframe(df.head(10), use_container_width=True)

            best = df.iloc[0]
            st.markdown(f"""
### 🏆 Recommended Best Offer:
- **Type:** {table_type}
- **Index:** ${index:.2f}  
- **Target:** ${best['Target']:.2f}  
- **Δ:** {best['Δ %']}%  
- **APR:** {best['APR %']}%  
- **Daily Yield:** {best['Daily %']}%  
- **Score:** {best['Score']}

💡 *This is the strongest opportunity based on APR × Δ%.*
""")

            st.download_button(
                label="⬇️ Download All Offers (CSV)",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"DualAsset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.error("⚠️ Failed to extract enough data from the image. Try another screenshot.")
    else:
        st.error("❌ OCR could not detect text in this image.")
else:
    st.info("📁 Upload a Bybit screenshot to begin.")
