import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import re
from datetime import datetime

st.set_page_config(page_title="DualAsset Analyzer Pro", page_icon="📊", layout="wide")
st.title("📊 DualAsset Analyzer Pro")
st.subheader("Auto-detect best Bybit Dual Asset offers (Buy Low / Sell High)")

# ===== Functions =====
def extract_text(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.splitlines()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return None

def parse_numbers(text):
    nums = re.findall(r'[\d,]+\.?\d*', text)
    return [float(n.replace(',', '')) for n in nums if n.replace(',', '').replace('.', '').isdigit()]

def detect_type(text):
    text_lower = text.lower()
    if "sell" in text_lower:
        return "Sell High"
    elif "buy" in text_lower:
        return "Buy Low"
    else:
        return "Unknown"

def analyze_ocr(ocr_list):
    text = "\n".join(ocr_list)
    all_nums = parse_numbers(text)

    # Index price (נמצא לפי המספר הגבוה הראשון)
    index_price = None
    if all_nums:
        index_price = max(all_nums[:10]) if len(all_nums) > 5 else np.median(all_nums)

    prices = [n for n in all_nums if n > 100 and abs(n - index_price) < index_price * 0.15]
    apr_values = [n for n in all_nums if 50 <= n <= 800]

    return index_price, prices, apr_values, text

# ===== UI =====
uploaded_file = st.file_uploader("📷 Upload Bybit Screenshot", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", width=400)

    with st.spinner("Analyzing image..."):
        ocr_results = extract_text(image)

    if ocr_results:
        st.success("✅ OCR completed successfully!")

        index, prices, aprs, raw_text = analyze_ocr(ocr_results)
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
            st.warning("⚠️ Could not extract enough data from image.")
    else:
        st.error("❌ No text detected in this image.")
else:
    st.info("📁 Upload a Bybit screenshot to begin.")
