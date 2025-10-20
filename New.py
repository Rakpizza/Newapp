import streamlit as st
import easyocr
import pandas as pd
import numpy as np
from PIL import Image
import re
from datetime import datetime

st.set_page_config(
    page_title="DualAsset Analyzer Pro",
    page_icon="📊",
    layout="wide",
)

st.title("📊 DualAsset Analyzer Pro")
st.subheader("Auto-detect best Bybit Dual Asset offers")

# ===== Load OCR =====
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

try:
    reader = load_ocr()
    st.write("OCR Ready")
except Exception as e:
    st.error(f"OCR Error: {e}")
    reader = None

# ===== Functions =====

def extract_ocr_text(image):
    if reader is None:
        st.error("OCR unavailable")
        return None
    
    try:
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0)
        return results
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return None

def parse_numbers(text):
    nums = re.findall(r'[\d,]+\.?\d*', text)
    result = []
    for n in nums:
        try:
            result.append(float(n.replace(',', '')))
        except:
            pass
    return result

def analyze_ocr(ocr_list):
    all_text = '\n'.join(ocr_list)
    
    st.write("**Text detected:**")
    st.text(all_text[:500])
    
    # Find Index
    index_price = None
    for line in ocr_list:
        if 'index' in line.lower() or 'mark' in line.lower():
            nums = parse_numbers(line)
            if nums:
                index_price = nums[0]
                break
    
    if not index_price:
        nums = parse_numbers(ocr_list[0])
        if nums and nums[0] > 100:
            index_price = nums[0]
    
    st.write(f"Index found: ${index_price}")
    
    # Find all numbers
    all_numbers = parse_numbers(all_text)
    st.write(f"Numbers found: {len(all_numbers)}")
    st.write(f"Values: {all_numbers[:20]}")
    
    # Separate prices and APR
    prices = []
    apr_values = []
    
    for num in all_numbers:
        if num > 1000 or (num > 100 and index_price and abs(num - index_price) < 10000):
            if num not in prices:
                prices.append(num)
        elif num >= 50 and num <= 500:
            if num not in apr_values:
                apr_values.append(num)
    
    return index_price, prices, apr_values

def detect_type(text):
    if 'sell' in text.lower():
        return 'Sell High'
    elif 'buy' in text.lower():
        return 'Buy Low'
    return 'Unknown'

# ===== UI =====

st.markdown("---")

uploaded_file = st.file_uploader("Upload Bybit screenshot", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=400)
    
    st.write("Processing...")
    
    ocr_results = extract_ocr_text(image)
    
    if ocr_results:
        st.success("OCR Success!")
        
        index_price, prices, apr_values = analyze_ocr(ocr_results)
        all_text = '\n'.join(ocr_results)
        table_type = detect_type(all_text)
        
        st.info(f"Type: {table_type}")
        
        if index_price and prices and apr_values:
            st.success("Data Complete!")
            
            # Create offers
            offers = []
            for p in prices[:10]:
                for a in apr_values[:5]:
                    delta = ((p - index_price) / index_price) * 100
                    daily = a / 365
                    score = a * abs(delta)
                    
                    offers.append({
                        'Target': p,
                        'APR': a,
                        'Delta': delta,
                        'Daily': daily,
                        'Score': score
                    })
            
            offers = sorted(offers, key=lambda x: x['Score'], reverse=True)
            
            st.subheader("Top Offers")
            df = pd.DataFrame(offers[:5])
            st.dataframe(df, use_container_width=True)
            
            # Best recommendation
            best = offers[0]
            
            st.markdown(f"""
## Recommended Best Offer:

- Type: {table_type}
- Index: ${index_price:.2f}
- Target: ${best['Target']:.2f}
- Delta: {best['Delta']:.3f}%
- APR: {best['APR']:.2f}%
- Daily Profit: {best['Daily']:.3f}%
- Score: {best['Score']:.2f}

**This is the best offer to invest!**
            """)
            
            # Download
            csv = pd.DataFrame(offers).to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"offers_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        
        else:
            st.error(f"""
Data extraction failed:
- Index: {index_price}
- Prices: {len(prices)}
- APR values: {len(apr_values)}
            """)
    
    else:
        st.error("OCR failed - try another image")

else:
    st.info("Select image to start")
