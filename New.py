import streamlit as st
import easyocr
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from datetime import datetime

st.set_page_config(
    page_title="DualAsset Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== סטיילינג =====
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stMetric {
        background: linear-gradient(135deg, #161b22, #0d1117);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #58a6ff;
    }
    .good { color: #3fb950; font-weight: bold; }
    .bad { color: #f85149; font-weight: bold; }
    .neutral { color: #d29922; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ===== כותרת והוראות =====
st.title("📊 DualAsset Analyzer Pro")
st.subheader("ניתוח אוטומטי של הצעות Dual Asset מ־Bybit")

with st.expander("ℹ️ איך להשתמש?", expanded=False):
    st.info("""
    1. **צלמו צילום מסך** של טבלת Dual Asset ב־Bybit (Buy Low או Sell High)
    2. **העלו את התמונה** בחלון למטה
    3. **מתן לי 5-10 שניות** לעיבוד ו־OCR
    4. **קבלו מיד** את הניתוח עם המלצות השקעה
    5. **הורידו את ה־CSV** לשימוש עתידי
    """)

# ===== יצירת OCR Reader =====
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ===== פונקציות עיבוד =====

def extract_numbers(text):
    """חילוץ מספרים עם עד 2 ספרות עשרוניות"""
    pattern = r'\d+\.?\d{0,2}'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]

def process_image(image):
    """קריאת טקסט מהתמונה באמצעות EasyOCR"""
    try:
        # המרה לnumpy array
        img_array = np.array(image)
        
        # הרצת OCR
        results = reader.readtext(img_array, detail=0)
        ocr_text = '\n'.join(results)
        
        return ocr_text, results
    except Exception as e:
        st.error(f"❌ שגיאה ב־OCR: {e}")
        return None, None

def parse_dual_asset_table(ocr_text):
    """ניתוח טקסט וחילוץ נתונים של הצעות Dual Asset"""
    
    lines = ocr_text.split('\n')
    offers = []
    
    index_price = None
    current_offer = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # זיהוי מחיר ה־Index
        if 'index' in line.lower() or 'mark price' in line.lower():
            nums = extract_numbers(line)
            if nums:
                index_price = nums[0]
                continue
        
        # זיהוי שורות עם Target, APR וכו'
        numbers = extract_numbers(line)
        
        if len(numbers) >= 2:
            # אם יש לנו מינימום 2 מספרים, זו כנראה שורת הצעה
            if len(numbers) >= 3:
                target = numbers[0]
                apr = numbers[1]
                probability = numbers[2] if len(numbers) > 2 else 0
                
                offers.append({
                    'Target Price': target,
                    'APR (%)': apr,
                    'Probability (%)': probability,
                    'Raw Line': line
                })
            else:
                target = numbers[0]
                apr = numbers[1]
                
                offers.append({
                    'Target Price': target,
                    'APR (%)': apr,
                    'Probability (%)': 0,
                    'Raw Line': line
                })
    
    return offers, index_price

def calculate_delta(target, index):
    """חישוב הפרש באחוזים בין Target ל־Index"""
    if index == 0:
        return 0
    return ((target - index) / index) * 100

def calculate_daily_profit(apr):
    """חישוב תשואה יומית משוערת"""
    return apr / 365

def classify_offer(delta, apr, type_offer):
    """סיווג ההצעה (Buy/Sell/Hold)"""
    
    if abs(delta) < 0.3:
        return "🟡 Hold", "delta_too_small"
    
    if apr < 150:
        return "🟡 Weak", "apr_too_low"
    
    if abs(delta) > 5:
        return "🟡 Target Far", "delta_too_large"
    
    if type_offer == "buy_low" and delta <= -0.3 and apr > 150:
        return "🟢 Buy Low", "good"
    elif type_offer == "sell_high" and delta >= 0.3 and apr > 150:
        return "🟢 Sell High", "good"
    
    return "🟡 Neutral", "neutral"

def rank_offers(df_offers, type_offer):
    """דירוג הצעות לפי איכות"""
    df = df_offers.copy()
    df['Score'] = df['APR (%)'] * abs(df['Delta (%)'])
    df = df.sort_values('Score', ascending=False)
    return df

# ===== ממשק ראשי =====

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📸 העלו צילום מסך של Dual Asset מ־Bybit",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

with col2:
    offer_type = st.selectbox(
        "🏷️ סוג הטבלה:",
        ["Buy Low", "Sell High"]
    )

# ===== עיבוד התמונה =====

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # הצגת התמונה
    st.image(image, caption="צילום המסך שהועלה", use_column_width=True)
    
    with st.spinner("⏳ מעבדים את התמונה..."):
        ocr_text, raw_results = process_image(image)
    
    if ocr_text:
        st.success("✅ קריאת OCR הצליחה!")
        
        # ניתוח הטקסט
        offers, index_price = parse_dual_asset_table(ocr_text)
        
        if offers and index_price:
            # יצירת DataFrame
            df = pd.DataFrame(offers)
            df['Index Price'] = index_price
            df['Delta (%)'] = df['Target Price'].apply(
                lambda x: calculate_delta(x, index_price)
            )
            df['Daily Profit (%)'] = df['APR (%)'].apply(calculate_daily_profit)
            
            # סיווג
            offer_type_key = "buy_low" if "buy" in offer_type.lower() else "sell_high"
            df['Decision'] = df['Delta (%)'].apply(
                lambda x: classify_offer(x, df['APR (%)'].mean(), offer_type_key)
            ).str.split(' ').str[0]
            
            # ===== תצוגה =====
            st.markdown("---")
            st.subheader("📋 טבלת ניתוח מלאה")
            
            # טבלה מעוצבת
            display_df = df[[
                'Target Price',
                'APR (%)',
                'Delta (%)',
                'Daily Profit (%)',
                'Decision'
            ]].copy()
            
            display_df['APR (%)'] = display_df['APR (%)'].round(2)
            display_df['Delta (%)'] = display_df['Delta (%)'].round(3)
            display_df['Daily Profit (%)'] = display_df['Daily Profit (%)'].round(3)
            display_df['Target Price'] = display_df['Target Price'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # ===== המלצות מובילות =====
            st.markdown("---")
            st.subheader("🎯 המלצות מובילות")
            
            good_offers = df[df['Decision'].str.contains('🟢', na=False)]
            
            if len(good_offers) > 0:
                col1, col2 = st.columns(2)
                
                top_offer = good_offers.iloc[0]
                
                with col1:
                    st.metric(
                        "💰 Index Price",
                        f"${index_price:.2f}",
                        delta=f"{top_offer['Delta (%)']:.3f}%"
                    )
                
                with col2:
                    st.metric(
                        "🎯 Target Price",
                        f"${top_offer['Target Price']:.2f}",
                        delta=f"{top_offer['APR (%)']:.2f}% APR"
                    )
                
                st.success(f"""
                ### ✅ הצעה מומלצת:
                - **מחיר יעד:** ${top_offer['Target Price']:.2f}
                - **ריבית שנתית:** {top_offer['APR (%)']:.2f}%
                - **הפרש:** {top_offer['Delta (%)']:.3f}%
                - **תשואה יומית משוערת:** {top_offer['Daily Profit (%)']:.3f}%
                """)
            else:
                st.warning("""
                ⚠️ לא נמצאו הצעות חזקות כרגע.
                
                הקריטריונים:
                - Δ בין -0.3% ל +0.3% (או גדול מ־5%)
                - APR מעל 150%
                """)
            
            # ===== הורדת CSV =====
            st.markdown("---")
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 הורד את הנתונים כ־CSV",
                data=csv,
                file_name=f"dual_asset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # ===== OCR Debug (לבדיקה) =====
            with st.expander("🔍 Debug - טקסט גולמי מ־OCR", expanded=False):
                st.text(ocr_text)
        
        else:
            st.error("❌ לא הצלחנו להוציא נתונים מהתמונה. בדקו שהתמונה ברורה.")
    
    else:
        st.error("❌ כשלון בקריאת OCR.")    try:
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
