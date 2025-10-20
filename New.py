import streamlit as st
import easyocr
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from datetime import datetime
import cv2

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
    .good { 
        background: linear-gradient(135deg, #238636, #1a6e2a) !important;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #3fb950;
        color: #fff;
        font-weight: bold;
    }
    .bad { 
        background: linear-gradient(135deg, #da3633, #9e1c23) !important;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #f85149;
        color: #fff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 DualAsset Analyzer Pro")
st.subheader("🎯 זיהוי אוטומטי של הצעות Bybit הכי רווחיות")

# ===== יצירת OCR Reader =====
@st.cache_resource
def load_ocr():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.error(f"❌ שגיאה בטעינת OCR: {e}")
        return None

reader = load_ocr()

# ===== פונקציות עיבוד =====

def preprocess_image(image):
    """שיפור התמונה לקריאה טובה יותר"""
    img_array = np.array(image)
    
    # המרה לגווני אפור
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # הגברת קונטרסט
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return Image.fromarray(enhanced)

def extract_ocr_text(image):
    """קריאת טקסט מהתמונה"""
    if reader is None:
        st.error("❌ OCR לא טוען כראוי")
        return None
    
    try:
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=1)
        
        return results
    except Exception as e:
        st.error(f"❌ שגיאה ב־OCR: {str(e)}")
        return None

def parse_ocr_results(results):
    """ניתוח תוצאות OCR וחילוץ מספרים"""
    
    prices = []
    apr_values = []
    index_price = None
    
    for (bbox, text, confidence) in results:
        text = text.strip()
        
        # חיפוש מחיר Index
        if 'index' in text.lower() or 'mark' in text.lower():
            nums = re.findall(r'\d+[\.,]\d+', text)
            if nums:
                index_price = float(nums[0].replace(',', '.'))
        
        # חיפוש מספרים (מחיר + APR)
        # דוגמה: "4,075  407.96%"
        nums = re.findall(r'[\d,]+\.?\d*', text)
        
        if len(nums) >= 1:
            try:
                # ניקוי ממפרידי אלפים
                main_num = float(nums[0].replace(',', ''))
                
                # זיהוי אם זה מחיר או APR
                if main_num > 100:  # אם גדול מ־100, זה כנראה מחיר
                    if main_num not in prices:
                        prices.append(main_num)
                elif main_num > 1:  # בין 1 ל־100 = אחוז
                    if main_num not in apr_values:
                        apr_values.append(main_num)
            except:
                pass
    
    return prices, apr_values, index_price

def detect_table_type(image):
    """זיהוי אוטומטי אם זה Buy Low או Sell High"""
    
    try:
        results = reader.readtext(np.array(image), detail=0)
        text = '\n'.join(results).lower()
        
        if 'sell' in text or 'high' in text:
            return 'Sell High'
        elif 'buy' in text or 'low' in text:
            return 'Buy Low'
        else:
            # אם אין ברור, בדוק את הנתונים
            return 'Unknown'
    except:
        return 'Unknown'

def create_offers(prices, apr_values, index_price):
    """יצירת רשימת הצעות מסודרת"""
    
    offers = []
    
    # שימוש בכל הצירוף הקרוב
    for price in prices:
        for apr in apr_values:
            delta = ((price - index_price) / index_price) * 100 if index_price else 0
            
            daily_profit = apr / 365
            
            offers.append({
                'Target Price': price,
                'APR (%)': apr,
                'Delta (%)': delta,
                'Daily Profit (%)': daily_profit,
                'Score': apr * abs(delta) if abs(delta) > 0.3 else 0
            })
    
    # מיון לפי Score
    offers = sorted(offers, key=lambda x: x['Score'], reverse=True)
    
    return offers

def rank_by_profitability(offers, table_type):
    """דירוג הצעות לפי רווחיות"""
    
    ranked = []
    
    for offer in offers:
        delta = offer['Delta (%)']
        apr = offer['APR (%)']
        
        # קריטריונים
        is_good = False
        reason = ""
        
        if table_type == 'Buy Low':
            # Buy Low: מחיר יעד נמוך מ־Index
            if delta <= -0.3 and apr > 150:
                is_good = True
                reason = "✅ כדאי מאוד - קנייה זולה עם ריבית גבוהה"
            elif delta <= -0.3:
                is_good = False
                reason = "⚠️ מחיר טוב אבל ריבית נמוכה"
            elif abs(delta) < 0.3:
                is_good = False
                reason = "⚠️ הפרש קטן מדי"
            else:
                is_good = False
                reason = "❌ מחיר גבוה מ־Index"
        
        elif table_type == 'Sell High':
            # Sell High: מחיר יעד גבוה מ־Index
            if delta >= 0.3 and apr > 150:
                is_good = True
                reason = "✅ כדאי מאוד - מכירה יקרה עם ריבית גבוהה"
            elif delta >= 0.3:
                is_good = False
                reason = "⚠️ מחיר טוב אבל ריבית נמוכה"
            elif abs(delta) < 0.3:
                is_good = False
                reason = "⚠️ הפרש קטן מדי"
            else:
                is_good = False
                reason = "❌ מחיר נמוך מ־Index"
        
        ranked.append({
            **offer,
            'Is Good': is_good,
            'Reason': reason
        })
    
    return ranked

# ===== ממשק ראשי =====

uploaded_file = st.file_uploader(
    "📸 העלו צילום מסך של Dual Asset מ־Bybit",
    type=['jpg', 'jpeg', 'png', 'bmp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # הצגת התמונה
    st.image(image, caption="📷 התמונה שהועלתה", use_column_width=True)
    
    with st.spinner("⏳ מעבדים את התמונה..."):
        # שיפור התמונה
        enhanced_image = preprocess_image(image)
        
        # קריאת OCR
        ocr_results = extract_ocr_text(enhanced_image)
    
    if ocr_results:
        st.success("✅ קריאת OCR הצליחה!")
        
        # זיהוי סוג הטבלה
        table_type = detect_table_type(image)
        
        # ניתוח נתונים
        prices, apr_values, index_price = parse_ocr_results(ocr_results)
        
        if prices and apr_values and index_price:
            st.info(f"🔍 **זוהה:** {table_type} | Index: ${index_price:.2f}")
            
            # יצירת הצעות
            offers = create_offers(prices, apr_values, index_price)
            
            if offers:
                # דירוג
                ranked_offers = rank_by_profitability(offers, table_type)
                
                # יצוג בטבלה
                st.subheader("📊 כל ההצעות שזוהו:")
                
                df_display = pd.DataFrame([{
                    'Target': f"${o['Target Price']:.2f}",
                    'APR': f"{o['APR (%)']:.2f}%",
                    'Delta': f"{o['Delta (%)']:.3f}%",
                    'Daily': f"{o['Daily Profit (%)']:.3f}%",
                    'הערכה': o['Reason']
                } for o in ranked_offers])
                
                st.dataframe(df_display, use_container_width=True)
                
                # ===== המלצה המובילה =====
                st.markdown("---")
                
                best_offers = [o for o in ranked_offers if o['Is Good']]
                
                if best_offers:
                    best = best_offers[0]
                    
                    st.markdown(f"""
<div class="good">
<h3>🎯 ההשקעה המומלצת ביותר:</h3>
<p><strong>סוג:</strong> {table_type}</p>
<p><strong>מחיר Index:</strong> ${index_price:.2f}</p>
<p><strong>מחיר יעד:</strong> ${best['Target Price']:.2f}</p>
<p><strong>הפרש:</strong> {best['Delta (%)']:.3f}%</p>
<p><strong>ריבית שנתית:</strong> {best['APR (%)']:.2f}%</p>
<p><strong>תשואה יומית משוערת:</strong> {best['Daily Profit (%)']:.3f}%</p>
<p><strong>⚡ הערכה:</strong> {best['Reason']}</p>
<hr>
<p style="font-size: 16px; font-weight: bold;">זו ההצעה עם הסיכוי הגבוה ביותר לרווח!</p>
</div>
""", unsafe_allow_html=True)
                    
                    # CSV להורדה
                    csv = pd.DataFrame(ranked_offers).to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 הורד את כל הנתונים (CSV)",
                        data=csv,
                        file_name=f"dual_asset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.markdown("""
<div class="bad">
<h3>❌ אין הצעות טובות כרגע</h3>
<p>הקריטריונים:</p>
<ul>
<li>Δ חייב להיות בכיוון הנכון</li>
<li>APR חייב להיות מעל 150%</li>
<li>הפרש חייב להיות מעל 0.3%</li>
</ul>
</div>
""", unsafe_allow_html=True)
            
            else:
                st.error("❌ לא הצלחנו ליצור הצעות מהנתונים")
        
        else:
            st.error(f"❌ חסרים נתונים - Index: {index_price}, Prices: {len(prices)}, APR: {len(apr_values)}")
            st.info("💡 ודקו שהתמונה כוללת: מחיר Index, רשימת מחירים יעד, וערכי APR")
    
    else:
        st.error("❌ כשלון בקריאת OCR - נסו תמונה אחרת")
