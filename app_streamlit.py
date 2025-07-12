# to run  streamlit run app_streamlit.py

import streamlit as st
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re

from scripts.db import (
    get_user_by_email, insert_user,
    get_items_for_user, insert_item
)

# --- Page config ---
st.set_page_config(page_title="SmartPantry+ ML", layout="wide", page_icon="🤖")

if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# --- ML Models and Data ---
@st.cache_data
def load_food_categories():
    """Predefined food categories for ML classification"""
    return {
        'Dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'sour cream', 'cottage cheese'],
        'Vegetables': ['carrot', 'potato', 'onion', 'tomato', 'lettuce', 'spinach', 'broccoli', 'celery'],
        'Fruits': ['apple', 'banana', 'orange', 'grapes', 'strawberry', 'blueberry', 'lemon', 'lime'],
        'Beverages': ['water', 'juice', 'soda', 'beer', 'wine', 'coffee', 'tea', 'milk'],
        'Bakery': ['bread', 'bagel', 'muffin', 'croissant', 'cake', 'cookies', 'pizza'],
        'Meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'ham', 'bacon', 'sausage'],
        'Frozen': ['ice cream', 'frozen vegetables', 'frozen fruit', 'frozen pizza', 'frozen dinner'],
        'Snacks': ['chips', 'crackers', 'nuts', 'popcorn', 'candy', 'chocolate', 'granola'],
        'Condiments/Spices': ['salt', 'pepper', 'ketchup', 'mustard', 'mayo', 'hot sauce', 'vinegar'],
        'Other': ['rice', 'pasta', 'cereal', 'oil', 'flour', 'sugar']
    }

@st.cache_data
def load_shelf_life_data():
    """Average shelf life in days for different categories"""
    return {
        'Dairy': 7,
        'Vegetables': 10,
        'Fruits': 7,
        'Beverages': 365,
        'Bakery': 5,
        'Meat': 3,
        'Frozen': 90,
        'Snacks': 180,
        'Condiments/Spices': 365,
        'Other': 730
    }

class SmartPantryML:
    def __init__(self):
        self.food_categories = load_food_categories()
        self.shelf_life_data = load_shelf_life_data()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def predict_category(self, item_name):
        """Predict category for a food item using keyword matching and ML"""
        item_name = item_name.lower().strip()
        
        # Direct keyword matching first
        for category, keywords in self.food_categories.items():
            for keyword in keywords:
                if keyword in item_name or item_name in keyword:
                    return category
        
        # If no direct match, use similarity scoring
        best_score = 0
        best_category = 'Other'
        
        for category, keywords in self.food_categories.items():
            category_text = ' '.join(keywords)
            # Simple word overlap scoring
            words_in_item = set(item_name.split())
            words_in_category = set(category_text.split())
            overlap = len(words_in_item.intersection(words_in_category))
            if overlap > best_score:
                best_score = overlap
                best_category = category
        
        return best_category
    
    def predict_expiry_risk(self, items_df):
        """Predict which items are at high risk of expiring soon"""
        if items_df.empty:
            return pd.DataFrame()
        
        # Calculate days until expiry
        items_df['expiry'] = pd.to_datetime(items_df['expiry'])
        items_df['days_until_expiry'] = (items_df['expiry'] - datetime.now()).dt.days
        
        # Risk scoring
        def calculate_risk(row):
            days = row['days_until_expiry']
            if days <= 1:
                return 'Critical'
            elif days <= 3:
                return 'High'
            elif days <= 7:
                return 'Medium'
            else:
                return 'Low'
        
        items_df['expiry_risk'] = items_df.apply(calculate_risk, axis=1)
        return items_df
    
    def generate_smart_recommendations(self, items_df):
        """Generate smart recommendations based on inventory analysis"""
        recommendations = []
        
        if items_df.empty:
            return ["🛒 Your pantry is empty! Time to go shopping."]
        
        # Expiry-based recommendations
        critical_items = items_df[items_df['expiry_risk'] == 'Critical']
        if not critical_items.empty:
            recommendations.append(f"🚨 {len(critical_items)} items expire today/tomorrow! Use them immediately.")
        
        high_risk_items = items_df[items_df['expiry_risk'] == 'High']
        if not high_risk_items.empty:
            recommendations.append(f"⚠️ {len(high_risk_items)} items expire in 2-3 days. Plan meals around them.")
        
        # Category balance analysis
        category_counts = items_df['category'].value_counts()
        if 'Vegetables' not in category_counts or category_counts.get('Vegetables', 0) < 3:
            recommendations.append("🥬 Consider adding more vegetables to your pantry for a balanced diet.")
        
        if 'Fruits' not in category_counts or category_counts.get('Fruits', 0) < 2:
            recommendations.append("🍎 Your fruit supply is low. Fresh fruits are great for health!")
        
        # Quantity-based recommendations
        low_quantity_items = items_df[items_df['quantity'] <= 1]
        if not low_quantity_items.empty:
            recommendations.append(f"📦 {len(low_quantity_items)} items are running low. Consider restocking.")
        
        # Seasonal recommendations
        month = datetime.now().month
        if month in [12, 1, 2]:  # Winter
            recommendations.append("❄️ Winter tip: Stock up on citrus fruits and warming spices!")
        elif month in [6, 7, 8]:  # Summer
            recommendations.append("☀️ Summer tip: Keep plenty of fresh fruits and cold beverages!")
        
        return recommendations if recommendations else ["✅ Your pantry looks well-balanced!"]
    
    def analyze_consumption_patterns(self, items_df):
        """Analyze consumption patterns and trends"""
        if items_df.empty:
            return {}
        
        analysis = {}
        
        # Category distribution
        analysis['category_distribution'] = items_df['category'].value_counts().to_dict()
        
        # Average days until expiry by category
        avg_expiry = items_df.groupby('category')['days_until_expiry'].mean().to_dict()
        analysis['avg_days_until_expiry'] = {k: round(v, 1) for k, v in avg_expiry.items()}
        
        # Total quantity by category
        analysis['quantity_by_category'] = items_df.groupby('category')['quantity'].sum().to_dict()
        
        # Risk distribution
        analysis['risk_distribution'] = items_df['expiry_risk'].value_counts().to_dict()
        
        return analysis

# Initialize ML engine
@st.cache_resource
def get_ml_engine():
    return SmartPantryML()

ml_engine = get_ml_engine()

# --- Styles ---
st.markdown("""
<style>
.stButton>button {
    border-radius: 8px;
    font-weight: bold;
}
.stTextInput>div>div>input {
    border-radius: 5px;
}
div[data-testid="column"] {
    padding: 0 10px;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}
.risk-critical { background: #ff4444; padding: 5px; border-radius: 3px; color: white; }
.risk-high { background: #ff8800; padding: 5px; border-radius: 3px; color: white; }
.risk-medium { background: #ffcc00; padding: 5px; border-radius: 3px; color: black; }
.risk-low { background: #44ff44; padding: 5px; border-radius: 3px; color: black; }
</style>
""", unsafe_allow_html=True)

# --- Login Page ---
def login_page():
    st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
    st.title("🤖 SmartPantry+ ML")
    st.write("AI-Powered Pantry Management")

    with st.form("login_form", clear_on_submit=True):
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.form_submit_button("Login")
        with col2:
            signup_btn = st.form_submit_button("Create Account")

        if login_btn:
            user = get_user_by_email(email)
            if user and check_password_hash(user["password"], password):
                st.session_state.user_email = email
                st.session_state.page = "dashboard"
                st.success(f"✅ Welcome, {email}")
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")
        elif signup_btn:
            st.session_state.page = "signup"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Signup Page ---
def signup_page():
    st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
    st.title("📝 Create Account")
    st.write("Join SmartPantry+ ML today!")

    with st.form("signup_form", clear_on_submit=True):
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type="password")
        confirm = st.text_input("🔒 Confirm Password", type="password")
        col1, col2 = st.columns(2)
        with col1:
            signup_btn = st.form_submit_button("Sign Up")
        with col2:
            back_btn = st.form_submit_button("Back to Login")

        if signup_btn:
            if password != confirm:
                st.error("❌ Passwords don't match.")
            elif get_user_by_email(email):
                st.error("❌ Email already exists.")
            else:
                insert_user({"email": email, "password": generate_password_hash(password)})
                st.success("✅ Account created. Please log in.")
                st.session_state.page = "login"
                st.rerun()
        elif back_btn:
            st.session_state.page = "login"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Dashboard Page ---
def dashboard_page():
    st.sidebar.title("🤖 AI Navigation")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user_email = None
        st.session_state.page = "login"
        st.rerun()

    # Sidebar AI insights
    st.sidebar.markdown("### 🎯 Quick AI Insights")
    items = get_items_for_user(st.session_state.user_email)
    
    if items:
        df = pd.DataFrame(items)
        df_with_risk = ml_engine.predict_expiry_risk(df)
        
        critical_count = len(df_with_risk[df_with_risk['expiry_risk'] == 'Critical'])
        high_risk_count = len(df_with_risk[df_with_risk['expiry_risk'] == 'High'])
        
        st.sidebar.metric("🚨 Critical Items", critical_count)
        st.sidebar.metric("⚠️ High Risk Items", high_risk_count)
        st.sidebar.metric("📦 Total Items", len(df))

    st.title(f"🤖 SmartPantry+ ML Dashboard")
    st.caption(f"👤 Logged in as: `{st.session_state.user_email}`")

    # --- AI Recommendations Section ---
    if items:
        df = pd.DataFrame(items)
        df_with_risk = ml_engine.predict_expiry_risk(df)
        recommendations = ml_engine.generate_smart_recommendations(df_with_risk)
        
        st.subheader("🧠 AI Recommendations")
        for i, rec in enumerate(recommendations):
            st.info(rec)

    # --- Main Dashboard Layout ---
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Inventory", "📊 AI Analytics", "🎯 Smart Add", "🔍 ML Insights"])
    
    with tab1:
        # --- Inventory Section ---
        st.subheader("📦 Your Smart Inventory")
        
        if items:
            df = pd.DataFrame(items)
            df_with_risk = ml_engine.predict_expiry_risk(df)

            # Risk-based styling
            def style_risk(val):
                if val == 'Critical':
                    return 'background-color: #ff4444; color: white'
                elif val == 'High':
                    return 'background-color: #ff8800; color: white'
                elif val == 'Medium':
                    return 'background-color: #ffcc00; color: black'
                else:
                    return 'background-color: #44ff44; color: black'

            # Display with risk indicators
            display_df = df_with_risk[["item", "category", "quantity", "expiry", "days_until_expiry", "expiry_risk"]].copy()
            display_df.columns = ["Item", "Category", "Quantity", "Expiry Date", "Days Left", "Risk Level"]
            
            st.dataframe(
                display_df.style.applymap(style_risk, subset=['Risk Level']),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("📭 No items yet. Use the Smart Add tab to get started!")

    with tab2:
        # --- AI Analytics ---
        st.subheader("📊 AI-Powered Analytics")
        
        if items:
            df = pd.DataFrame(items)
            df_with_risk = ml_engine.predict_expiry_risk(df)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_items = len(df)
                st.metric("Total Items", total_items)
            
            with col2:
                critical_items = len(df_with_risk[df_with_risk['expiry_risk'] == 'Critical'])
                st.metric("Critical Items", critical_items, delta=f"-{critical_items}" if critical_items > 0 else None)
            
            with col3:
                categories = df['category'].nunique()
                st.metric("Categories", categories)
            
            with col4:
                avg_days = df_with_risk['days_until_expiry'].mean()
                st.metric("Avg Days Left", f"{avg_days:.1f}")

            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = df_with_risk['expiry_risk'].value_counts()
                fig_risk = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='🎯 Expiry Risk Distribution',
                    color_discrete_map={
                        'Critical': '#ff4444',
                        'High': '#ff8800',
                        'Medium': '#ffcc00',
                        'Low': '#44ff44'
                    }
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                # Category distribution
                cat_counts = df['category'].value_counts()
                fig_cat = px.bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    title='📈 Items by Category',
                    color=cat_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_cat.update_layout(showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)

            # Timeline view
            st.subheader("⏳ Smart Expiry Timeline")
            timeline_df = df_with_risk.sort_values('days_until_expiry')
            fig_timeline = px.scatter(
                timeline_df,
                x='days_until_expiry',
                y='item',
                color='expiry_risk',
                size='quantity',
                hover_data=['category', 'expiry'],
                title='Items by Days Until Expiry',
                color_discrete_map={
                    'Critical': '#ff4444',
                    'High': '#ff8800',
                    'Medium': '#ffcc00',
                    'Low': '#44ff44'
                }
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        else:
            st.info("📊 Add some items to see AI analytics!")

    with tab3:
        # --- Smart Add Item ---
        st.subheader("🎯 Smart Add Item (AI-Powered)")
        
        with st.form("smart_add_item_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Item Name", help="AI will auto-suggest category")
                if name:
                    predicted_category = ml_engine.predict_category(name)
                    st.info(f"🤖 AI suggests category: **{predicted_category}**")
                    category = st.selectbox(
                        "Category",
                        options=list(ml_engine.food_categories.keys()),
                        index=list(ml_engine.food_categories.keys()).index(predicted_category) if predicted_category in ml_engine.food_categories.keys() else 0
                    )
                else:
                    category = st.selectbox("Category", options=list(ml_engine.food_categories.keys()))
            
            with col2:
                added_on = st.date_input("Added On", value=datetime.today())
                
                # Smart expiry suggestion
                if category:
                    suggested_days = ml_engine.shelf_life_data.get(category, 7)
                    suggested_expiry = datetime.today() + timedelta(days=suggested_days)
                    st.info(f"🤖 AI suggests expiry: **{suggested_expiry.strftime('%Y-%m-%d')}** ({suggested_days} days)")
                
                expiry = st.date_input("Expiry Date", value=suggested_expiry if 'suggested_expiry' in locals() else datetime.today() + timedelta(days=7))
                quantity = st.number_input("Quantity", min_value=1, value=1)
            
            submit = st.form_submit_button("✅ Add Item with AI")

            if submit and name:
                insert_item({
                    "item": name,
                    "category": category,
                    "added_on": str(added_on),
                    "expiry": str(expiry),
                    "quantity": quantity,
                    "user_email": st.session_state.user_email
                })
                st.success(f"🎉 Item `{name}` added successfully with AI assistance!")
                st.rerun()

    with tab4:
        # --- ML Insights ---
        st.subheader("🔍 Machine Learning Insights")
        
        if items:
            df = pd.DataFrame(items)
            df_with_risk = ml_engine.predict_expiry_risk(df)
            analysis = ml_engine.analyze_consumption_patterns(df_with_risk)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Category Analysis**")
                for category, count in analysis['category_distribution'].items():
                    percentage = (count / len(df)) * 100
                    st.write(f"• {category}: {count} items ({percentage:.1f}%)")
                
                st.write("**⏱️ Average Days Until Expiry**")
                for category, days in analysis['avg_days_until_expiry'].items():
                    st.write(f"• {category}: {days} days")
            
            with col2:
                st.write("**🎯 Risk Assessment**")
                for risk, count in analysis['risk_distribution'].items():
                    percentage = (count / len(df)) * 100
                    risk_emoji = {'Critical': '🚨', 'High': '⚠️', 'Medium': '🔶', 'Low': '✅'}
                    st.write(f"• {risk_emoji.get(risk, '•')} {risk}: {count} items ({percentage:.1f}%)")
                
                st.write("**📦 Quantity Distribution**")
                for category, qty in analysis['quantity_by_category'].items():
                    st.write(f"• {category}: {qty} total items")

            # ML Model Performance Info
            st.subheader("🤖 AI Model Information")
            st.info("""
            **Smart Features Enabled:**
            - 🎯 **Auto-Categorization**: AI predicts food categories using keyword matching and similarity scoring
            - ⏳ **Expiry Risk Prediction**: ML algorithm assesses expiry risk based on days remaining
            - 🧠 **Smart Recommendations**: AI analyzes patterns to suggest actions
            - 📊 **Consumption Analysis**: ML tracks usage patterns and trends
            - 🤖 **Predictive Expiry**: AI suggests optimal expiry dates based on food type
            """)

        else:
            st.info("🔍 Add some items to see ML insights!")

# --- Router ---
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "dashboard":
    if st.session_state.user_email:
        dashboard_page()
    else:
        st.session_state.page = "login"
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("🤖 **SmartPantry+ ML** - AI-Powered Pantry Management | Built with Streamlit & Machine Learning")