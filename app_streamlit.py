import streamlit as st
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import plotly.express as px

from scripts.db import (
    get_user_by_email, insert_user,
    get_items_for_user, insert_item
)

# --- Page config ---
st.set_page_config(page_title="SmartPantry+", layout="centered", page_icon="🥫")

if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_email" not in st.session_state:
    st.session_state.user_email = None

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
</style>
""", unsafe_allow_html=True)

# --- Login Page ---
def login_page():
    st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
    st.title("🔐 Login to SmartPantry+")
    st.write("Welcome back! Please login to continue.")

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
    st.write("Join SmartPantry+ today!")

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
                st.error("❌ Passwords don’t match.")
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
    st.sidebar.title("Navigation")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user_email = None
        st.session_state.page = "login"
        st.rerun()

    st.title(f"🥫 SmartPantry+ Dashboard")
    st.caption(f"👤 Logged in as: `{st.session_state.user_email}`")

    # --- Inventory Section ---
    st.subheader("📦 Your Inventory")
    items = get_items_for_user(st.session_state.user_email)

    if items:
        df = pd.DataFrame(items)

        # Show as table
        st.dataframe(
            df[["item", "category", "quantity", "expiry"]],
            use_container_width=True,
            hide_index=True
        )

        # --- Charts ---
        st.markdown("---")
        st.subheader("📊 Inventory Insights")

        # 🎨 Custom colors
        color_map = {
            "Dairy": "#FFB347",
            "Vegetables": "#77DD77",
            "Fruits": "#FF6961",
            "Beverages": "#779ECB",
            "Bakery": "#FDFD96",
            "Meat": "#C23B22",
            "Frozen": "#AEC6CF",
            "Snacks": "#DEA5A4",
            "Condiments/Spices": "#B39EB5",
            "Other": "#D3D3D3"
        }

        # Pie chart: Categories
        if "category" in df.columns:
            pie_fig = px.pie(
                df, names='category', title='📈 Items by Category',
                color='category', color_discrete_map=color_map
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        # Bar chart: Expiry timeline
        if "expiry" in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
            df_expiry = df.groupby(['expiry', 'category']).size().reset_index(name='count')
            bar_fig = px.bar(
                df_expiry, x='expiry', y='count', color='category',
                title='⏳ Items Expiry Timeline', color_discrete_map=color_map
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.info("📭 No items yet. Add some below!")

    st.markdown("---")

    # --- Add Item Form ---
    with st.form("add_item_form"):
        st.subheader("➕ Add New Item")
        cols = st.columns(2)
        name = cols[0].text_input("Item Name")
        category = cols[1].text_input("Category")
        cols2 = st.columns(2)
        added_on = cols2[0].date_input("Added On", value=datetime.today())
        expiry = cols2[1].date_input("Expiry")
        quantity = st.number_input("Quantity", min_value=1, value=1)
        submit = st.form_submit_button("✅ Add Item")

        if submit:
            insert_item({
                "item": name,
                "category": category,
                "added_on": str(added_on),
                "expiry": str(expiry),
                "quantity": quantity,
                "user_email": st.session_state.user_email
            })
            st.success(f"🎉 Item `{name}` added successfully!")
            st.rerun()

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

