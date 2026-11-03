"""
app.py
======
MessMind AI: Multi-Plate Adaptive Food Waste Intelligence & Smart Hostel Mess Management System
Developed for IIIT Dharwad (Frugal Innovation).

Architecture:
- Role-Based Portals: Mess Manager Portal vs Student Portal
- PBKDF2-HMAC-SHA256 Authentication & RBAC Isolation
- Multi-Plate OpenCV Vision Analyzer with Adaptive Preprocessing & Waste Fingerprinting
- Scikit-Learn RandomForest Demand Forecasting (+5% Safety Buffer)
- Verifiable SQLite Relational Database Engine (data/messmind.db - 3,858 Historical Shifts)
- Eco-Tech SaaS Aesthetics (Deep Forest Green, Emerald Green, Fresh Lime, Warm Off-White)
"""

import os
import sys
import socket
from datetime import datetime, date
import streamlit as st

# Local modules
import database as db
import portal_login
import portal_manager
import portal_student

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MessMind AI — Smart Hostel Dining & Food Waste Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. ECO-TECH DESIGN SYSTEM (Vanilla CSS)
# -----------------------------------------------------------------------------
ECO_TECH_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    :root {
        --forest-green: #12372A;
        --emerald-green: #16A36A;
        --fresh-lime: #C5F467;
        --warm-bg: #F7F8F2;
        --card-white: #FFFFFF;
        --charcoal: #18231F;
        --muted-gray: #8A9990;
        --soft-orange: #F4A261;
        --alert-red: #E76F51;
    }

    /* Completely hide Streamlit sidebar for full-width horizontal navigation */
    [data-testid="stSidebar"], 
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
        transform: translateX(-500px) !important;
    }

    /* Global Typography & Background */
    html, body, [class*="css"], .stMarkdown, div, p, span, label {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--charcoal);
    }
    h1, h2, h3, h4, .brand-title, .hero-title, .kpi-value, .card-title {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.025em;
    }

    .stApp {
        background-color: #F7F8F2 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(22, 163, 106, 0.05) 0px, transparent 40%),
            radial-gradient(at 100% 0%, rgba(18, 55, 42, 0.04) 0px, transparent 45%),
            radial-gradient(at 50% 100%, rgba(197, 244, 103, 0.04) 0px, transparent 50%);
        background-attachment: fixed;
    }

    .main .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1300px !important;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #16A36A 0%, #12372A 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.60rem 1.3rem !important;
        box-shadow: 0 4px 14px rgba(22, 163, 106, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #12372A 0%, #0d281e 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(18, 55, 42, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid rgba(18, 55, 42, 0.15) !important;
        background-color: #FFFFFF !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid rgba(18, 55, 42, 0.08);
        padding-bottom: 4px;
        background: #FFFFFF;
        padding: 8px 14px 4px 14px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(18, 55, 42, 0.03);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        color: #8A9990;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(22, 163, 106, 0.12) !important;
        color: #12372A !important;
        border-bottom: 3px solid #16A36A !important;
        font-weight: 800 !important;
    }
</style>
"""
st.markdown(ECO_TECH_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATABASE INITIALIZATION & SESSION STATE
# -----------------------------------------------------------------------------
db.init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_data" not in st.session_state:
    st.session_state.user_data = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None


def render_top_navbar():
    """Renders the top branding and user profile header with logout."""
    u_role = st.session_state.get("user_role", "user")
    u_name = st.session_state.get("full_name", "User")
    u_id = st.session_state.get("username", "")

    role_badge = "MESS MANAGER" if u_role == "manager" else "STUDENT PORTAL"
    role_bg = "rgba(18,55,42,0.12)" if u_role == "manager" else "rgba(22,163,106,0.15)"
    role_color = "#12372A" if u_role == "manager" else "#16A36A"

    nav_col1, nav_col2, nav_col3 = st.columns([2, 2.5, 1.2])

    with nav_col1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.75rem; padding:0.4rem 0;">
            <div style="width:38px; height:38px; background:linear-gradient(135deg, #12372A 0%, #16A36A 100%); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; box-shadow:0 4px 12px rgba(22, 163, 106, 0.25);">
                🌿
            </div>
            <div>
                <div style="font-size:1.3rem; font-weight:800; color:#12372A; line-height:1.1;">MessMind <span style="color:#16A36A;">AI</span></div>
                <div style="font-size:0.72rem; font-weight:700; color:#8A9990; text-transform:uppercase; letter-spacing:0.5px;">IIIT Dharwad Hostel Mess 1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem; height:100%; padding-top:0.4rem;">
            <span style="background:{role_bg}; color:{role_color}; font-weight:800; font-size:0.75rem; padding:4px 12px; border-radius:20px; letter-spacing:0.5px; border:1px solid rgba(18,55,42,0.1);">
                {role_badge}
            </span>
            <span style="font-size:0.85rem; color:#12372A; font-weight:700;">
                👤 {u_name} <span style="color:#8A9990; font-weight:500;">({u_id.upper()})</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

    with nav_col3:
        if st.button("🚪 Sign Out", use_container_width=True, key="top_signout_btn"):
            db.log_audit_event("USER_LOGOUT", u_id, f"User {u_id} logged out")
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.user_role = None
            st.session_state.username = None
            st.session_state.full_name = None
            st.rerun()

    st.markdown("<hr style='margin:0.4rem 0 1.0rem 0; border:none; border-top:1px solid rgba(18,55,42,0.08);'>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. MAIN ROUTER
# -----------------------------------------------------------------------------
def main():
    if not st.session_state.get("authenticated", False):
        portal_login.render_login_page()
    else:
        render_top_navbar()
        role = st.session_state.get("user_role")
        if role == "manager":
            portal_manager.render_manager_portal()
        elif role == "student":
            portal_student.render_student_portal()
        else:
            st.error("Unknown user role detected. Re-authenticating...")
            st.session_state.authenticated = False
            st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:3.5rem; padding-top:1.5rem; border-top:1px solid rgba(18,55,42,0.08); font-size:0.78rem; color:#8A9990;">
        🌿 <strong>MessMind AI</strong> • Multi-Plate Adaptive Food Waste Intelligence & Smart Hostel Mess Management System<br>
        Frugal Innovation Coursework • Indian Institute of Information Technology (IIIT) Dharwad • Powered by OpenCV, Scikit-learn & SQLite
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
