"""
portal_login.py
===============
Common Login Page for MessMind AI with Role-Based Authentication.
Supports PBKDF2-HMAC password verification, role-based redirection,
and instant 1-click demo login buttons for college viva evaluation.
"""

import streamlit as st
import database as db


def render_login_page():
    """Renders the common login page for Mess Manager and Students."""
    
    # Outer layout container
    col_l, col_center, col_r = st.columns([1, 1.8, 1])

    with col_center:
        st.markdown("""
        <div style="text-align:center; margin-top:2.0rem; margin-bottom:1.5rem;">
            <div style="display:inline-flex; align-items:center; gap:0.6rem; background:rgba(22,163,106,0.12); padding:6px 16px; border-radius:30px; margin-bottom:0.8rem; border:1px solid rgba(22,163,106,0.25);">
                <span style="font-size:1.1rem;">🌿</span>
                <span style="color:#12372A; font-weight:800; font-size:0.85rem; letter-spacing:0.04em; text-transform:uppercase;">Hostel Dining Intelligence</span>
            </div>
            <h1 style="color:#12372A; font-size:2.4rem; font-weight:900; margin:0 0 0.4rem 0; letter-spacing:-0.03em;">Welcome to MessMind AI</h1>
            <p style="color:#64748B; font-size:1.05rem; font-weight:500; margin:0;">Smarter Meals. Less Waste. Better Mess Management.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.10); border-radius:20px; padding:2.2rem 2.0rem; box-shadow:0 12px 36px rgba(18,55,42,0.06); margin-bottom:1.5rem;">
            <div style="font-size:1.15rem; font-weight:800; color:#12372A; margin-bottom:0.3rem;">Sign In to Your Portal</div>
            <div style="font-size:0.85rem; color:#8A9990; margin-bottom:1.4rem;">Select your role credentials to access administrative or student dining features.</div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input(
                "Username or Roll Number",
                placeholder="e.g. manager or 24bcs042",
                key="login_user_input"
            )
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_pass_input"
            )
            
            submit_btn = st.form_submit_button(
                "🚀 Sign In to Portal",
                use_container_width=True
            )

            if submit_btn:
                if not username_input or not password_input:
                    st.error("Please enter both username/roll number and password.")
                else:
                    user = db.authenticate_user(username_input.strip(), password_input.strip())
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user
                        st.session_state.user_role = user.get("role", "student")
                        st.session_state.user_id = user.get("id")
                        st.session_state.username = user.get("username")
                        st.session_state.full_name = user.get("full_name")
                        st.session_state.hostel = user.get("hostel", "Mess 1")
                        st.session_state.room_number = user.get("room_number", "")
                        st.session_state.dietary_preference = user.get("dietary_preference", "Vegetarian")
                        
                        # Set default initial page based on role
                        if user.get("role") == "manager":
                            st.session_state.current_page = "🏢 Dashboard"
                        else:
                            st.session_state.current_page = "🏠 Student Home"
                        
                        db.log_audit_event("USER_LOGIN", user.get("username"), f"User logged in as {user.get('role')}")
                        st.success(f"Welcome back, {user.get('full_name')}! Redirecting to {user.get('role').capitalize()} Portal...")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Invalid username or password. Please verify your credentials.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Viva Demo Logins Card
        st.markdown("""
        <div style="background:#FFFFFF; border:1.5px dashed rgba(22,163,106,0.35); border-radius:18px; padding:1.4rem 1.6rem; box-shadow:0 4px 18px rgba(18,55,42,0.03); margin-bottom:1.5rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span style="font-size:0.92rem; font-weight:800; color:#12372A;">⚡ Quick Evaluation Demo Logins</span>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.72rem; padding:2px 8px; border-radius:6px; text-transform:uppercase;">Viva Ready</span>
            </div>
            <p style="font-size:0.82rem; color:#64748B; margin:0 0 1.0rem 0; line-height:1.4;">
                Click below to instantly authenticate into either role without manual typing during demonstrations.
            </p>
        """, unsafe_allow_html=True)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            if st.button("👔 Sign in as Mess Manager", use_container_width=True, key="demo_btn_mgr"):
                user = db.authenticate_user("manager", "Manager@123")
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_data = user
                    st.session_state.user_role = "manager"
                    st.session_state.user_id = user.get("id")
                    st.session_state.username = user.get("username")
                    st.session_state.full_name = user.get("full_name")
                    st.session_state.hostel = user.get("hostel", "Mess 1")
                    st.session_state.current_page = "🏢 Dashboard"
                    db.log_audit_event("DEMO_LOGIN", "manager", "Demo login as Mess Manager")
                    st.rerun()

        with d_col2:
            if st.button("🎓 Sign in as Student", use_container_width=True, key="demo_btn_stu"):
                user = db.authenticate_user("24bcs042", "Student@123")
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_data = user
                    st.session_state.user_role = "student"
                    st.session_state.user_id = user.get("id")
                    st.session_state.username = user.get("username")
                    st.session_state.full_name = user.get("full_name")
                    st.session_state.hostel = user.get("hostel", "Ganga Block")
                    st.session_state.room_number = user.get("room_number", "B-204")
                    st.session_state.dietary_preference = user.get("dietary_preference", "Vegetarian")
                    st.session_state.current_page = "🏠 Student Home"
                    db.log_audit_event("DEMO_LOGIN", "24bcs042", "Demo login as Student")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Security Architecture Info Box
        st.markdown("""
        <div style="background:#F7F8F2; border-left:4px solid #16A36A; border-radius:12px; padding:1.1rem 1.3rem; font-size:0.82rem; color:#18231F; line-height:1.5;">
            <div style="font-weight:800; color:#12372A; margin-bottom:0.3rem;">🛡️ Role-Based Access Control (RBAC) Architecture</div>
            • <strong>Manager Role:</strong> Complete operational authority (Menu CRUD, Demand approvals, Multi-Plate Vision AI, Staff Records, Settings).<br>
            • <strong>Student Role:</strong> Lightweight read/write interface (Today's Published Menu, Meal RSVP Intent, Food Feedback, Notice Board).<br>
            • <strong>Security:</strong> PBKDF2-HMAC-SHA256 password hashing with unique 16-byte random salt per user. Session state role isolation prevents URL escalation.
        </div>
        """, unsafe_allow_html=True)

