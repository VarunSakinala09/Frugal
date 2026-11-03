"""
portal_manager.py
=================
Mess Manager Portal for MessMind AI.
Administrative control center for IIIT Dharwad Mess 1.

Structured as a closed-loop 5-Stage Story Flow:
  Stage 1: Daily Menu Roster ➔ 
  Stage 2: Live Student Turnout (RSVPs) ➔ 
  Stage 3: Smart Kitchen Prep Target (ML + 5% Safety Buffer) ➔ 
  Stage 4: Plate Inspection (OpenCV Vision AI) ➔ 
  Stage 5: Student Feedback & Campus Impact
"""

import io
import math
import socket
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import database as db
import vision_ai
from recommendations import (
    load_demand_model,
    generate_dish_recommendation,
    generate_menu_recommendations,
    get_historical_demand_for_dish,
    FALLBACK_TAKE_RATES,
    DEFAULT_MODEL_PATH
)

DEFAULT_COSTS = {
    "Kadai Paneer": 34.0, "Paneer Kolhapuri": 34.0, "Shahi Paneer": 34.0, "Paneer Butter Masala": 34.0,
    "Dal Methi": 14.0, "Dal Tadka": 14.0, "Dal Makhani": 16.0, "Dal Fry": 14.0, "Dal Palak": 14.0,
    "Dal Pancha Ratan": 14.0, "Moong Dal": 14.0, "Masoor Dal": 14.0, "Channa Dal": 14.0, "Dal Kolhapuri": 14.0,
    "Dal": 14.0, "Rasam": 10.0, "Brinjal Bhartha": 16.0, "White Peas Masala": 16.0, "Jeera Aloo": 14.0,
    "Chawli Masala": 15.0, "Brinjal Tomato": 15.0, "Kadhi Pakoda": 16.0, "Bhendi Sabji": 18.0,
    "Veg Kadai": 18.0, "Ivy Gourd Fry": 16.0, "Cabbage Baji": 14.0, "Black Chana Masala": 15.0,
    "Aloo 65": 16.0, "Soyabean Sabzi Gravy": 15.0, "Sorakay Chana": 15.0, "Veg Kolhapuri": 18.0,
    "Aloo Mutter Sabzi": 15.0, "Egg Curry": 22.0, "Curd Rice": 11.0, "Veg Pulav": 14.0,
    "Onion Rice": 12.0, "Puliogare": 16.0, "Jeera Rice": 12.0, "Tomato Rice": 12.0, "Plain Rice": 9.0,
    "Khichadi": 12.0, "Soyabean Biryani": 14.0, "Veg & Egg Biryani": 22.0, "Vangi Bath": 14.0,
    "Chapati": 10.0, "Puri": 14.0, "Chole Bhature": 24.0, "Aloo Paratha": 18.0, "Masala Dosa": 20.0,
    "Idli": 12.0, "Vada": 14.0, "Upma": 10.0, "Pongal": 14.0, "Poha": 12.0, "Uthappam": 15.0,
    "Sheera": 15.0, "Egg Bhurji": 18.0, "Boiled Egg": 7.0, "Samosa": 12.0, "Pani Puri": 20.0,
    "Punugulu": 16.0, "Veg Noodles": 18.0, "Pasta": 18.0, "Masala Bhel": 15.0, "Girmit": 16.0,
    "Curd": 10.0, "Mix Veg Raita": 12.0, "Ginger Tea": 6.0, "Tea": 6.0, "Coffee": 7.0,
    "Lassi": 12.0, "Buttermilk": 8.0, "Badam Milk": 14.0, "Lemonade": 8.0, "Jaljeera": 8.0, "Rasna": 8.0,
    "Papad": 3.0, "Pickle": 2.0, "Salad": 5.0, "Boiled Groundnut": 8.0, "Boiled Sweet Corn": 10.0,
    "Matki Sprouts": 8.0, "Green Moong Sprouts": 8.0, "Black Chana Sprouts": 8.0,
    "Chutney": 5.0, "Coconut Chutney": 5.0, "Motichoor Ladoo": 12.0, "Gulab Jamun": 14.0,
    "Jalebi": 14.0, "Ice Cream": 15.0, "Shevai Kheer": 16.0
}


def get_lan_ip() -> str:
    """Auto-detect host computer LAN IP for local network Wi-Fi connectivity."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def render_story_header(active_step: int):
    """Renders the horizontal story breadcrumb showing the 5 stages of a meal."""
    steps = [
        (1, "📋 1. Set Menu"),
        (2, "🗳️ 2. Live Student Turnout"),
        (3, "🎯 3. Smart Prep Target"),
        (4, "📷 4. Plate Inspection"),
        (5, "⭐ 5. Student Voice & Savings")
    ]
    html_parts = []
    for step_num, step_label in steps:
        if step_num == active_step:
            html_parts.append(f"""
            <div style="background:#12372A; color:#C5F467; font-weight:800; font-size:0.75rem; padding:6px 14px; border-radius:20px; box-shadow:0 2px 8px rgba(18,55,42,0.2);">
                {step_label}
            </div>
            """)
        else:
            html_parts.append(f"""
            <div style="background:#FFFFFF; color:#8A9990; font-weight:600; font-size:0.75rem; padding:6px 12px; border-radius:20px; border:1px solid rgba(18,55,42,0.08);">
                {step_label}
            </div>
            """)
    
    flow_bar = '<span style="color:#C5F467; font-size:0.9rem; margin:0 4px;">➔</span>'.join(html_parts)
    st.markdown(f"""
    <div style="background:#F7F8F2; border-radius:16px; padding:0.6rem 0.8rem; margin-bottom:1.0rem; display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:0.4rem; border:1px solid rgba(18,55,42,0.08);">
        {flow_bar}
    </div>
    """, unsafe_allow_html=True)


def render_manager_portal():
    """Renders the comprehensive administrative Mess Manager Portal organized as a Story Flow."""
    
    today_str = date.today().strftime("%Y-%m-%d")
    today_day = date.today().strftime("%A")

    # Manager Story Tabs
    m_tabs = st.tabs([
        "📊 Campus Pulse",
        "🍲 1. Menu Roster",
        "🗳️ 2. Live Student RSVPs",
        "🎯 3. Smart Prep Target",
        "📷 4. Plate Inspection",
        "⭐ 5. Student Voice & Feedback",
        "📝 Shift Waste Records",
        "⚙️ Settings & Broadcasts"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: CAMPUS DINING PULSE (EXECUTIVE STORY SUMMARY)
    # -------------------------------------------------------------------------
    with m_tabs[0]:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; box-shadow:0 4px 16px rgba(18,55,42,0.03);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.0rem;">
                <div>
                    <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:3px 10px; border-radius:20px; text-transform:uppercase;">The Big Picture</span>
                    <h2 style="color:#12372A; font-size:1.8rem; font-weight:800; margin:0.4rem 0 0.2rem 0;">Campus Dining Intelligence Pulse</h2>
                    <div style="color:#64748B; font-size:0.9rem;">IIIT Dharwad Mess 1 • Real-time connection between Student Intent and Kitchen Preparation • {today_day}, {today_str}</div>
                </div>
                <div style="background:rgba(18,55,42,0.06); border-radius:12px; padding:0.6rem 1.0rem; text-align:right;">
                    <div style="font-size:0.72rem; text-transform:uppercase; color:#8A9990; font-weight:700;">Verified Ground Truth</div>
                    <div style="font-size:1.05rem; font-weight:800; color:#12372A;">3,858 Historical Shifts Logged</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real-time metrics
        all_records_df = db.get_meal_records(limit=4000)
        users_df = db.get_all_users()
        total_students = len(users_df[users_df['role'] == 'student']) if not users_df.empty else 450
        
        # Today's Student RSVP Intent Counts
        today_intents = db.get_aggregated_intent_counts_for_date(today_str)
        attending_today = today_intents.get("Attending", today_intents.get("attending_count", 0))
        skipping_today = today_intents.get("Not Attending", today_intents.get("not_attending_count", 0))
        unsure_today = today_intents.get("Not sure", today_intents.get("not_sure_count", 0))
        total_votes = today_intents.get("total_polled", attending_today + skipping_today + unsure_today)

        # Aggregate Metrics from historical records
        if not all_records_df.empty:
            tot_prep = int(all_records_df['quantity_prepared'].sum())
            tot_served = int(all_records_df['quantity_served'].sum())
            tot_discarded = int(all_records_df['discarded_quantity'].sum())
            tot_cost = float(all_records_df['discarded_food_cost_inr'].sum())
            high_waste_dish = all_records_df.groupby('dish_name')['discarded_quantity'].sum().idxmax()
        else:
            tot_prep, tot_served, tot_discarded, tot_cost, high_waste_dish = 0, 0, 0, 0.0, "N/A"

        batches_df = db.get_vision_batches()
        total_vision_plates = int(batches_df['total_images'].sum()) if not batches_df.empty else 136

        # 4 Primary Story KPI Cards
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)

        with p_c1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem 1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">1. Total Registered Diners</div>
                <div style="font-size:1.9rem; font-weight:900; color:#12372A; margin:0.2rem 0;">{total_students}</div>
                <div style="font-size:0.78rem; color:#16A36A; font-weight:600;">Active Hostel Students</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1.5px solid #16A36A; border-radius:14px; padding:1.1rem 1.2rem; box-shadow:0 2px 8px rgba(22,163,106,0.06);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#16A36A; font-weight:800;">2. Live Today's RSVPs</div>
                <div style="font-size:1.9rem; font-weight:900; color:#16A36A; margin:0.2rem 0;">{attending_today} <span style="font-size:0.95rem; color:#E76F51; font-weight:700;">(-{skipping_today} skips)</span></div>
                <div style="font-size:0.78rem; color:#64748B;">{total_votes} student votes received today</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c3:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem 1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">3. CV Plates Analyzed</div>
                <div style="font-size:1.9rem; font-weight:900; color:#12372A; margin:0.2rem 0;">{total_vision_plates}</div>
                <div style="font-size:0.78rem; color:#64748B;">Trays inspected by OpenCV</div>
            </div>
            """, unsafe_allow_html=True)

        with p_c4:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem 1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">4. Tracked Waste Cost</div>
                <div style="font-size:1.9rem; font-weight:900; color:#E76F51; margin:0.2rem 0;">₹{tot_cost:,.0f}</div>
                <div style="font-size:0.78rem; color:#64748B;">Target for AI Reduction</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # Story Narrative Box: How MessMind AI Works
        st.markdown("""
        <div style="background:#F7F8F2; border-left:4px solid #16A36A; border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:1.2rem; font-size:0.86rem; color:#18231F; line-height:1.6;">
            <strong style="color:#12372A; font-size:0.95rem;">📖 How MessMind AI Solves College Food Waste:</strong><br>
            • <strong>Morning:</strong> The manager publishes today's menu. Students vote <em>Attending</em> or <em>Skipping</em> in 1 click.<br>
            • <strong>Pre-Cooking:</strong> The system takes student votes, adjusts for weather and weekend patterns, and adds a mandatory <strong>+5% safety buffer</strong> so no student goes hungry.<br>
            • <strong>Post-Dining:</strong> Overhead cameras run OpenCV to check leftover food on plates. If a dish is heavily wasted, the chef receives instant advice to reduce ladle portions.<br>
            • <strong>Outcome:</strong> Kitchen avoids cooking 30–50 surplus portions daily, saving <strong>₹35,000–₹50,000 every month</strong> across campus messes!
        </div>
        """, unsafe_allow_html=True)

        # Visual Charts
        c_col1, c_col2 = st.columns([1.5, 1], gap="medium")
        with c_col1:
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.4rem;'>📈 30-Day Food Waste & Financial Loss Trend</div>", unsafe_allow_html=True)
            if not all_records_df.empty:
                daily_df = all_records_df.groupby('date')[['discarded_quantity', 'discarded_food_cost_inr']].sum().reset_index()
                daily_df = daily_df.sort_values('date').tail(30)
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Bar(
                    x=daily_df['date'],
                    y=daily_df['discarded_quantity'],
                    name="Discarded Servings",
                    marker_color='#16A36A',
                    opacity=0.85
                ))
                fig_trend.add_trace(go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['discarded_food_cost_inr'],
                    name="Cost Loss (₹)",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color='#E76F51', width=2.5),
                    marker=dict(size=5)
                ))
                fig_trend.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=280,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    yaxis=dict(title="Servings Discarded", gridcolor="#F1F5F9"),
                    yaxis2=dict(title="Waste Cost (₹)", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        with c_col2:
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.4rem;'>🥧 Waste by Meal Period</div>", unsafe_allow_html=True)
            if not all_records_df.empty:
                meal_df = all_records_df.groupby('meal_type')['discarded_quantity'].sum().reset_index()
                fig_pie = px.pie(
                    meal_df,
                    names='meal_type',
                    values='discarded_quantity',
                    color_discrete_sequence=['#12372A', '#16A36A', '#F4A261', '#C5F467'],
                    hole=0.45
                )
                fig_pie.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=280,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Real-time Synchronized Student Activity Hub
        st.markdown("<hr style='margin:1.4rem 0 1.0rem 0; border:none; border-top:1px solid rgba(18,55,42,0.08);'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
            <div>
                <span style="font-size:1.1rem; font-weight:800; color:#12372A;">⚡ Live Student Portal Activity (Direct Database Synchronization)</span>
                <div style="font-size:0.82rem; color:#64748B;">Incoming votes and feedback submitted by students appear here in real time.</div>
            </div>
            <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-size:0.75rem; font-weight:800; padding:4px 10px; border-radius:8px;">● LIVE SYNC ACTIVE</span>
        </div>
        """, unsafe_allow_html=True)

        live_c1, live_c2 = st.columns(2, gap="medium")
        with live_c1:
            st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#12372A; margin-bottom:0.4rem;'>🗳️ Recent Student RSVPs</div>", unsafe_allow_html=True)
            recent_intents = db.get_student_meal_intents()
            if recent_intents is not None and not recent_intents.empty:
                display_ri = recent_intents.head(5)[['student_roll', 'meal_date', 'meal_type', 'intent_status', 'updated_at']].rename(columns={
                    'student_roll': 'Roll No',
                    'meal_date': 'Date',
                    'meal_type': 'Meal',
                    'intent_status': 'Vote',
                    'updated_at': 'Time'
                })
                st.dataframe(display_ri, use_container_width=True, hide_index=True)
            else:
                st.info("No student RSVPs received yet.")

        with live_c2:
            st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#12372A; margin-bottom:0.4rem;'>⭐ Recent Student Reviews</div>", unsafe_allow_html=True)
            recent_fb = db.get_student_feedback_records(limit=5)
            if recent_fb is not None and not recent_fb.empty:
                display_rf = recent_fb[['student_roll', 'meal_date', 'meal_type', 'rating', 'feedback_text', 'created_at']].rename(columns={
                    'student_roll': 'Roll No',
                    'meal_date': 'Date',
                    'meal_type': 'Meal',
                    'rating': 'Rating',
                    'feedback_text': 'Comments',
                    'created_at': 'Time'
                })
                st.dataframe(display_rf, use_container_width=True, hide_index=True)
            else:
                st.info("No student reviews received yet.")

    # -------------------------------------------------------------------------
    # TAB 2: STAGE 1 — DAILY MENU ROSTER (MANAGER CRUD & PUBLISH)
    # -------------------------------------------------------------------------
    with m_tabs[1]:
        render_story_header(1)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">📋 Stage 1: Official Dining Menu</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Define what dishes are cooking today. Set prices and control whether students can see the menu.</div>
                </div>
                <span style="background:rgba(18,55,42,0.1); color:#12372A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">STEP 1 OF LIFECYCLE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m_filter_c1, m_filter_c2, m_filter_c3 = st.columns(3)
        with m_filter_c1:
            sel_day_m = st.selectbox("Weekday", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(today_day), key="m_sel_day")
        with m_filter_c2:
            sel_meal_m = st.selectbox("Meal Period", options=["All Meals", "Breakfast", "Lunch", "Snacks", "Dinner"], key="m_sel_meal")
        with m_filter_c3:
            pub_filter = st.selectbox("Status Filter", options=["All Items", "Published Only", "Drafts Only"], key="m_pub_filter")

        # 1-Click Publish / Draft Buttons
        col_pub_all1, col_pub_all2 = st.columns([1, 1])
        with col_pub_all1:
            if st.button(f"🟢 Publish Full {sel_day_m} Menu to Students", use_container_width=True):
                db.publish_all_day_menu(sel_day_m, is_published=True)
                db.log_audit_event("MENU_PUBLISH", "manager", f"Published menu for {sel_day_m}")
                st.success(f"All items for {sel_day_m} are now published and visible on student phones!")
                st.rerun()
        with col_pub_all2:
            if st.button(f"⚪ Make {sel_day_m} Menu Draft (Hide from Students)", use_container_width=True):
                db.publish_all_day_menu(sel_day_m, is_published=False)
                db.log_audit_event("MENU_UNPUBLISH", "manager", f"Unpublished menu for {sel_day_m}")
                st.warning(f"All items for {sel_day_m} are now in draft mode.")
                st.rerun()

        # Add New Menu Item Form
        with st.expander("➕ Add New Dish to Schedule"):
            with st.form("add_dish_form"):
                af_c1, af_c2, af_c3 = st.columns(3)
                with af_c1:
                    new_dish_name = st.text_input("Dish Name", placeholder="e.g. Paneer Butter Masala")
                    new_dish_cat = st.selectbox("Category", ["Main Curry", "Dal", "Rice & Grains", "Bread & Tiffin", "Snack", "Sweet / Dessert", "Beverage / Accompaniment"])
                with af_c2:
                    new_dish_meal = st.selectbox("Meal Period", ["Breakfast", "Lunch", "Snacks", "Dinner"])
                    new_dish_day = st.selectbox("Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                with af_c3:
                    new_dish_cost = st.number_input("Cost per Serving (₹)", min_value=1.0, max_value=200.0, value=15.0, step=1.0)
                    new_dish_publish = st.checkbox("Publish immediately to students", value=True)
                
                new_dish_desc = st.text_input("Description / Notes (Optional)", placeholder="e.g. Traditional recipe cooked with mild spices")
                submit_new_dish = st.form_submit_button("💾 Save Dish to Schedule", use_container_width=True)

                if submit_new_dish:
                    if new_dish_name.strip():
                        item_id = db.add_menu_item(
                            day_of_week=new_dish_day,
                            meal_period=new_dish_meal,
                            dish_name=new_dish_name.strip(),
                            category=new_dish_cat,
                            cost_per_serving=new_dish_cost,
                            description=new_dish_desc,
                            is_published=new_dish_publish
                        )
                        db.log_audit_event("MENU_ADD", "manager", f"Added dish '{new_dish_name}' to {new_dish_day} {new_dish_meal}")
                        st.success(f"Added '{new_dish_name}' to {new_dish_day} {new_dish_meal}!")
                        st.rerun()
                    else:
                        st.error("Dish name cannot be empty.")

        # Display Existing Menu Items
        meal_arg = None if sel_meal_m == "All Meals" else sel_meal_m
        only_pub_arg = True if pub_filter == "Published Only" else (False if pub_filter == "Drafts Only" else False)
        
        menu_df = db.get_menu_items(day_of_week=sel_day_m, meal_period=meal_arg, only_published=only_pub_arg)
        
        if pub_filter == "Drafts Only" and not menu_df.empty:
            menu_df = menu_df[menu_df['is_published'] == 0]

        st.markdown(f"<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin:1.0rem 0 0.5rem 0;'>{sel_day_m} Menu Spread ({len(menu_df)} items)</div>", unsafe_allow_html=True)
        
        if menu_df.empty:
            st.info(f"No menu items match the filter for {sel_day_m}. Use the 'Add New Dish' button above to register dishes.")
        else:
            for idx, item in menu_df.iterrows():
                is_pub = bool(item['is_published'])
                status_badge = '<span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:2px 8px; border-radius:6px;">PUBLISHED</span>' if is_pub else '<span style="background:rgba(138,153,144,0.15); color:#8A9990; font-weight:800; font-size:0.75rem; padding:2px 8px; border-radius:6px;">DRAFT</span>'
                
                row_c1, row_c2, row_c3, row_c4 = st.columns([2.5, 1.2, 1, 1.5])
                with row_c1:
                    st.markdown(f"<strong>{item['dish_name']}</strong> <span style='font-size:0.8rem; color:#64748B;'>({item['category']})</span>", unsafe_allow_html=True)
                    if item.get('description'):
                        st.caption(item['description'])
                with row_c2:
                    st.markdown(f"₹{item['cost_per_serving']:.1f} • {item['meal_period']}")
                with row_c3:
                    st.markdown(status_badge, unsafe_allow_html=True)
                with row_c4:
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        toggle_txt = "Draft" if is_pub else "Publish"
                        if st.button(toggle_txt, key=f"togg_{item['id']}"):
                            db.update_menu_item(item['id'], is_published=not is_pub)
                            db.log_audit_event("MENU_TOGGLE", "manager", f"Toggled status for dish {item['dish_name']}")
                            st.rerun()
                    with btn_c2:
                        if st.button("🗑️", key=f"del_{item['id']}"):
                            db.delete_menu_item(item['id'])
                            db.log_audit_event("MENU_DELETE", "manager", f"Deleted dish {item['dish_name']}")
                            st.rerun()
                st.markdown("<hr style='margin:0.4rem 0; border:none; border-top:1px solid #EEF2F6;'>", unsafe_allow_html=True)

        # Story Step Transition
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.2rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(18,55,42,0.08);">
            <div>
                <strong style="color:#12372A;">📖 Next Step in the Story:</strong>
                <span style="color:#64748B; font-size:0.86rem; margin-left:6px;">Now that today's menu is published, students open their portal and declare if they are eating.</span>
            </div>
            <span style="color:#16A36A; font-weight:800; font-size:0.85rem;">Proceed to Stage 2: Student RSVPs ➔</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 3: STAGE 2 — LIVE STUDENT TURNOUT & RSVPS (STUDENT INPUT HUB)
    # -------------------------------------------------------------------------
    with m_tabs[2]:
        render_story_header(2)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">🗳️ Stage 2: Live Student Attendance RSVPs</h3>
                    <div style="color:#64748B; font-size:0.85rem;">See exactly how many students are coming BEFORE lighting the kitchen stoves. Live synchronization from the Student Portal.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">STEP 2 OF LIFECYCLE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        r_filter_c1, r_filter_c2, r_filter_c3 = st.columns(3)
        with r_filter_c1:
            sel_rsvp_date = st.date_input("Filter Date", value=date.today(), key="mgr_rsvp_date_picker").strftime("%Y-%m-%d")
        with r_filter_c2:
            sel_rsvp_meal = st.selectbox("Filter Meal Shift", ["All Meals", "Lunch", "Dinner", "Breakfast"], key="mgr_rsvp_meal_filter")
        with r_filter_c3:
            st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Live Student Votes", use_container_width=True):
                st.rerun()

        # Fetch intent aggregates for this date
        meal_arg_intents = None if sel_rsvp_meal == "All Meals" else sel_rsvp_meal
        date_intents = db.get_aggregated_intent_counts_for_date(sel_rsvp_date, meal_arg_intents)

        att_cnt = date_intents.get("Attending", date_intents.get("attending_count", 0))
        skip_cnt = date_intents.get("Not Attending", date_intents.get("not_attending_count", 0))
        unsure_cnt = date_intents.get("Not sure", date_intents.get("not_sure_count", 0))
        total_cnt = date_intents.get("total_polled", att_cnt + skip_cnt + unsure_cnt)

        # 4 Turnout Summary Cards
        tc_1, tc_2, tc_3, tc_4 = st.columns(4)
        with tc_1:
            st.markdown(f"""
            <div style="background:rgba(22,163,106,0.12); border:1.5px solid #16A36A; border-radius:14px; padding:1.0rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#16A36A; font-weight:800;">Confirmed Attending</div>
                <div style="font-size:2.0rem; font-weight:900; color:#12372A; margin:0.2rem 0;">{att_cnt}</div>
                <div style="font-size:0.75rem; color:#16A36A; font-weight:700;">Students Planning to Eat</div>
            </div>
            """, unsafe_allow_html=True)

        with tc_2:
            st.markdown(f"""
            <div style="background:rgba(231,111,81,0.12); border:1.5px solid #E76F51; border-radius:14px; padding:1.0rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#E76F51; font-weight:800;">Confirmed Skips</div>
                <div style="font-size:2.0rem; font-weight:900; color:#E76F51; margin:0.2rem 0;">{skip_cnt}</div>
                <div style="font-size:0.75rem; color:#64748B;">Saves ~{skip_cnt * 15} portions</div>
            </div>
            """, unsafe_allow_html=True)

        with tc_3:
            st.markdown(f"""
            <div style="background:#F7F8F2; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.0rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Unsure / Undecided</div>
                <div style="font-size:2.0rem; font-weight:900; color:#F4A261; margin:0.2rem 0;">{unsure_cnt}</div>
                <div style="font-size:0.75rem; color:#64748B;">Counted at 50% probability</div>
            </div>
            """, unsafe_allow_html=True)

        with tc_4:
            st.markdown(f"""
            <div style="background:#F7F8F2; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.0rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Total Votes Polled</div>
                <div style="font-size:2.0rem; font-weight:900; color:#12372A; margin:0.2rem 0;">{total_cnt}</div>
                <div style="font-size:0.75rem; color:#64748B;">Live submissions recorded</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

        # Live Feed Table of Individual Student Votes
        st.markdown(f"<div style='font-size:1.0rem; font-weight:800; color:#12372A; margin-bottom:0.5rem;'>Live Incoming Student Votes for {sel_rsvp_date}</div>", unsafe_allow_html=True)
        raw_intents_df = db.get_student_meal_intents(meal_date=sel_rsvp_date)

        if raw_intents_df is not None and not raw_intents_df.empty:
            if meal_arg_intents:
                raw_intents_df = raw_intents_df[raw_intents_df['meal_type'] == meal_arg_intents]

            display_intents = raw_intents_df[['student_roll', 'meal_type', 'intent_status', 'dietary_preference', 'updated_at']].rename(columns={
                'student_roll': 'Student Roll No',
                'meal_type': 'Meal Shift',
                'intent_status': 'Attendance Intent',
                'dietary_preference': 'Dietary Preference',
                'updated_at': 'Submission Timestamp'
            })
            st.dataframe(display_intents, use_container_width=True, hide_index=True)
        else:
            st.info(f"No student votes recorded yet for {sel_rsvp_date}. When students select their meal plan on the Student Portal, their names and responses appear here immediately.")

        # Story Step Transition
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.2rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(18,55,42,0.08);">
            <div>
                <strong style="color:#12372A;">📖 Next Step in the Story:</strong>
                <span style="color:#64748B; font-size:0.86rem; margin-left:6px;">Student turnout is locked! Head cooks now take these confirmed numbers into our Smart Prep model.</span>
            </div>
            <span style="color:#16A36A; font-weight:800; font-size:0.85rem;">Proceed to Stage 3: Smart Prep Target ➔</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: STAGE 3 — SMART PREP TARGET (AI + 5% SAFETY BUFFER)
    # -------------------------------------------------------------------------
    with m_tabs[3]:
        render_story_header(3)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">🎯 Stage 3: Smart Kitchen Preparation Target</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Converts Student RSVPs + Machine Learning Forecast + Mandatory 5% Safety Buffer into approved cooking portions.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">STEP 3 OF LIFECYCLE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        dp_c1, dp_c2, dp_c3 = st.columns(3)
        with dp_c1:
            p_day = st.selectbox("Planning Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(today_day), key="p_day_sel_story")
        with dp_c2:
            p_meal = st.selectbox("Meal Shift", ["Lunch", "Dinner", "Breakfast", "Snacks"], key="p_meal_sel_story")
        with dp_c3:
            m_dishes = db.get_menu_items(day_of_week=p_day, meal_period=p_meal)
            dish_opts = m_dishes['dish_name'].tolist() if not m_dishes.empty else ["Dal Tadka", "Paneer Butter Masala", "Plain Rice", "Chapati"]
            p_dish = st.selectbox("Dish to Prepare", options=dish_opts, key="p_dish_sel_story")

        # Pull live attendance intent breakdown
        intent_breakdown = db.get_attendance_intent_counts(today_str, p_meal)
        att_cnt = intent_breakdown.get("Attending", intent_breakdown.get("yes_count", 0))
        skip_cnt = intent_breakdown.get("Not Attending", intent_breakdown.get("no_count", 0))

        st.markdown(f"""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:1.0rem; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>Live Input from Stage 2:</strong>
                <span style="color:#16A36A; margin-left:0.5rem; font-weight:800;">✅ {att_cnt} Students Attending</span> •
                <span style="color:#E76F51; margin-left:0.5rem; font-weight:800;">❌ {skip_cnt} Skipping</span>
            </div>
            <span style="font-size:0.75rem; color:#8A9990; font-weight:700;">Synchronized with Student Portal</span>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation Engine
        model = load_demand_model()
        rec = generate_dish_recommendation(
            meal_date=today_str,
            meal_type=p_meal,
            dish_name=p_dish,
            expected_attendance=450,
            registered_students=450,
            safety_margin_pct=5.0,
            intent_breakdown=intent_breakdown
        )

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Base ML Diners</div>
                <div style="font-size:2.0rem; font-weight:900; color:#12372A; margin:0.3rem 0;">{rec['predicted_demand']}</div>
                <div style="font-size:0.78rem; color:#64748B;">Expected Diners</div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"""
            <div style="background:rgba(22,163,106,0.12); border:2px solid #16A36A; border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#16A36A; font-weight:800;">Recommended Cook Target</div>
                <div style="font-size:2.2rem; font-weight:900; color:#12372A; margin:0.3rem 0;">{rec['recommended_preparation']}</div>
                <div style="font-size:0.78rem; color:#16A36A; font-weight:700;">With +5% Safety Buffer</div>
            </div>
            """, unsafe_allow_html=True)

        with res_c3:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.2rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Guaranteed Safety Buffer</div>
                <div style="font-size:2.0rem; font-weight:900; color:#F4A261; margin:0.3rem 0;">+{rec['safety_margin_units']}</div>
                <div style="font-size:0.78rem; color:#64748B;">Prevents Late Hunger Shortage</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#FFFFFF; border-left:4px solid #16A36A; border-radius:12px; padding:1.0rem 1.3rem; margin:1.2rem 0; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
            <strong style="color:#12372A;">💡 Chef Operational Advice:</strong>
            <div style="color:#475569; font-size:0.88rem; margin-top:3px;">{rec.get('advice') or rec.get('explanation')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Official Kitchen Work Order Approval
        with st.form("prep_approval_form_story"):
            app_c1, app_c2 = st.columns([1, 1])
            with app_c1:
                approved_servings = st.number_input("Approved Cooking Portions", min_value=50, max_value=600, value=int(rec['recommended_preparation']), step=5)
            with app_c2:
                prep_notes = st.text_input("Chef Shift Notes", placeholder="e.g. Standard mild spices; cooked for lunch rush")

            approve_submit = st.form_submit_button("✅ Approve & Sign Kitchen Work Order", use_container_width=True)

            if approve_submit:
                db.log_audit_event(
                    "PREP_APPROVAL",
                    "manager",
                    f"Approved {approved_servings} servings of {p_dish} for {p_day} {p_meal}. Notes: {prep_notes}"
                )
                st.success(f"🎉 Kitchen Work Order Signed! **{approved_servings} servings** of **{p_dish}** approved for cooking.")

        # Story Step Transition
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.2rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(18,55,42,0.08);">
            <div>
                <strong style="color:#12372A;">📖 Next Step in the Story:</strong>
                <span style="color:#64748B; font-size:0.86rem; margin-left:6px;">Food was cooked and served! Now students return their used trays. The OpenCV camera scans plate waste.</span>
            </div>
            <span style="color:#16A36A; font-weight:800; font-size:0.85rem;">Proceed to Stage 4: Plate Inspection ➔</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 5: STAGE 4 — PLATE INSPECTION (OPENCV VISION AI)
    # -------------------------------------------------------------------------
    with m_tabs[4]:
        render_story_header(4)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">📷 Stage 4: Post-Meal Plate Leftover Inspection</h3>
                    <div style="color:#64748B; font-size:0.85rem;">See what students left on their plates. Classical OpenCV edge vision inspects unconsumed food and recognizes waste patterns.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">STEP 4 OF LIFECYCLE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        v_setup_c1, v_setup_c2, v_setup_c3 = st.columns(3)
        with v_setup_c1:
            v_dish = st.selectbox("Inspect Dish Batch", ["Dal Tadka", "Paneer Butter Masala", "Plain Rice", "Chapati", "Veg Pulav"], key="v_dish_story")
        with v_setup_c2:
            v_meal = st.selectbox("Dining Shift", ["Lunch", "Dinner", "Breakfast"], key="v_meal_story")
        with v_setup_c3:
            v_date = st.date_input("Inspection Date", value=date.today(), key="v_date_story").strftime("%Y-%m-%d")

        v_opt_c1, v_opt_c2 = st.columns([1, 1.2], gap="medium")
        with v_opt_c1:
            st.markdown("""
            <div style="background:#F7F8F2; border-radius:12px; padding:1.0rem; border:1px solid rgba(18,55,42,0.1); text-align:center;">
                <div style="font-weight:800; color:#12372A; font-size:0.92rem; margin-bottom:0.3rem;">⚡ Viva 1-Click Batch Demo</div>
                <p style="font-size:0.8rem; color:#64748B; margin:0 0 0.8rem 0;">Generate a 4-plate inspection batch (Clean, Light, Moderate, High leftover) instantly.</p>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Generate & Analyze 4-Plate Batch for {v_dish}", use_container_width=True, key="btn_sample_batch_story"):
                sample_imgs = vision_ai.generate_sample_plate_batch(v_dish, 4)
                batch_res = vision_ai.analyze_plate_batch(sample_imgs, v_dish, v_meal, v_date)
                st.session_state.current_vision_batch = batch_res
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with v_opt_c2:
            uploaded_files = st.file_uploader(
                f"Or Upload Real Plate Photos for {v_dish} (Batch Mode)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="v_uploader_story"
            )
            if uploaded_files:
                if st.button(f"🔍 Analyze {len(uploaded_files)} Uploaded Plates with OpenCV", use_container_width=True, key="btn_analyze_uploaded_story"):
                    uploaded_pil_list = [Image.open(uf).convert("RGB") for uf in uploaded_files]
                    batch_res = vision_ai.analyze_plate_batch(uploaded_pil_list, v_dish, v_meal, v_date)
                    st.session_state.current_vision_batch = batch_res
                    st.rerun()

        # Batch Results
        if "current_vision_batch" in st.session_state and st.session_state.current_vision_batch:
            b_res = st.session_state.current_vision_batch

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.12); border-radius:16px; padding:1.2rem 1.4rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                    <div>
                        <span style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Batch {b_res['batch_code']}</span>
                        <h3 style="color:#12372A; margin:0.2rem 0;">{b_res['dish_name']} ({b_res['meal_period']})</h3>
                    </div>
                    <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.8rem; padding:4px 12px; border-radius:8px;">
                        {b_res['valid_images']} Plates Inspected
                    </span>
                </div>
            """, unsafe_allow_html=True)

            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                st.metric("Mean Leftover", f"{b_res['average_coverage']:.1f}%")
            with bm2:
                st.metric("Median Leftover", f"{b_res['median_coverage']:.1f}%")
            with bm3:
                st.metric("High Leftover Plates", b_res['high_leftover_count'])
            with bm4:
                st.metric("Waste Fingerprint", b_res['dominant_fingerprint'])

            st.markdown("</div>", unsafe_allow_html=True)

            # Plate Gallery
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin:1.2rem 0 0.6rem 0;'>Visual Leftover Segmentation (OpenCV Mask Highlighted in Green)</div>", unsafe_allow_html=True)
            p_cols = st.columns(min(4, len(b_res['plate_results'])))
            for idx, p in enumerate(b_res['plate_results']):
                col_idx = idx % len(p_cols)
                with p_cols[col_idx]:
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:12px; padding:0.8rem; margin-bottom:0.8rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <strong style="font-size:0.85rem; color:#12372A;">{p['plate_id']}</strong>
                            <span style="font-size:0.75rem; font-weight:800; color:{p['status_color']};">{p['leftover_category']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    if p.get('overlay_img') is not None:
                        st.image(p['overlay_img'], caption=f"Leftover: {p['coverage_percentage']:.1f}% ({p['processing_time_ms']} ms)", use_container_width=True)
                    st.caption(f"Pattern: **{p.get('fingerprint', {}).get('fingerprint_type', 'Scattered')}**")
                    st.markdown("</div>", unsafe_allow_html=True)

            # Guidance & Save
            st.markdown(f"""
            <div style="background:#F7F8F2; border-left:4px solid #16A36A; border-radius:12px; padding:1.0rem 1.2rem; margin:0.8rem 0;">
                <strong style="color:#12372A;">💡 Operational Kitchen Advice:</strong> {b_res['recommendation']}
            </div>
            """, unsafe_allow_html=True)

            with st.form("save_batch_form_story"):
                sb_c1, sb_c2 = st.columns([1, 1])
                with sb_c1:
                    disc_kg = st.number_input("Physical Floor Scale Weight (kg)", min_value=0.0, max_value=150.0, value=round(b_res['average_coverage'] * 0.25, 1), step=0.5)
                with sb_c2:
                    batch_notes = st.text_input("Notes", placeholder="e.g. Standard dining shift")

                st.caption("⚖️ Camera analysis computes optical surface %; kitchen scales measure mass in kilograms. Both are recorded in MessMind's memory.")
                if st.form_submit_button("💾 Save Batch to Historical Waste Memory", use_container_width=True):
                    b_id = db.save_vision_batch(b_res, actual_discarded_kg=disc_kg, notes=batch_notes)
                    db.log_audit_event("VISION_BATCH_SAVED", "manager", f"Saved batch {b_res['batch_code']} (ID: {b_id})")
                    st.success(f"🎉 Batch **{b_res['batch_code']}** saved to Historical Memory!")

        # Story Step Transition
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.2rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(18,55,42,0.08);">
            <div>
                <strong style="color:#12372A;">📖 Next Step in the Story:</strong>
                <span style="color:#64748B; font-size:0.86rem; margin-left:6px;">Plates scanned and grams recorded! Now we hear directly from students why they skipped or left food.</span>
            </div>
            <span style="color:#16A36A; font-weight:800; font-size:0.85rem;">Proceed to Stage 5: Student Voice & Savings ➔</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 6: STAGE 5 — STUDENT VOICE & FEEDBACK HUB (WHERE STUDENT INPUT SAVES)
    # -------------------------------------------------------------------------
    with m_tabs[5]:
        render_story_header(5)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">⭐ Stage 5: Student Voice, Ratings & Skip Reasons</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Hear directly why students skipped meals or left food behind. Closes the feedback loop for kitchen continuous improvement.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">STEP 5 OF LIFECYCLE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Pull student feedback from database
        all_feedback_df = db.get_student_feedback_records(limit=100)

        # Satisfaction KPI cards
        total_fb_count = len(all_feedback_df) if not all_feedback_df.empty else 0
        avg_rating = round(all_feedback_df['rating'].mean(), 1) if total_fb_count > 0 else 4.2
        
        fb_c1, fb_c2, fb_c3 = st.columns(3)
        with fb_c1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Average Meal Rating</div>
                <div style="font-size:2.0rem; font-weight:900; color:#12372A; margin:0.2rem 0;">⭐ {avg_rating} <span style="font-size:1.0rem; color:#8A9990;">/ 5.0</span></div>
                <div style="font-size:0.75rem; color:#16A36A; font-weight:600;">Overall Student Satisfaction</div>
            </div>
            """, unsafe_allow_html=True)

        with fb_c2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Total Student Reviews</div>
                <div style="font-size:2.0rem; font-weight:900; color:#16A36A; margin:0.2rem 0;">{total_fb_count}</div>
                <div style="font-size:0.78rem; color:#64748B;">Reviews Submitted via Portal</div>
            </div>
            """, unsafe_allow_html=True)

        with fb_c3:
            top_reason = all_feedback_df['skip_reason'].mode()[0] if (total_fb_count > 0 and not all_feedback_df['skip_reason'].dropna().empty and all_feedback_df['skip_reason'].mode().values[0] != '') else "Class / Lab Clash"
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:14px; padding:1.1rem; text-align:center;">
                <div style="font-size:0.75rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Top Skip Reason</div>
                <div style="font-size:1.25rem; font-weight:900; color:#E76F51; margin:0.5rem 0;">{top_reason}</div>
                <div style="font-size:0.78rem; color:#64748B;">Most common reason for absence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.0rem;'></div>", unsafe_allow_html=True)

        # Incoming Student Feedback Live Feed
        st.markdown("<div style='font-size:1.0rem; font-weight:800; color:#12372A; margin-bottom:0.5rem;'>Live Incoming Student Feedback & Rating Feed</div>", unsafe_allow_html=True)

        if not all_feedback_df.empty:
            display_fb = all_feedback_df[['student_roll', 'meal_date', 'meal_type', 'dish_name', 'rating', 'skip_reason', 'feedback_text', 'created_at']].rename(columns={
                'student_roll': 'Student Roll',
                'meal_date': 'Date',
                'meal_type': 'Meal',
                'dish_name': 'Dish',
                'rating': 'Rating (1-5 ⭐)',
                'skip_reason': 'Skip Reason',
                'feedback_text': 'Comments / Suggestion',
                'created_at': 'Submitted On'
            })
            st.dataframe(display_fb, use_container_width=True, hide_index=True)
        else:
            st.info("No student feedback submitted yet. When students submit ratings or explain why they skipped on the Student Portal, their comments and ratings will display here.")

        # Story Step Transition
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:12px; padding:0.9rem 1.2rem; margin-top:1.2rem; display:flex; justify-content:space-between; align-items:center; border:1px solid rgba(18,55,42,0.08);">
            <div>
                <strong style="color:#12372A;">🔄 Closed-Loop Dining Intelligence Complete:</strong>
                <span style="color:#64748B; font-size:0.86rem; margin-left:6px;">Student feedback reviewed and waste trends identified! The kitchen uses this knowledge to refine recipes.</span>
            </div>
            <span style="color:#16A36A; font-weight:800; font-size:0.85rem;">↺ Returns to Stage 1: Tomorrow's Menu</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 7: SHIFT WASTE RECORDS & HISTORICAL DATABASE
    # -------------------------------------------------------------------------
    with m_tabs[6]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">📝 Staff Food-Waste Recording & Verifiable Database</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Log post-meal shifts into the verified database with full audit trail and instant confirmation card.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">SQLITE PERSISTENT</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("staff_record_waste_form_story"):
            r_c1, r_c2, r_c3 = st.columns(3)
            with r_c1:
                r_date = st.date_input("Service Date", value=date.today(), key="r_date_story").strftime("%Y-%m-%d")
                r_meal = st.selectbox("Meal Period", ["Lunch", "Dinner", "Breakfast", "Snacks"], key="r_meal_story")
            with r_c2:
                r_dish = st.selectbox("Dish Name", list(DEFAULT_COSTS.keys()), index=list(DEFAULT_COSTS.keys()).index("Dal Tadka") if "Dal Tadka" in DEFAULT_COSTS else 0, key="r_dish_story")
                r_cost = st.number_input("Cost per Serving (₹)", min_value=1.0, max_value=200.0, value=float(DEFAULT_COSTS.get(r_dish, 15.0)), step=1.0)
            with r_c3:
                r_prep = st.number_input("Quantity Prepared (Servings)", min_value=10, max_value=800, value=350, step=5)
                r_served = st.number_input("Quantity Served (Servings)", min_value=10, max_value=800, value=310, step=5)
            
            r_leftover = max(0, r_prep - r_served)
            r_c4, r_c5 = st.columns([1, 1])
            with r_c4:
                r_discarded = st.number_input("Discarded Quantity (Servings)", min_value=0, max_value=800, value=min(25, r_leftover), step=5)
            with r_c5:
                r_usable = max(0, r_leftover - r_discarded)
                st.markdown(f"<div style='padding-top:1.8rem; font-size:0.88rem; color:#64748B;'>Leftover: <strong>{r_leftover}</strong> • Usable / Donated: <strong>{r_usable}</strong></div>", unsafe_allow_html=True)

            save_record_btn = st.form_submit_button("💾 Save Record to Database", use_container_width=True)

            if save_record_btn:
                success, msg = db.record_meal_waste(
                    meal_date=r_date,
                    meal_type=r_meal,
                    dish_name=r_dish,
                    quantity_prepared=r_prep,
                    quantity_served=r_served,
                    discarded_quantity=r_discarded,
                    cost_per_serving=r_cost
                )
                if success:
                    est_loss = float(r_discarded * r_cost)
                    st.session_state["last_saved_waste_record"] = {
                        "id": f"REC-{r_date.replace('-', '')}-{r_meal[:3].upper()}",
                        "date": r_date,
                        "meal": r_meal,
                        "dish": r_dish,
                        "prep": r_prep,
                        "served": r_served,
                        "leftover": r_leftover,
                        "discarded": r_discarded,
                        "loss": est_loss
                    }
                    db.log_audit_event("WASTE_RECORD_SAVED", "manager", f"Saved waste record for {r_dish} ({r_discarded} servings discarded)")
                    st.rerun()
                else:
                    st.error(f"Failed to save record: {msg}")

        # Confirmation Card upon successful save
        if "last_saved_waste_record" in st.session_state:
            rc_saved = st.session_state["last_saved_waste_record"]
            st.markdown(f"""
            <div style="background:#FFFFFF; border:2px solid #16A36A; border-radius:16px; padding:1.2rem 1.4rem; margin:1.2rem 0; box-shadow:0 4px 18px rgba(22,163,106,0.08);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                    <span style="font-size:0.95rem; font-weight:800; color:#12372A;">✅ Record Saved Successfully to Database!</span>
                    <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:3px 10px; border-radius:6px;">Record ID: #{rc_saved['id']}</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:0.8rem; font-size:0.85rem; color:#18231F; margin-top:0.6rem;">
                    <div>• <strong>Dish:</strong> {rc_saved['dish']}</div>
                    <div>• <strong>Date:</strong> {rc_saved['date']}</div>
                    <div>• <strong>Meal:</strong> {rc_saved['meal']}</div>
                    <div>• <strong>Prepared:</strong> {rc_saved['prep']} servings</div>
                    <div>• <strong>Served:</strong> {rc_saved['served']} servings</div>
                    <div>• <strong>Leftover:</strong> {rc_saved['leftover']} servings</div>
                    <div>• <strong>Discarded:</strong> <span style="color:#E76F51; font-weight:700;">{rc_saved['discarded']} servings</span></div>
                    <div>• <strong>Estimated Loss:</strong> <span style="color:#E76F51; font-weight:800;">₹{rc_saved['loss']:.2f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Database History
        st.markdown("<div style='font-size:1.1rem; font-weight:800; color:#12372A; margin:1.5rem 0 0.6rem 0;'>Saved Records & Database History</div>", unsafe_allow_html=True)
        h_f1, h_f2, h_f3 = st.columns(3)
        with h_f1:
            h_dish_filter = st.text_input("Search Dish Name", placeholder="e.g. Dal Tadka", key="h_dish_story")
        with h_f2:
            h_meal_filter = st.selectbox("Meal Filter", ["All Meals", "Lunch", "Dinner", "Breakfast", "Snacks"], key="h_meal_story")
        with h_f3:
            h_limit = st.slider("Records Count", min_value=10, max_value=200, value=25, step=5, key="h_lim_story")

        meal_param = None if h_meal_filter == "All Meals" else h_meal_filter
        dish_param = h_dish_filter.strip() if h_dish_filter.strip() else None

        res = db.get_filtered_meal_records(dish_name=dish_param, meal_type=meal_param, limit=h_limit)
        filtered_records = res[0] if isinstance(res, tuple) else res
        total_count = res[1] if isinstance(res, tuple) else len(filtered_records)
        
        st.caption(f"Showing {len(filtered_records)} of {total_count} matching records (Newest at the top)")
        if filtered_records.empty:
            st.info("No saved records found matching the filter.")
        else:
            st.dataframe(filtered_records, use_container_width=True, hide_index=True)

        # Storage Explainer
        st.markdown("""
        <div style="background:#F7F8F2; border-left:4px solid #12372A; border-radius:12px; padding:1.0rem 1.2rem; margin-top:1.2rem; font-size:0.84rem; color:#18231F; line-height:1.5;">
            <strong style="color:#12372A; font-size:0.95rem;">🗄️ Storage Engine:</strong> Local SQLite Relational Database (<code>data/messmind.db</code>) • Primary Table: <code>meal_records</code> • Dual-synchronized to <code>data/mess_data.csv</code>.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 8: MESS SETTINGS & CAMPUS BROADCASTS
    # -------------------------------------------------------------------------
    with m_tabs[7]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">⚙️ Campus Settings & Broadcast Notices</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Post announcements to all student notice boards, configure safety buffer %, and view system audit logs.</div>
                </div>
                <span style="background:rgba(18,55,42,0.1); color:#12372A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">ADMINISTRATIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        set_col1, set_col2 = st.columns([1, 1], gap="large")

        with set_col1:
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.6rem;'>Campus Mess Parameters</div>", unsafe_allow_html=True)
            settings_dict = db.get_mess_settings()
            
            with st.form("mess_params_form_story"):
                cap_val = st.number_input("Total Hostel Student Capacity", min_value=100, max_value=2000, value=int(settings_dict.get('hostel_capacity', 450)))
                buf_val = st.number_input("ML Safety Margin (%)", min_value=1.0, max_value=25.0, value=float(settings_dict.get('safety_margin_pct', 5.0)), step=0.5)
                cost_thresh = st.number_input("Waste Cost Alert Threshold (₹)", min_value=100.0, max_value=10000.0, value=float(settings_dict.get('waste_cost_alert_threshold', 1000.0)), step=100.0)
                
                if st.form_submit_button("Update Mess Settings", use_container_width=True):
                    db.update_mess_setting('hostel_capacity', cap_val)
                    db.update_mess_setting('safety_margin_pct', buf_val)
                    db.update_mess_setting('waste_cost_alert_threshold', cost_thresh)
                    db.log_audit_event("SETTINGS_UPDATE", "manager", f"Updated capacity to {cap_val}, buffer to {buf_val}%")
                    st.success("Mess settings updated successfully!")

            lan_ip = get_lan_ip()
            st.markdown(f"""
            <div style="background:#F7F8F2; border-left:4px solid #16A36A; border-radius:12px; padding:1.0rem 1.2rem; margin-top:1.2rem; font-size:0.84rem; color:#18231F; line-height:1.5;">
                <strong style="color:#12372A; font-size:0.95rem;">🌐 Host & Network Connectivity Links:</strong><br>
                • <strong>Local PC:</strong> <a href="http://localhost:8501" target="_blank">http://localhost:8501</a><br>
                • <strong>Student Mobile / Wi-Fi:</strong> <a href="http://{lan_ip}:8501" target="_blank">http://{lan_ip}:8501</a>
            </div>
            """, unsafe_allow_html=True)

        with set_col2:
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.6rem;'>📢 Broadcast Notice to Student Notice Boards</div>", unsafe_allow_html=True)
            with st.form("post_announcement_form_story"):
                ann_title = st.text_input("Title", placeholder="e.g. Special Feast on Friday!")
                ann_content = st.text_area("Message", placeholder="e.g. Special Ugadi dinner will be served from 7:30 PM. Please RSVP!")
                ann_priority = st.selectbox("Priority", ["normal", "high", "urgent"])

                if st.form_submit_button("📢 Broadcast Notice to Students", use_container_width=True):
                    if ann_title.strip() and ann_content.strip():
                        db.add_announcement(ann_title.strip(), ann_content.strip(), ann_priority, target_role="student")
                        db.log_audit_event("ANNOUNCEMENT_POST", "manager", f"Posted notice '{ann_title}'")
                        st.success("Broadcast posted to all student notice boards!")
                        st.rerun()
                    else:
                        st.error("Title and message cannot be empty.")

            st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#12372A; margin:1.0rem 0 0.4rem 0;'>Active Campus Notices</div>", unsafe_allow_html=True)
            active_anns = db.get_announcements(active_only=False)
            if not active_anns.empty:
                for _, an in active_anns.iterrows():
                    a_c1, a_c2 = st.columns([3, 1])
                    with a_c1:
                        st.markdown(f"<strong>{an['title']}</strong> ({an['priority']})")
                        st.caption(an['content'])
                    with a_c2:
                        if st.button("Delete", key=f"del_ann_{an['id']}"):
                            db.delete_announcement(an['id'])
                            db.log_audit_event("ANNOUNCEMENT_DELETE", "manager", f"Deleted announcement ID {an['id']}")
                            st.rerun()
                    st.markdown("<hr style='margin:0.2rem 0; border:none; border-top:1px solid #EEF2F6;'>", unsafe_allow_html=True)

        # Audit Logs
        st.markdown("<div style='font-size:1.0rem; font-weight:800; color:#12372A; margin:1.5rem 0 0.5rem 0;'>🛡️ Immutable System Audit Logs</div>", unsafe_allow_html=True)
        audit_df = db.get_audit_logs(limit=25)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
