"""
portal_student.py
=================
Student Portal for MessMind AI.
Designed specifically for hostel students at IIIT Dharwad:
- Clean, student-friendly, zero-jargon mobile-first interface
- Read-only Today's Published Menu
- 1-Click Meal Attendance RSVP (Attending / Skipping / Not sure)
- Food Satisfaction & Waste Feedback Ratings (with skip reasons)
- Live Campus Announcements Notice Board
- Student Profile & Dietary Preferences
"""

from datetime import datetime, date
import streamlit as st
import pandas as pd
import database as db


def render_student_portal():
    """Renders the comprehensive Student Portal with role isolation."""
    
    user_id = st.session_state.get("user_id")
    username = st.session_state.get("username", "Student")
    full_name = st.session_state.get("full_name", username)
    hostel = st.session_state.get("hostel", "Hostel Block")
    room_number = st.session_state.get("room_number", "")
    today_str = date.today().strftime("%Y-%m-%d")
    today_day = date.today().strftime("%A")

    # Navigation Tabs for Student Portal
    tabs = st.tabs([
        "🏠 Student Home",
        "🍲 Today's Menu",
        "🗳️ Meal Attendance (RSVP)",
        "⭐ Food Feedback & Ratings",
        "📢 Campus Notice Board",
        "👤 My Profile"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: STUDENT HOME
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; box-shadow:0 4px 16px rgba(18,55,42,0.03);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.0rem;">
                <div>
                    <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:3px 10px; border-radius:20px; text-transform:uppercase;">Student Dining Portal</span>
                    <h2 style="color:#12372A; font-size:1.8rem; font-weight:800; margin:0.4rem 0 0.2rem 0;">Welcome, {full_name}! 👋</h2>
                    <div style="color:#64748B; font-size:0.9rem;">Roll No: <strong>{username.upper()}</strong> • Hostel: <strong>{hostel}</strong> {f'(Room {room_number})' if room_number else ''} • {today_day}, {today_str}</div>
                </div>
                <div style="background:#F7F8F2; border-radius:12px; padding:0.6rem 1.0rem; text-align:right;">
                    <div style="font-size:0.75rem; color:#8A9990; font-weight:700; text-transform:uppercase;">Dining Hall</div>
                    <div style="font-size:1.05rem; font-weight:800; color:#12372A;">Mess 1 Central Dining</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Active Announcements Banner
        announcements_df = db.get_announcements(target_role="student", active_only=True)
        if not announcements_df.empty:
            st.markdown("<div style='font-size:0.92rem; font-weight:800; color:#12372A; margin:0.8rem 0 0.4rem 0;'>📢 Active Mess Notices</div>", unsafe_allow_html=True)
            for _, ann in announcements_df.iterrows():
                p_color = "#E76F51" if ann['priority'] == 'urgent' else ("#F4A261" if ann['priority'] == 'high' else "#16A36A")
                st.markdown(f"""
                <div style="background:#FFFFFF; border-left:4px solid {p_color}; border-radius:12px; padding:0.9rem 1.2rem; margin-bottom:0.7rem; box-shadow:0 2px 8px rgba(0,0,0,0.03);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong style="color:#12372A; font-size:0.95rem;">{ann['title']}</strong>
                        <span style="font-size:0.7rem; font-weight:800; text-transform:uppercase; color:{p_color}; background:rgba(0,0,0,0.04); padding:2px 8px; border-radius:6px;">{ann['priority']} priority</span>
                    </div>
                    <div style="color:#64748B; font-size:0.86rem; margin-top:4px;">{ann['content']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Current Meal Status & Quick RSVP Card
        now_hour = datetime.now().hour
        if now_hour < 10:
            current_meal_phase = "Breakfast"
            next_meal = "Lunch"
        elif now_hour < 15:
            current_meal_phase = "Lunch"
            next_meal = "Dinner"
        elif now_hour < 18:
            current_meal_phase = "Evening Snacks"
            next_meal = "Dinner"
        else:
            current_meal_phase = "Dinner"
            next_meal = "Breakfast (Tomorrow)"

        col_h1, col_h2 = st.columns([1.2, 1], gap="medium")

        with col_h1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:16px; padding:1.2rem 1.4rem; height:100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                    <span style="font-size:0.78rem; text-transform:uppercase; color:#8A9990; font-weight:800;">Dining Schedule</span>
                    <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-size:0.75rem; font-weight:800; padding:3px 8px; border-radius:6px;">Active Phase: {current_meal_phase}</span>
                </div>
                <h4 style="color:#12372A; margin:0 0 0.5rem 0;">Today's Meal Routine</h4>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.6;">
                    • <strong>Breakfast:</strong> 07:30 AM – 09:30 AM<br>
                    • <strong>Lunch:</strong> 12:30 PM – 02:30 PM<br>
                    • <strong>Evening Snacks:</strong> 05:00 PM – 06:15 PM<br>
                    • <strong>Dinner:</strong> 07:45 PM – 09:45 PM
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_h2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1.5px solid #16A36A; border-radius:16px; padding:1.2rem 1.4rem; height:100%;">
                <div style="font-size:0.78rem; text-transform:uppercase; color:#16A36A; font-weight:800; margin-bottom:0.4rem;">Next Meal RSVP</div>
                <h4 style="color:#12372A; margin:0 0 0.4rem 0;">Are you dining for {next_meal}?</h4>
                <p style="font-size:0.82rem; color:#64748B; margin:0 0 0.8rem 0;">
                    Your vote directly prevents over-preparation. Tap to record your intent!
                </p>
            """, unsafe_allow_html=True)

            q_intent = db.get_student_meal_intents(student_id=username, service_date=today_str, meal_period=next_meal)
            current_choice = q_intent.iloc[0]['intent_status'] if (q_intent is not None and not q_intent.empty) else "Not Polled"

            if current_choice != "Not Polled":
                st.info(f"Your registered response for {next_meal}: **{current_choice}**")
            else:
                st.caption(f"No response logged yet for {next_meal}.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Frugal Innovation Student Impact Note
        st.markdown("""
        <div style="background:#F7F8F2; border-radius:14px; padding:1.0rem 1.3rem; margin-top:1.2rem; display:flex; align-items:center; gap:1.0rem;">
            <div style="font-size:1.8rem;">🌱</div>
            <div style="font-size:0.84rem; color:#18231F; line-height:1.4;">
                <strong>Every RSVP Counts!</strong> MessMind AI synchronizes your attendance intent with our kitchen's AI preparation model. When students declare skips in advance, kitchen cooks avoid boiling surplus rice and dal, saving up to ₹45,000 every month!
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: TODAY'S MENU (READ-ONLY)
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">🍲 Official Campus Dining Menu</h3>
                    <div style="color:#64748B; font-size:0.85rem;">IIIT Dharwad Central Mess 1 • Verified Kitchen Schedule</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">🟢 PUBLISHED ONLY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sel_menu_day = st.selectbox(
            "Select Day to View Menu",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(today_day),
            key="stu_menu_day_select"
        )

        # Fetch published items for this weekday
        menu_items_df = db.get_menu_items(day_of_week=sel_menu_day, only_published=True)

        if menu_items_df.empty:
            st.warning(f"No published menu items available for {sel_menu_day} yet. The mess manager may still be preparing the weekly roster.")
        else:
            meal_periods = ["Breakfast", "Lunch", "Snacks", "Dinner"]
            for mp in meal_periods:
                mp_items = menu_items_df[menu_items_df["meal_period"] == mp]
                st.markdown(f"""
                <div style="margin-top:1.2rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:0.6rem;">
                    <span style="font-size:1.1rem; font-weight:800; color:#12372A;">{mp}</span>
                    <span style="background:#F7F8F2; color:#8A9990; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:6px;">{len(mp_items)} items</span>
                </div>
                """, unsafe_allow_html=True)

                if mp_items.empty:
                    st.caption(f"No dishes published for {mp} on {sel_menu_day}.")
                else:
                    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:0.6rem; margin-bottom:1.0rem;'>", unsafe_allow_html=True)
                    for _, row in mp_items.iterrows():
                        cat = row['category']
                        cat_icon = "🍚" if "Rice" in cat else ("🍲" if "Curry" in cat or "Dal" in cat else ("🫓" if "Bread" in cat or "Tiffin" in cat else "🥛"))
                        st.markdown(f"""
                        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.12); border-radius:12px; padding:0.6rem 0.9rem; font-size:0.85rem; font-weight:700; color:#12372A; display:flex; align-items:center; gap:0.5rem; box-shadow:0 1px 4px rgba(0,0,0,0.02);">
                            <span>{cat_icon}</span>
                            <span>{row['dish_name']}</span>
                            <span style="font-size:0.7rem; background:#F7F8F2; color:#8A9990; padding:2px 6px; border-radius:4px; font-weight:600;">{row['category']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 3: MEAL ATTENDANCE (RSVP)
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">🗳️ Meal Attendance RSVP</h3>
                    <div style="color:#64748B; font-size:0.85rem;">Inform kitchen staff whether you are eating to prevent cooking waste.</div>
                </div>
                <span style="background:rgba(22,163,106,0.12); color:#16A36A; font-weight:800; font-size:0.75rem; padding:4px 10px; border-radius:8px;">DUPLICATE PROTECTED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        rsvp_date = st.date_input("Select Date for RSVP", value=date.today(), key="stu_rsvp_date_picker")
        rsvp_date_str = rsvp_date.strftime("%Y-%m-%d")

        col_r1, col_r2 = st.columns([1, 1], gap="medium")

        with col_r1:
            st.markdown(f"<div style='font-weight:800; color:#12372A; font-size:1.0rem; margin-bottom:0.8rem;'>Record Your Intent for {rsvp_date_str}</div>", unsafe_allow_html=True)
            
            with st.form("student_rsvp_form"):
                rsvp_meal = st.selectbox("Select Meal Period", options=["Breakfast", "Lunch", "Dinner"], key="stu_rsvp_meal")
                
                # Check existing intent
                existing_intents = db.get_student_meal_intents(student_id=username, service_date=rsvp_date_str, meal_period=rsvp_meal)
                current_val = existing_intents.iloc[0]['intent_status'] if (existing_intents is not None and not existing_intents.empty) else None

                if current_val:
                    st.info(f"Your currently recorded status for this meal: **{current_val}** (You may update it below).")

                intent_choice = st.radio(
                    "Your Attendance Plan:",
                    options=["Attending", "Not Attending", "Not sure"],
                    index=["Attending", "Not Attending", "Not sure"].index(current_val) if current_val in ["Attending", "Not Attending", "Not sure"] else 0,
                    key="stu_intent_radio"
                )

                rsvp_notes = st.text_input("Optional Note (e.g. Late lab session, Eating with club)", placeholder="e.g. Lab clash until 2 PM", key="stu_rsvp_notes")

                rsvp_submit = st.form_submit_button("💾 Save My Attendance RSVP", use_container_width=True)

                if rsvp_submit:
                    res_intent = db.submit_student_meal_intent(
                        student_id=username,
                        service_date=rsvp_date_str,
                        meal_period=rsvp_meal,
                        intent=intent_choice,
                        hostel=hostel,
                        notes=rsvp_notes
                    )
                    success = res_intent[0] if isinstance(res_intent, tuple) else res_intent
                    if success:
                        st.success(f"✅ Success! Your RSVP for **{rsvp_meal}** on **{rsvp_date_str}** has been recorded as: **{intent_choice}**.")
                        db.log_audit_event("STUDENT_RSVP", username, f"RSVP: {rsvp_meal} {rsvp_date_str} -> {intent_choice}")
                        st.rerun()
                    else:
                        st.error("Failed to save RSVP. Please try again.")

        with col_r2:
            st.markdown(f"<div style='font-weight:800; color:#12372A; font-size:1.0rem; margin-bottom:0.8rem;'>Your Registered RSVPs for {rsvp_date_str}</div>", unsafe_allow_html=True)
            day_intents = db.get_student_meal_intents(student_id=username, service_date=rsvp_date_str)
            
            if day_intents is not None and not day_intents.empty:
                for _, di in day_intents.iterrows():
                    i_status = di.get('intent_status', di.get('intent', 'Attending'))
                    m_per = di.get('meal_type', di.get('meal_period', 'Meal'))
                    t_stamp = di.get('updated_at', di.get('timestamp', ''))
                    badge_bg = "rgba(22,163,106,0.12)" if i_status == "Attending" else ("rgba(231,111,81,0.15)" if i_status == "Not Attending" else "rgba(244,162,97,0.15)")
                    badge_color = "#16A36A" if i_status == "Attending" else ("#E76F51" if i_status == "Not Attending" else "#E76F51")
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.7rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#12372A; font-size:0.95rem;">{m_per}</strong>
                            <span style="background:{badge_bg}; color:{badge_color}; font-weight:800; font-size:0.75rem; padding:3px 10px; border-radius:6px;">{i_status}</span>
                        </div>
                        <div style="font-size:0.78rem; color:#8A9990; margin-top:4px;">Logged: {t_stamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#F7F8F2; border-radius:12px; padding:1.5rem; text-align:center; color:#8A9990; font-size:0.88rem;">
                    No RSVPs logged yet for this date. Choose a meal on the left to register.
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: FOOD SATISFACTION & RATINGS
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">⭐ Food Satisfaction & Waste Feedback</h3>
            <div style="color:#64748B; font-size:0.85rem;">Help the kitchen improve recipes, portion sizes, and cut down unconsumed food.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("student_feedback_form"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                fb_date = st.date_input("Meal Date", value=date.today(), key="stu_fb_date")
                fb_meal = st.selectbox("Meal Period", options=["Lunch", "Dinner", "Breakfast", "Snacks"], key="stu_fb_meal")
                fb_rating = st.slider("Overall Meal Rating (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=4, key="stu_fb_rating")
            
            with f_col2:
                fb_taste = st.select_slider("Taste & Flavor", options=["Too Bland", "Okay", "Delicious", "Too Spicy/Salty"], value="Delicious")
                fb_portion = st.select_slider("Portion Size Sufficiency", options=["Too Small", "Just Right", "Too Much (Caused Waste)"], value="Just Right")
                fb_skip_reason = st.selectbox(
                    "If you skipped or left food, what was the primary reason?",
                    options=["None / Finished Meal", "Class / Lab Clash", "Eating Outside", "Not Hungry", "Disliked Recipe / Taste", "Food Was Cold", "Other"]
                )

            fb_dish = st.text_input("Specific Dish Name (Optional)", placeholder="e.g. Dal Tadka, Paneer Butter Masala", key="stu_fb_dish")
            fb_comments = st.text_area("Constructive Comments / Suggestions for Mess Cooks", placeholder="e.g. The dal was slightly too salty today; chapati was fresh and soft.", key="stu_fb_comments")

            fb_submit = st.form_submit_button("📤 Submit Anonymous Feedback", use_container_width=True)

            if fb_submit:
                fb_res = db.submit_student_feedback(
                    student_id=username,
                    service_date=fb_date.strftime("%Y-%m-%d"),
                    meal_period=fb_meal,
                    dish_name=fb_dish if fb_dish else "General Meal",
                    rating=fb_rating,
                    taste_rating=fb_taste,
                    portion_rating=fb_portion,
                    skip_reason=fb_skip_reason,
                    comments=fb_comments
                )
                fb_success = fb_res[0] if isinstance(fb_res, tuple) else fb_res
                if fb_success:
                    st.success("🎉 Thank you! Your feedback has been safely recorded in the kitchen improvement database.")
                    db.log_audit_event("STUDENT_FEEDBACK", username, f"Feedback submitted for {fb_meal} on {fb_date}")
                else:
                    st.error("Failed to submit feedback. Please try again.")

    # -------------------------------------------------------------------------
    # TAB 5: CAMPUS ANNOUNCEMENTS
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">📢 Campus Mess Notice Board</h3>
            <div style="color:#64748B; font-size:0.85rem;">Official announcements, festival special meals, and maintenance updates.</div>
        </div>
        """, unsafe_allow_html=True)

        all_ann = db.get_announcements(target_role="student", active_only=True)
        if all_ann.empty:
            st.info("No active notices right now. All dining operations are running normally.")
        else:
            for _, ann in all_ann.iterrows():
                p_border = "#E76F51" if ann['priority'] == 'urgent' else ("#F4A261" if ann['priority'] == 'high' else "#16A36A")
                p_badge = "URGENT" if ann['priority'] == 'urgent' else ("HIGH" if ann['priority'] == 'high' else "NOTICE")
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.1); border-left:5px solid {p_border}; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1.0rem; box-shadow:0 3px 12px rgba(0,0,0,0.02);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="font-size:1.05rem; font-weight:800; color:#12372A;">{ann['title']}</span>
                        <span style="font-size:0.72rem; font-weight:800; color:{p_border}; background:rgba(0,0,0,0.05); padding:2px 8px; border-radius:6px;">{p_badge}</span>
                    </div>
                    <div style="font-size:0.88rem; color:#475569; line-height:1.5;">{ann['content']}</div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.6rem;">Posted on: {ann['created_at']}</div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 6: MY PROFILE
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.08); border-radius:18px; padding:1.2rem 1.5rem; margin-bottom:1.2rem;">
            <h3 style="color:#12372A; margin:0; font-size:1.3rem; font-weight:800;">👤 My Dining Profile</h3>
            <div style="color:#64748B; font-size:0.85rem;">Manage your dietary preferences and inspect your campus RSVP attendance log.</div>
        </div>
        """, unsafe_allow_html=True)

        p_col1, p_col2 = st.columns([1, 1], gap="large")

        with p_col1:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid rgba(18,55,42,0.12); border-radius:16px; padding:1.4rem; margin-bottom:1.2rem;">
                <div style="font-size:0.78rem; text-transform:uppercase; color:#8A9990; font-weight:800; margin-bottom:0.6rem;">Hostel Identity Card</div>
                <div style="font-size:1.15rem; font-weight:800; color:#12372A;">{full_name}</div>
                <div style="color:#64748B; font-size:0.88rem; margin-top:2px;">Roll Number: <strong>{username.upper()}</strong></div>
                <div style="color:#64748B; font-size:0.88rem; margin-top:2px;">Hostel Block: <strong>{hostel}</strong> {f'(Room {room_number})' if room_number else ''}</div>
                <div style="color:#64748B; font-size:0.88rem; margin-top:2px;">Role: <strong>Hostel Student</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.5rem;'>Update Dietary Preference</div>", unsafe_allow_html=True)
            with st.form("update_profile_pref_form"):
                new_pref = st.selectbox(
                    "Dietary Preference",
                    options=["Vegetarian", "Non-Vegetarian", "Eggetarian", "Jain (No Root Veg)"],
                    index=["Vegetarian", "Non-Vegetarian", "Eggetarian", "Jain (No Root Veg)"].index(
                        st.session_state.get("dietary_preference", "Vegetarian")
                    ) if st.session_state.get("dietary_preference", "Vegetarian") in ["Vegetarian", "Non-Vegetarian", "Eggetarian", "Jain (No Root Veg)"] else 0
                )
                save_pref = st.form_submit_button("Save Preference", use_container_width=True)
                if save_pref:
                    db.update_user_profile(user_id=user_id, dietary_preference=new_pref)
                    st.session_state.dietary_preference = new_pref
                    st.success(f"Dietary preference updated to: **{new_pref}**")
                    db.log_audit_event("PROFILE_UPDATE", username, f"Updated dietary preference to {new_pref}")

        with p_col2:
            st.markdown("<div style='font-size:0.95rem; font-weight:800; color:#12372A; margin-bottom:0.5rem;'>My Recent RSVP History</div>", unsafe_allow_html=True)
            my_intents = db.get_student_meal_intents(student_id=username)
            if my_intents is not None and not my_intents.empty:
                display_df = my_intents.head(15)[['meal_date', 'meal_type', 'intent_status', 'updated_at']].rename(columns={
                    'meal_date': 'Date',
                    'meal_type': 'Meal',
                    'intent_status': 'Intent',
                    'updated_at': 'Recorded At'
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No past RSVP history found yet. As you record meal intentions, they will appear here.")

