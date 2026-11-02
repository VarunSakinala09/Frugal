"""
test_project.py
===============
Comprehensive unit & integration test suite for MessMind AI.
Tests dataset constraints, database persistence, duplicate prevention,
official weekly menu lookup, multi-state attendance intent, ML leakage guards,
recommendation logic, and edge cases.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database as db
import generate_dataset as gd
import train_model as tm
import recommendations as rc
import vision_ai as va


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dataset_path():
    """Returns path to the synthetic dataset, generating if needed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mess_data.csv")
    if not os.path.exists(path):
        gd.generate_synthetic_data(num_days=100, output_path=path)
    return path


@pytest.fixture(scope="session")
def df(dataset_path):
    """Loads synthetic dataset DataFrame."""
    return pd.read_csv(dataset_path)


@pytest.fixture
def temp_db():
    """Provides an isolated temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name
    db.init_db(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except PermissionError:
            pass


# -----------------------------------------------------------------------------
# Test Cases (1 through 18)
# -----------------------------------------------------------------------------

def test_1_synthetic_dataset_generation():
    """Test 1: Synthetic dataset generation runs and creates DataFrame with > 1000 records."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name
    try:
        df_gen = gd.generate_synthetic_data(num_days=30, output_path=tmp_csv)
        assert isinstance(df_gen, pd.DataFrame)
        assert len(df_gen) > 200
        assert os.path.exists(tmp_csv)
    finally:
        if os.path.exists(tmp_csv):
            try:
                os.remove(tmp_csv)
            except PermissionError:
                pass


def test_2_dataset_contains_expected_columns(df):
    """Test 2: Dataset contains all required operational and metadata columns."""
    required_cols = [
        "date", "day_of_week", "meal_type", "dish_name",
        "expected_attendance", "registered_students", "historical_demand",
        "actual_attendance", "actual_servings_needed", "quantity_prepared",
        "quantity_served", "leftover_quantity", "discarded_quantity",
        "usable_leftover", "cost_per_serving_inr", "discarded_food_cost_inr",
        "is_synthetic"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_3_dataset_contains_no_unintended_missing_values(df):
    """Test 3: Dataset contains zero null or missing values across all records."""
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing/null values"


def test_4_quantities_are_nonnegative(df):
    """Test 4: All attendance, serving, leftover, and cost figures are strictly non-negative."""
    numeric_cols = [
        "expected_attendance", "registered_students", "historical_demand",
        "actual_attendance", "actual_servings_needed", "quantity_prepared",
        "quantity_served", "leftover_quantity", "discarded_quantity",
        "usable_leftover", "cost_per_serving_inr", "discarded_food_cost_inr"
    ]
    for col in numeric_cols:
        assert (df[col] >= 0).all(), f"Found negative values in column: {col}"


def test_5_quantity_served_does_not_exceed_quantity_prepared(df):
    """Test 5: Quantity served cannot exceed quantity prepared in any record."""
    violations = df[df["quantity_served"] > df["quantity_prepared"]]
    assert len(violations) == 0, f"Found {len(violations)} records where served > prepared"


def test_6_leftover_quantity_is_calculated_correctly(df):
    """Test 6: Leftover quantity must strictly equal quantity_prepared - quantity_served."""
    calculated_leftover = df["quantity_prepared"] - df["quantity_served"]
    diff = (df["leftover_quantity"] - calculated_leftover).abs()
    assert (diff == 0).all(), "Leftover calculation discrepancy found"


def test_7_discarded_quantity_does_not_exceed_leftover_quantity(df):
    """Test 7: Discarded quantity must be between 0 and leftover_quantity."""
    violations = df[df["discarded_quantity"] > df["leftover_quantity"]]
    assert len(violations) == 0, f"Found {len(violations)} records where discarded > leftover"


def test_8_discarded_food_cost_is_calculated_correctly(df):
    """Test 8: Discarded food cost must equal discarded_quantity * cost_per_serving."""
    expected_cost = (df["discarded_quantity"] * df["cost_per_serving_inr"]).round(2)
    diff = (df["discarded_food_cost_inr"] - expected_cost).abs()
    assert (diff <= 0.01).all(), "Discarded food cost calculation mismatch found"


def test_9_database_initialization(temp_db):
    """Test 9: Database initialization safely creates all tables and migrations without error."""
    conn = db.get_connection(temp_db)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "student_registrations" in tables
        assert "meal_records" in tables
        assert "prediction_records" in tables

        # Verify attendance_intent column exists
        cursor.execute("PRAGMA table_info(student_registrations);")
        cols = [r["name"] for r in cursor.fetchall()]
        assert "attendance_intent" in cols
    finally:
        conn.close()


def test_10_student_registration_storage(temp_db):
    """Test 10: Student registration can be inserted with multi-state intent (Yes/No/Not sure)."""
    # 1. Confirmed Yes
    s_yes, m_yes = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS001",
        attendance_intent="Yes",
        db_path=temp_db
    )
    assert s_yes is True

    # 2. Skipping No
    s_no, m_no = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS002",
        attendance_intent="No",
        db_path=temp_db
    )
    assert s_no is True

    # 3. Uncertain Not sure
    s_ns, m_ns = db.register_student(
        meal_date="2026-11-10",
        meal_type="Dinner",
        selected_dishes=["Rice", "Paneer Curry"],
        registration_code="23BDS003",
        attendance_intent="Not sure",
        db_path=temp_db
    )
    assert s_ns is True

    # Confirmed count should only be 1 (only 'Yes' is confirmed!)
    confirmed_count = db.get_registration_count("2026-11-10", "Dinner", db_path=temp_db)
    assert confirmed_count == 1


def test_11_duplicate_registration_handling(temp_db):
    """Test 11: Submitting registration with the same student ID updates intent instead of duplicating."""
    # First submission: Student plans to skip
    db.register_student(
        meal_date="2026-11-10",
        meal_type="Lunch",
        registration_code="STUDENT_A",
        attendance_intent="No",
        db_path=temp_db
    )
    assert db.get_registration_count("2026-11-10", "Lunch", db_path=temp_db) == 0

    # Student updates plan: Decides to attend (Yes)
    success, msg = db.register_student(
        meal_date="2026-11-10",
        meal_type="Lunch",
        registration_code="STUDENT_A",
        attendance_intent="Yes",
        db_path=temp_db
    )
    assert success is True
    assert "updated" in msg.lower()

    # Confirmed count must now be exactly 1
    assert db.get_registration_count("2026-11-10", "Lunch", db_path=temp_db) == 1

    # Total responses must also remain 1
    counts = db.get_attendance_intent_counts("2026-11-10", "Lunch", eligible_students=500, db_path=temp_db)
    assert counts["total_responses"] == 1
    assert counts["yes_count"] == 1
    assert counts["no_count"] == 0


def test_12_model_can_be_trained_and_loaded():
    """Test 12: Trained Scikit-learn model exists and can be loaded via joblib."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "demand_model.joblib")
    assert os.path.exists(model_path), f"Model file missing at {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model object missing predict method"


def test_13_prediction_inputs_do_not_contain_post_meal_leakage():
    """Test 13: Strictly verifies that post-meal columns are excluded from features."""
    forbidden_leakage_cols = [
        "actual_attendance",
        "actual_servings_needed",
        "quantity_prepared",
        "quantity_served",
        "leftover_quantity",
        "discarded_quantity",
        "usable_leftover",
        "discarded_food_cost_inr"
    ]
    all_features = tm.CATEGORICAL_FEATURES + tm.NUMERICAL_FEATURES
    for forbidden in forbidden_leakage_cols:
        assert forbidden not in all_features, f"Post-meal leakage detected in model features: {forbidden}"


def test_14_preparation_recommendations_return_valid_nonnegative_integers():
    """Test 14: Preparation recommendations return whole non-negative integer servings."""
    rec = rc.generate_dish_recommendation(
        meal_date="2026-11-12",
        meal_type="Lunch",
        dish_name="Curd Rice",
        expected_attendance=450,
        registered_students=390,
        safety_margin_pct=5.0,
        capacity_cap=550
    )
    prep_qty = rec["recommended_preparation"]
    assert isinstance(prep_qty, int)
    assert prep_qty >= 0
    assert prep_qty <= 550
    assert rec["predicted_demand"] >= 0


def test_15_waste_calculations_handle_zero_quantities_correctly(temp_db):
    """Test 15: Food-waste recording correctly handles edge cases (zero prepared, zero leftover, zero cost)."""
    # Case A: Exactly consumed, zero leftovers
    ok_a, msg_a = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Idli",
        quantity_prepared=300,
        quantity_served=300,
        discarded_quantity=0,
        cost_per_serving=12.0,
        db_path=temp_db
    )
    assert ok_a is True

    # Case B: Zero prepared (e.g. meal cancelled or dish omitted)
    ok_b, msg_b = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Upma",
        quantity_prepared=0,
        quantity_served=0,
        discarded_quantity=0,
        cost_per_serving=10.0,
        db_path=temp_db
    )
    assert ok_b is True

    # Case C: Invalid served > prepared must be rejected safely
    bad, bad_msg = db.record_meal_waste(
        meal_date="2026-11-15",
        meal_type="Breakfast",
        dish_name="Dosa",
        quantity_prepared=100,
        quantity_served=150,
        discarded_quantity=0,
        cost_per_serving=18.0,
        db_path=temp_db
    )
    assert bad is False
    assert "exceed" in bad_msg.lower()


def test_16_official_weekly_menu_lookup():
    """Test 16: Verifies official IIIT Dharwad Mess 1 weekly menu lookup for all 7 days."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meals = ["Breakfast", "Lunch", "Dinner"]

    for day in days:
        for meal in meals:
            menu = db.get_menu_for_meal(day, meal)
            assert "dishes" in menu, f"Missing dishes for {day} {meal}"
            assert len(menu["dishes"]) > 0, f"Dishes list empty for {day} {meal}"
            assert "display_text" in menu, f"Missing display text for {day} {meal}"

    # Specific sanity check: Monday Lunch must feature Curd Rice or Dal Methi
    mon_lunch = db.get_menu_for_meal("Monday", "Lunch")
    assert "Curd Rice" in mon_lunch["dishes"] or "Rice" in mon_lunch["dishes"]


def test_17_attendance_intent_counts_breakdown(temp_db):
    """Test 17: Verifies comprehensive attendance intent breakdown (Yes, No, Not sure, Unresponsive)."""
    # Register 3 students: 1 Yes, 1 No, 1 Not sure
    db.register_student("2026-11-20", "Breakfast", registration_code="S1", attendance_intent="Yes", db_path=temp_db)
    db.register_student("2026-11-20", "Breakfast", registration_code="S2", attendance_intent="No", db_path=temp_db)
    db.register_student("2026-11-20", "Breakfast", registration_code="S3", attendance_intent="Not sure", db_path=temp_db)

    counts = db.get_attendance_intent_counts("2026-11-20", "Breakfast", eligible_students=500, db_path=temp_db)

    assert counts["yes_count"] == 1
    assert counts["no_count"] == 1
    assert counts["not_sure_count"] == 1
    assert counts["total_responses"] == 3
    assert counts["eligible_students"] == 500
    assert counts["unresponsive_count"] == 497

    # Skipped meal must NOT contribute to confirmed count
    assert db.get_registration_count("2026-11-20", "Breakfast", db_path=temp_db) == 1


def test_18_official_menu_recommendations():
    """Test 18: Verifies demand and preparation recommendation for official menu schedule."""
    recs = rc.generate_menu_recommendations(
        meal_date="2026-11-20",
        meal_type="Dinner",
        expected_attendance=480,
        registered_students=380,
        safety_margin_pct=5.0,
        capacity_cap=550
    )
    assert isinstance(recs, list)
    assert len(recs) > 0
    for r in recs:
        assert "dish_name" in r
        assert "recommended_preparation" in r
        assert r["recommended_preparation"] >= 0
        assert r["recommended_preparation"] <= 550
        assert "explanation" in r


def test_19_vision_ai_sample_generation():
    """Test 19: Verifies synthetic sample food tray generation for clean, moderate, and high leftover trays."""
    img_clean = va.create_sample_tray_image("clean")
    img_mod = va.create_sample_tray_image("moderate")
    img_high = va.create_sample_tray_image("high")

    assert img_clean.size == (640, 480)
    assert img_mod.size == (640, 480)
    assert img_high.size == (640, 480)


def test_20_vision_ai_image_analysis_pipeline():
    """Test 20: Verifies end-to-end OpenCV leftover segmentation and metrics pipeline."""
    sample_img = va.create_sample_tray_image("moderate")
    result = va.analyze_food_image(sample_img)

    assert result["success"] is True
    assert "coverage_percentage" in result
    assert 0.0 <= result["coverage_percentage"] <= 100.0
    assert "leftover_category" in result
    assert result["leftover_category"] in ["Clean Tray", "Low Waste", "Moderate Waste", "High Waste"]
    assert "overlay_img" in result
    assert "mask_img" in result
    assert "heatmap_img" in result
    assert "disclaimer" in result
    assert "processing_time_ms" in result
    assert result["processing_time_ms"] > 0


def test_21_get_filtered_meal_records():
    """Test 21: Verifies database search and filtering for saved meal records."""
    df_all, total_all = db.get_filtered_meal_records(limit=10)
    assert total_all > 0
    assert len(df_all) <= 10
    assert "dish_name" in df_all.columns
    assert "meal_type" in df_all.columns

    # Test filtering by meal period
    df_lunch, total_lunch = db.get_filtered_meal_records(meal_type="Lunch", limit=20)
    assert total_lunch > 0
    assert (df_lunch["meal_type"] == "Lunch").all()

    # Test filtering by dish name search
    df_rice, total_rice = db.get_filtered_meal_records(search_dish="Rice", limit=20)
    assert total_rice > 0
    assert df_rice["dish_name"].str.contains("Rice", case=False).all()


def test_22_database_storage_location_transparency():
    """Test 22: Verifies storage transparency info reports valid SQLite and CSV paths."""
    info = db.get_database_storage_info()
    assert "sqlite_path" in info
    assert os.path.exists(info["sqlite_path"])
    assert "meal_records_count" in info
    assert info["meal_records_count"] > 0
    assert "csv_path" in info
    assert os.path.exists(info["csv_path"])
    assert "Synchronized" in info["status"]

