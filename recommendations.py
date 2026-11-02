"""
recommendations.py
==================
Smart Food Demand Prediction & Preparation Recommendation Engine for MessMind AI.
Combines student advance registration counts (Yes, No, Not sure intent breakdown),
strictly chronological historical demand, and the trained Scikit-learn RandomForest pipeline
to recommend optimal preparation quantities for official weekly menu items.
Includes configurable safety margins, fallback estimation, and nontechnical explanations.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from database import DEFAULT_DB_PATH, get_connection, save_prediction, get_menu_for_meal

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "demand_model.joblib")

# Baseline take-rate priors for cold-start fallbacks
FALLBACK_TAKE_RATES = {
    "Rice": 0.90,
    "Curd Rice": 0.88,
    "Dal": 0.85,
    "Chapati": 0.80,
    "Vegetable Curry": 0.78,
    "Curd": 0.65,
    "Paneer Curry": 0.94,
    "Idli": 0.82,
    "Dosa": 0.92,
    "Upma": 0.68,
    "Pongal": 0.74,
    "Aloo Paratha": 0.85,
    "Samosa": 0.88,
    "Pani Puri": 0.92,
    "Punugulu": 0.80,
    "Veg Noodles": 0.85,
    "Pasta": 0.84,
    "Masala Bhel": 0.75,
    "Girmit": 0.78
}

# Dish category mapping for the official IIIT Dharwad menu to model training categories
DISH_CATEGORY_MAPPING = {
    # Rice varieties
    "Curd Rice": "Rice",
    "Veg Pulav": "Rice",
    "Puliogare": "Rice",
    "Onion Rice": "Rice",
    "Tomato Rice": "Rice",
    "Jeera Rice": "Rice",
    "Plain Rice": "Rice",
    "Rice": "Rice",
    "Khichadi": "Rice",
    "Soyabean Biryani": "Rice",
    "Vangi Bath": "Rice",

    # Dal varieties
    "Dal Methi": "Dal",
    "Dal Tadka": "Dal",
    "Dal Makhani": "Dal",
    "Dal Fry": "Dal",
    "Dal Palak": "Dal",
    "Dal Pancha ratan": "Dal",
    "Moong Dal": "Dal",
    "Masoor Dal": "Dal",
    "Channa Dal": "Dal",
    "Dal": "Dal",
    "Rasam": "Dal",

    # Paneer & high-demand curries
    "Kadai Paneer": "Paneer Curry",
    "Paneer Kholapuri": "Paneer Curry",
    "Shahi Paneer": "Paneer Curry",
    "Panner Butter Masala": "Paneer Curry",
    "Paneer Curry": "Paneer Curry",

    # Veg curries & gravies
    "Brinjal Bhartha": "Vegetable Curry",
    "White Pese Masala": "Vegetable Curry",
    "Jeera Aloo": "Vegetable Curry",
    "Chawli Masala": "Vegetable Curry",
    "Brinjal Tomato": "Vegetable Curry",
    "Kadhi Pakoda": "Vegetable Curry",
    "Bhendi Sabji": "Vegetable Curry",
    "Veg Kadai": "Vegetable Curry",
    "Ivy Gourd Fry": "Vegetable Curry",
    "Cabbage Baji": "Vegetable Curry",
    "Aloo 65": "Vegetable Curry",
    "Soyabean Sabzi Gravy": "Vegetable Curry",
    "Sorakay Chana": "Vegetable Curry",
    "Veg Kholapuri": "Vegetable Curry",
    "Aloo Mutter Sabzi": "Vegetable Curry",
    "Vegetable Curry": "Vegetable Curry",

    # Breads
    "Chapati / Fulka": "Chapati",
    "Chapati": "Chapati",
    "Puri": "Chapati",
    "Chole Bhature": "Chapati",
    "Aloo Paratha": "Chapati",
    "Aloo Partha": "Chapati",

    # Breakfast staples
    "Masala Dosa": "Dosa",
    "Dosa": "Dosa",
    "Uthappamm": "Idli",
    "Idli": "Idli",
    "Vada": "Idli",
    "Upma": "Upma",
    "Sewaiya Upma": "Upma",
    "Pongal": "Pongal",
    "Poha": "Pongal",

    # Dairy / Extras
    "Curd": "Curd",
    "Buttermilk": "Curd",
    "Lassi": "Curd"
}


def load_demand_model(model_path: str = DEFAULT_MODEL_PATH):
    """Safely load the trained joblib model pipeline."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    return None


def get_canonical_dish(dish_name: str) -> str:
    """Map an official menu dish to its nearest base category for ML inference."""
    return DISH_CATEGORY_MAPPING.get(dish_name.strip(), dish_name.strip())


class HistoricalDemandResult(float):
    """Dual float-and-tuple result object for bulletproof compatibility."""
    is_fallback: bool

    def __new__(cls, value: float, is_fallback: bool = False):
        obj = super().__new__(cls, float(value))
        obj.is_fallback = is_fallback
        return obj

    def __iter__(self):
        yield float(self)
        yield self.is_fallback

    def __getitem__(self, index):
        if index == 0:
            return float(self)
        elif index == 1:
            return self.is_fallback
        raise IndexError("HistoricalDemandResult index out of range")

    def __len__(self):
        return 2


def get_historical_demand_for_dish(
    meal_date: Optional[str] = None,
    meal_type: str = "Lunch",
    dish_name: Optional[str] = None,
    expected_attendance: int = 450,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> HistoricalDemandResult:
    """
    Compute historical demand using ONLY records chronologically strictly before meal_date.
    Checks for exact dish name first, then canonical category.
    Supports both (meal_date, meal_type, dish_name) AND (dish_name, meal_type) calling signatures.
    Returns HistoricalDemandResult (acts as both float and (value, is_fallback) tuple).
    """
    # Detect if meal_date is actually dish_name (e.g. called as get_historical_demand_for_dish(dish, meal_type))
    if meal_date is not None and dish_name is None:
        if not ("-" in str(meal_date) and len(str(meal_date)) >= 8):
            dish_name = meal_date
            meal_date = datetime.now().strftime("%Y-%m-%d")

    if dish_name is None:
        dish_name = kwargs.get("dish") or kwargs.get("dish_name") or "Plain Rice"
    if meal_date is None:
        meal_date = datetime.now().strftime("%Y-%m-%d")

    canonical = get_canonical_dish(dish_name)

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            # 1. Try exact dish name
            cursor.execute("""
                SELECT quantity_served FROM meal_records
                WHERE dish_name = ? AND meal_type = ? AND meal_date < ?
                ORDER BY meal_date DESC LIMIT 14
            """, (dish_name, meal_type, meal_date))
            rows = cursor.fetchall()
            if rows and len(rows) >= 3:
                values = [r["quantity_served"] for r in rows]
                return HistoricalDemandResult(round(float(np.mean(values)), 1), False)

            # 2. Try canonical category (e.g. 'Rice', 'Dal', 'Paneer Curry')
            if canonical != dish_name:
                cursor.execute("""
                    SELECT quantity_served FROM meal_records
                    WHERE dish_name = ? AND meal_type = ? AND meal_date < ?
                    ORDER BY meal_date DESC LIMIT 14
                """, (canonical, meal_type, meal_date))
                rows_canon = cursor.fetchall()
                if rows_canon and len(rows_canon) >= 2:
                    values = [r["quantity_served"] for r in rows_canon]
                    return HistoricalDemandResult(round(float(np.mean(values)), 1), False)
    except Exception:
        pass

    # Fallback to configured domain priors
    take_rate = FALLBACK_TAKE_RATES.get(canonical, FALLBACK_TAKE_RATES.get(dish_name, 0.75))
    fallback_val = round(float(expected_attendance * take_rate), 1)
    return HistoricalDemandResult(fallback_val, True)


def generate_dish_recommendation(
    meal_date: str,
    meal_type: str,
    dish_name: str,
    expected_attendance: int = 450,
    registered_students: int = 400,
    safety_margin_pct: float = 5.0,
    capacity_cap: int = 550,
    model_path: str = DEFAULT_MODEL_PATH,
    db_path: str = DEFAULT_DB_PATH,
    persist_prediction: bool = False,
    intent_breakdown: Optional[Dict[str, int]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate demand prediction and preparation recommendation for an official menu item.
    Follows:
      Recommended preparation = predicted demand adjusted by safety margin
      Whole number of servings, capped at mess capacity.
    """
    # Accommodate flexible kwargs from UI
    if "intent_count" in kwargs and intent_breakdown is None:
        intent_breakdown = {"yes_count": int(kwargs["intent_count"]), "not_sure_count": 0}

    # 1. Parse date and day of week
    try:
        dt = datetime.strptime(meal_date, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
        meal_date = dt.strftime("%Y-%m-%d")

    day_of_week = kwargs.get("day_of_week", dt.strftime("%A"))
    is_weekend = int(day_of_week in ["Saturday", "Sunday"])

    # 2. Attendance intent consideration
    # If detailed intent breakdown is available, adjust effective registered count:
    # 100% of 'Yes' + ~50% of 'Not sure' attend, 'No' are excluded.
    if intent_breakdown:
        yes_c = intent_breakdown.get("yes_count", registered_students)
        not_sure_c = intent_breakdown.get("not_sure_count", 0)
        effective_registered = yes_c + int(round(not_sure_c * 0.50))
    else:
        effective_registered = registered_students

    # 3. Derive historical demand strictly from past dates
    hist_demand, is_fallback = get_historical_demand_for_dish(
        meal_date=meal_date,
        meal_type=meal_type,
        dish_name=dish_name,
        expected_attendance=expected_attendance,
        db_path=db_path
    )

    # 4. Map dish to canonical name for ML feature row
    canonical_dish = get_canonical_dish(dish_name)
    reg_ratio = round(effective_registered / max(expected_attendance, 1), 4)

    feature_row = pd.DataFrame([{
        "day_of_week": day_of_week,
        "meal_type": meal_type,
        "dish_name": canonical_dish,
        "expected_attendance": int(expected_attendance),
        "registered_students": int(effective_registered),
        "historical_demand": float(hist_demand),
        "registration_ratio": float(reg_ratio),
        "is_weekend": is_weekend
    }])

    # 5. Model Prediction
    model = kwargs.get("model") or load_demand_model(model_path)
    model_loaded = model is not None

    if model_loaded:
        try:
            pred_raw = float(model.predict(feature_row)[0])
            predicted_demand = max(10.0, round(pred_raw, 1))
        except Exception:
            predicted_demand = hist_demand
            model_loaded = False
    else:
        # Fallback heuristic using domain prior
        take_rate = FALLBACK_TAKE_RATES.get(canonical_dish, 0.75)
        heuristic = effective_registered * take_rate * 0.95 + (expected_attendance - effective_registered) * 0.10
        predicted_demand = round(float(np.clip(heuristic, 10, capacity_cap)), 1)

    # 6. Safety Margin & Final Preparation Recommendation
    margin_multiplier = 1.0 + (max(0.0, float(safety_margin_pct)) / 100.0)
    raw_recommendation = round(predicted_demand * margin_multiplier)
    recommended_preparation = int(np.clip(raw_recommendation, 10, capacity_cap))
    safety_margin_units = max(0, recommended_preparation - int(round(predicted_demand)))

    # 7. Non-technical explanation for mess staff
    source_tag = "ML Model (Random Forest)" if model_loaded else "Historical Fallback Heuristic"
    hist_tag = "Prior Mess Records" if not is_fallback else "Official Menu Fallback Prior"
    explanation = (
        f"For official menu dish '{dish_name}' on {day_of_week} {meal_type}, based on {effective_registered} "
        f"active registered students and {hist_tag} ({hist_demand} servings), {source_tag} estimated "
        f"{predicted_demand:.0f} servings needed. Adding a +{safety_margin_pct:.1f}% safety buffer yields "
        f"a recommended prep of {recommended_preparation} servings (capped at {capacity_cap}). Final decision rests with mess operator."
    )

    if persist_prediction:
        save_prediction(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish_name,
            predicted_demand=predicted_demand,
            safety_margin_pct=safety_margin_pct,
            recommended_preparation=recommended_preparation,
            actual_demand=None,
            db_path=db_path
        )

    return {
        "dish_name": dish_name,
        "canonical_category": canonical_dish,
        "meal_type": meal_type,
        "meal_date": meal_date,
        "day_of_week": day_of_week,
        "predicted_demand": predicted_demand,
        "safety_margin_pct": safety_margin_pct,
        "safety_margin_units": safety_margin_units,
        "recommended_preparation": recommended_preparation,
        "historical_demand": hist_demand,
        "is_fallback": is_fallback,
        "registered_students": effective_registered,
        "expected_attendance": expected_attendance,
        "model_loaded": model_loaded,
        "explanation": explanation,
        "advice": explanation
    }


def generate_menu_recommendations(
    meal_date: str,
    meal_type: str,
    dish_list: Optional[List[str]] = None,
    expected_attendance: int = 480,
    registered_students: int = 410,
    safety_margin_pct: float = 5.0,
    capacity_cap: int = 550,
    model_path: str = DEFAULT_MODEL_PATH,
    db_path: str = DEFAULT_DB_PATH,
    persist: bool = False,
    intent_breakdown: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Generate recommendations for all dishes on the official menu for the selected date & meal.
    If dish_list is None, automatically loads official dishes from weekly_menu.json!
    """
    # 1. Parse date for day of week
    try:
        dt = datetime.strptime(meal_date, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
        meal_date = dt.strftime("%Y-%m-%d")
    day_of_week = dt.strftime("%A")

    # 2. Look up official menu if not provided
    if dish_list is None or len(dish_list) == 0:
        menu_info = get_menu_for_meal(day_of_week, meal_type)
        dish_list = menu_info.get("dishes", ["Rice", "Dal", "Chapati"])

    results = []
    for dish in dish_list:
        rec = generate_dish_recommendation(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish,
            expected_attendance=expected_attendance,
            registered_students=registered_students,
            safety_margin_pct=safety_margin_pct,
            capacity_cap=capacity_cap,
            model_path=model_path,
            db_path=db_path,
            persist_prediction=persist,
            intent_breakdown=intent_breakdown
        )
        results.append(rec)
    return results


if __name__ == "__main__":
    recs = generate_menu_recommendations(
        meal_date="2026-11-05",
        meal_type="Lunch",
        expected_attendance=480,
        registered_students=410,
        safety_margin_pct=5.0
    )
    print("Official Menu Recommendations for 2026-11-05 Lunch:")
    for r in recs:
        print(f"Dish: {r['dish_name']:<18} | Predicted: {r['predicted_demand']:<5} | Recommended: {r['recommended_preparation']} servings")
