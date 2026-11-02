"""
database.py
===========
SQLite Database Interface for MessMind AI.
Provides persistent storage, parameterized queries, schema migrations,
and automated seeding from synthetic demonstration data.
Ensures deterministic connection closure for robust Windows file handling.
Supports Official Weekly Menu configuration and multi-state attendance intent:
  - 'Yes' (Planning to attend)
  - 'No' (Planning to skip)
  - 'Not sure' (Uncertain / undecided)
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "messmind.db")
DEFAULT_MENU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "weekly_menu.json")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mess_data.csv")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection with Row factory and foreign keys enabled."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """
    Create database tables automatically if they do not exist.
    Performs non-destructive schema migrations to add attendance_intent if missing.
    Safe to call multiple times without altering existing data.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. Student registrations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_registrations (
                registration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                selected_dishes TEXT NOT NULL,
                registration_code TEXT,
                intention_to_attend INTEGER NOT NULL DEFAULT 1,
                attendance_intent TEXT NOT NULL DEFAULT 'Yes',
                submission_timestamp TEXT NOT NULL
            );
        """)

        # Safe schema migration: check if attendance_intent column exists
        cursor.execute("PRAGMA table_info(student_registrations);")
        existing_cols = [row["name"] for row in cursor.fetchall()]
        if "attendance_intent" not in existing_cols:
            cursor.execute("ALTER TABLE student_registrations ADD COLUMN attendance_intent TEXT DEFAULT 'Yes';")

        # Index for fast lookup and duplicate detection
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_registrations_lookup 
            ON student_registrations(meal_date, meal_type, registration_code);
        """)

        # 2. Meal and dish records table (for staff waste recording & history)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                dish_name TEXT NOT NULL,
                quantity_prepared INTEGER NOT NULL,
                quantity_served INTEGER NOT NULL,
                leftover_quantity INTEGER NOT NULL,
                discarded_quantity INTEGER NOT NULL,
                usable_leftover INTEGER NOT NULL,
                cost_per_serving REAL NOT NULL,
                estimated_discarded_food_cost REAL NOT NULL,
                is_synthetic INTEGER NOT NULL DEFAULT 0,
                recorded_timestamp TEXT NOT NULL,
                CONSTRAINT uq_meal_dish UNIQUE (meal_date, meal_type, dish_name)
            );
        """)

        # 3. Demand predictions log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                dish_name TEXT NOT NULL,
                predicted_demand REAL NOT NULL,
                safety_margin_pct REAL NOT NULL,
                recommended_preparation INTEGER NOT NULL,
                actual_demand INTEGER,
                created_timestamp TEXT NOT NULL
            );
        """)

        conn.commit()
    finally:
        conn.close()


def register_student(
    meal_date: str,
    meal_type: str,
    selected_dishes: Optional[List[str]] = None,
    registration_code: Optional[str] = None,
    intention_to_attend: bool = True,
    attendance_intent: str = "Yes",
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, str]:
    """
    Register a student's intention for a specific meal.
    Accepts attendance_intent: 'Yes', 'No', or 'Not sure'.
    Prevents duplicate submissions for the same registration_code + meal_date + meal_type.
    """
    init_db(db_path)
    clean_code = registration_code.strip() if registration_code else None

    # Handle dishes formatting
    if selected_dishes is None:
        dishes_str = "Official Menu"
    elif isinstance(selected_dishes, list):
        dishes_str = ", ".join(selected_dishes) if selected_dishes else "Official Menu"
    else:
        dishes_str = str(selected_dishes)

    # Normalize intent
    intent_clean = attendance_intent.strip()
    if intent_clean not in ["Yes", "No", "Not sure"]:
        intent_clean = "Yes" if intention_to_attend else "No"

    # intention_to_attend integer: strictly 1 for confirmed 'Yes', 0 for 'No' or 'Not sure'
    attend_int = 1 if intent_clean == "Yes" else 0
    now_str = datetime.now().isoformat()

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # Check for duplicate if code is provided
        if clean_code:
            cursor.execute("""
                SELECT registration_id FROM student_registrations
                WHERE meal_date = ? AND meal_type = ? AND registration_code = ?
            """, (meal_date, meal_type, clean_code))
            row = cursor.fetchone()
            if row:
                # Update existing registration with new intent
                cursor.execute("""
                    UPDATE student_registrations
                    SET selected_dishes = ?, intention_to_attend = ?, attendance_intent = ?, submission_timestamp = ?
                    WHERE registration_id = ?
                """, (dishes_str, attend_int, intent_clean, now_str, row["registration_id"]))
                conn.commit()
                return True, f"Registration updated successfully for {clean_code} on {meal_date} ({meal_type}) with intent: '{intent_clean}'."

        # Insert new registration
        cursor.execute("""
            INSERT INTO student_registrations (
                meal_date, meal_type, selected_dishes, registration_code,
                intention_to_attend, attendance_intent, submission_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            meal_date,
            meal_type,
            dishes_str,
            clean_code,
            attend_int,
            intent_clean,
            now_str
        ))
        conn.commit()
        return True, f"Registration recorded successfully for {meal_date} ({meal_type}) with intent: '{intent_clean}'."
    finally:
        conn.close()


def get_registration_count(meal_date: str, meal_type: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """Return count of registered students with confirmed attendance (Yes) for the meal."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count FROM student_registrations
            WHERE meal_date = ? AND meal_type = ? 
              AND (attendance_intent = 'Yes' OR (attendance_intent IS NULL AND intention_to_attend = 1))
        """, (meal_date, meal_type))
        row = cursor.fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()


def get_attendance_intent_counts(
    meal_date: str,
    meal_type: str,
    eligible_students: int = 500,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, int]:
    """
    Retrieve comprehensive attendance intention breakdown:
      - yes_count: planning to attend
      - no_count: planning to skip
      - not_sure_count: uncertain / undecided
      - total_responses: total unique registrations submitted
      - eligible_students: total configured hostel mess strength
      - unresponsive_count: eligible_students - total_responses
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # Query all registrations for this date and meal
        cursor.execute("""
            SELECT attendance_intent, intention_to_attend FROM student_registrations
            WHERE meal_date = ? AND meal_type = ?
        """, (meal_date, meal_type))
        rows = cursor.fetchall()

        yes_count = 0
        no_count = 0
        not_sure_count = 0

        for r in rows:
            intent = r["attendance_intent"]
            if intent == "Yes":
                yes_count += 1
            elif intent == "No":
                no_count += 1
            elif intent == "Not sure":
                not_sure_count += 1
            else:
                # Fallback for historical rows without attendance_intent column
                if r["intention_to_attend"] == 1:
                    yes_count += 1
                else:
                    no_count += 1

        total_responses = len(rows)
        unresponsive_count = max(0, eligible_students - total_responses)

        return {
            "yes_count": yes_count,
            "no_count": no_count,
            "not_sure_count": not_sure_count,
            "total_responses": total_responses,
            "eligible_students": eligible_students,
            "unresponsive_count": unresponsive_count
        }
    finally:
        conn.close()


def get_all_registrations(meal_date: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve student registrations optionally filtered by meal date."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        if meal_date:
            query = """
                SELECT registration_id, meal_date, meal_type, selected_dishes, 
                       registration_code, intention_to_attend, attendance_intent, submission_timestamp
                FROM student_registrations
                WHERE meal_date = ?
                ORDER BY submission_timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=(meal_date,))
        else:
            query = """
                SELECT registration_id, meal_date, meal_type, selected_dishes, 
                       registration_code, intention_to_attend, attendance_intent, submission_timestamp
                FROM student_registrations
                ORDER BY submission_timestamp DESC LIMIT 500
            """
            return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def record_meal_waste(
    meal_date: str,
    meal_type: str,
    dish_name: str,
    quantity_prepared: int,
    quantity_served: int,
    discarded_quantity: int,
    cost_per_serving: float,
    is_synthetic: bool = False,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, str]:
    """
    Record or update mess staff food preparation and waste logs.
    Strictly validates business rules.
    """
    init_db(db_path)

    if quantity_prepared < 0 or quantity_served < 0 or discarded_quantity < 0:
        return False, "Quantities cannot be negative."

    if quantity_served > quantity_prepared:
        return False, f"Quantity served ({quantity_served}) cannot exceed quantity prepared ({quantity_prepared})."

    leftover_quantity = quantity_prepared - quantity_served

    if discarded_quantity > leftover_quantity:
        return False, f"Discarded quantity ({discarded_quantity}) cannot exceed leftover quantity ({leftover_quantity})."

    usable_leftover = leftover_quantity - discarded_quantity
    estimated_cost = round(float(discarded_quantity * cost_per_serving), 2)
    now_str = datetime.now().isoformat()

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meal_records (
                meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                leftover_quantity, discarded_quantity, usable_leftover,
                cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(meal_date, meal_type, dish_name) DO UPDATE SET
                quantity_prepared = excluded.quantity_prepared,
                quantity_served = excluded.quantity_served,
                leftover_quantity = excluded.leftover_quantity,
                discarded_quantity = excluded.discarded_quantity,
                usable_leftover = excluded.usable_leftover,
                cost_per_serving = excluded.cost_per_serving,
                estimated_discarded_food_cost = excluded.estimated_discarded_food_cost,
                is_synthetic = excluded.is_synthetic,
                recorded_timestamp = excluded.recorded_timestamp;
        """, (
            meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
            leftover_quantity, discarded_quantity, usable_leftover,
            cost_per_serving, estimated_cost, 1 if is_synthetic else 0, now_str
        ))
        conn.commit()
        
        # Dual-persistence: Also synchronize to CSV so both SQLite and CSV stay in sync
        sync_ok = sync_record_to_csv(
            meal_date=meal_date,
            meal_type=meal_type,
            dish_name=dish_name,
            quantity_prepared=quantity_prepared,
            quantity_served=quantity_served,
            discarded_quantity=discarded_quantity,
            cost_per_serving=cost_per_serving
        )
        csv_sync_note = " & synchronized to CSV (data/mess_data.csv)" if sync_ok else ""
        return True, f"Record saved for {dish_name} ({meal_type}, {meal_date}) into SQLite database (data/messmind.db){csv_sync_note}."
    finally:
        conn.close()


def sync_record_to_csv(
    meal_date: str,
    meal_type: str,
    dish_name: str,
    quantity_prepared: int,
    quantity_served: int,
    discarded_quantity: int,
    cost_per_serving: float,
    csv_path: str = DEFAULT_CSV_PATH
) -> bool:
    """
    Append or update a live meal record in mess_data.csv so that
    both the SQLite database and CSV file remain 100% synchronized.
    """
    try:
        if not os.path.exists(csv_path):
            return False

        try:
            day_of_week = datetime.strptime(meal_date, "%Y-%m-%d").strftime("%A")
        except Exception:
            day_of_week = "Monday"

        leftover_quantity = max(0, quantity_prepared - quantity_served)
        usable_leftover = max(0, leftover_quantity - discarded_quantity)
        discarded_food_cost = round(float(discarded_quantity * cost_per_serving), 2)

        df = pd.read_csv(csv_path)
        mask = (df["date"] == meal_date) & (df["meal_type"] == meal_type) & (df["dish_name"] == dish_name)
        if mask.any():
            df.loc[mask, "quantity_prepared"] = quantity_prepared
            df.loc[mask, "quantity_served"] = quantity_served
            df.loc[mask, "leftover_quantity"] = leftover_quantity
            df.loc[mask, "discarded_quantity"] = discarded_quantity
            df.loc[mask, "usable_leftover"] = usable_leftover
            df.loc[mask, "cost_per_serving_inr"] = cost_per_serving
            df.loc[mask, "discarded_food_cost_inr"] = discarded_food_cost
            df.loc[mask, "is_synthetic"] = False
            df.loc[mask, "data_source"] = "Mess Staff Live Kitchen Log"
        else:
            new_row = {
                "date": meal_date,
                "day_of_week": day_of_week,
                "meal_type": meal_type,
                "dish_name": dish_name,
                "expected_attendance": quantity_prepared,
                "registered_students": quantity_prepared,
                "historical_demand": float(quantity_prepared),
                "actual_attendance": quantity_served,
                "actual_servings_needed": quantity_served,
                "quantity_prepared": quantity_prepared,
                "quantity_served": quantity_served,
                "leftover_quantity": leftover_quantity,
                "discarded_quantity": discarded_quantity,
                "usable_leftover": usable_leftover,
                "cost_per_serving_inr": cost_per_serving,
                "discarded_food_cost_inr": discarded_food_cost,
                "is_synthetic": False,
                "data_source": "Mess Staff Live Kitchen Log"
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Warning: CSV sync skipped: {e}")
        return False


def get_recent_meal_records(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve the most recently recorded meal records (newest first)."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
            SELECT id, meal_date, meal_type, dish_name, quantity_prepared,
                   quantity_served, leftover_quantity, discarded_quantity,
                   usable_leftover, cost_per_serving, estimated_discarded_food_cost,
                   recorded_timestamp
            FROM meal_records
            ORDER BY recorded_timestamp DESC, id DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


def get_filtered_meal_records(
    search_dish: Optional[str] = None,
    meal_type: Optional[str] = None,
    date_filter: Optional[str] = None,
    custom_start_date: Optional[str] = None,
    custom_end_date: Optional[str] = None,
    limit: Optional[int] = 50,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[pd.DataFrame, int]:
    """Retrieve meal records matching specific search filters, returning (df, total_matching_count)."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        where_clauses = []
        params: List[Any] = []

        if search_dish and search_dish.strip():
            where_clauses.append("dish_name LIKE ?")
            params.append(f"%{search_dish.strip()}%")

        if meal_type and meal_type not in ["All", "All Meals", ""]:
            where_clauses.append("meal_type = ?")
            params.append(meal_type)

        today_dt = datetime.now().date()
        if date_filter == "Today":
            where_clauses.append("meal_date = ?")
            params.append(today_dt.strftime("%Y-%m-%d"))
        elif date_filter == "Last 7 Days":
            start = (today_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            where_clauses.append("meal_date >= ?")
            params.append(start)
        elif date_filter == "Last 30 Days":
            start = (today_dt - timedelta(days=30)).strftime("%Y-%m-%d")
            where_clauses.append("meal_date >= ?")
            params.append(start)
        elif date_filter == "Custom":
            if custom_start_date:
                where_clauses.append("meal_date >= ?")
                params.append(str(custom_start_date))
            if custom_end_date:
                where_clauses.append("meal_date <= ?")
                params.append(str(custom_end_date))

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Count total matches
        count_query = f"SELECT COUNT(*) FROM meal_records {where_sql}"
        cursor = conn.cursor()
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Fetch records (newest first by meal_date, id)
        query = f"""
            SELECT id, meal_date, meal_type, dish_name, quantity_prepared,
                   quantity_served, leftover_quantity, discarded_quantity,
                   usable_leftover, cost_per_serving, estimated_discarded_food_cost,
                   recorded_timestamp
            FROM meal_records
            {where_sql}
            ORDER BY meal_date DESC, id DESC
        """
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql_query(query, conn, params=params)
        return df, total_count
    finally:
        conn.close()


def get_database_storage_info(db_path: str = DEFAULT_DB_PATH, csv_path: str = DEFAULT_CSV_PATH) -> Dict[str, Any]:
    """Retrieve exact file paths, sizes, and record counts for auditing transparency."""
    init_db(db_path)
    sqlite_abs = os.path.abspath(db_path)
    csv_abs = os.path.abspath(csv_path)

    sqlite_size_kb = round(os.path.getsize(db_path) / 1024, 1) if os.path.exists(db_path) else 0
    csv_size_kb = round(os.path.getsize(csv_path) / 1024, 1) if os.path.exists(csv_path) else 0

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM meal_records")
        meal_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM student_registrations")
        student_count = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM prediction_records")
        pred_count = cursor.fetchone()["cnt"]
    finally:
        conn.close()

    csv_rows = 0
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_rows = max(0, sum(1 for _ in f) - 1)
        except Exception:
            csv_rows = 0

    return {
        "sqlite_path": sqlite_abs,
        "sqlite_rel_path": "data/messmind.db",
        "sqlite_size_kb": sqlite_size_kb,
        "meal_records_count": meal_count,
        "student_registrations_count": student_count,
        "prediction_records_count": pred_count,
        "total_sqlite_rows": meal_count + student_count + pred_count,
        "csv_path": csv_abs,
        "csv_rel_path": "data/mess_data.csv",
        "csv_size_kb": csv_size_kb,
        "csv_rows_count": csv_rows,
        "status": "Synchronized & Live"
    }


def get_meal_records(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    meal_type: Optional[str] = None,
    dish_name: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieve operational meal records with optional filters."""
    init_db(db_path)
    conditions = []
    params = []

    if start_date:
        conditions.append("meal_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("meal_date <= ?")
        params.append(end_date)
    if meal_type and meal_type != "All":
        conditions.append("meal_type = ?")
        params.append(meal_type)
    if dish_name and dish_name != "All":
        conditions.append("dish_name = ?")
        params.append(dish_name)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT id, meal_date, meal_type, dish_name, quantity_prepared,
               quantity_served, leftover_quantity, discarded_quantity,
               usable_leftover, cost_per_serving, estimated_discarded_food_cost,
               is_synthetic, recorded_timestamp
        FROM meal_records
        {where_clause}
        ORDER BY meal_date DESC, meal_type ASC, dish_name ASC
    """
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def save_prediction(
    meal_date: str,
    meal_type: str,
    dish_name: str,
    predicted_demand: float,
    safety_margin_pct: float,
    recommended_preparation: int,
    actual_demand: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH
):
    """Log an AI demand recommendation for auditing and evaluation."""
    init_db(db_path)
    now_str = datetime.now().isoformat()
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_records (
                meal_date, meal_type, dish_name, predicted_demand,
                safety_margin_pct, recommended_preparation, actual_demand, created_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meal_date, meal_type, dish_name, round(predicted_demand, 2),
            safety_margin_pct, recommended_preparation, actual_demand, now_str
        ))
        conn.commit()
    finally:
        conn.close()


def get_prediction_records(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve logged predictions."""
    init_db(db_path)
    query = """
        SELECT id, meal_date, meal_type, dish_name, predicted_demand,
               safety_margin_pct, recommended_preparation, actual_demand, created_timestamp
        FROM prediction_records
        ORDER BY created_timestamp DESC LIMIT ?
    """
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


def seed_database_from_csv(csv_path: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Initialize SQLite database with synthetic demonstration records from CSV.
    Uses INSERT OR IGNORE so re-running will not duplicate records.
    Returns number of records inserted.
    """
    init_db(db_path)
    if not os.path.exists(csv_path):
        return 0

    df = pd.read_csv(csv_path)
    inserted_count = 0

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            now_str = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR IGNORE INTO meal_records (
                    meal_date, meal_type, dish_name, quantity_prepared, quantity_served,
                    leftover_quantity, discarded_quantity, usable_leftover,
                    cost_per_serving, estimated_discarded_food_cost, is_synthetic, recorded_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                str(row["date"]),
                str(row["meal_type"]),
                str(row["dish_name"]),
                int(row["quantity_prepared"]),
                int(row["quantity_served"]),
                int(row["leftover_quantity"]),
                int(row["discarded_quantity"]),
                int(row["usable_leftover"]),
                float(row["cost_per_serving_inr"]),
                float(row["discarded_food_cost_inr"]),
                now_str
            ))
            if cursor.rowcount > 0:
                inserted_count += 1
        conn.commit()
        return inserted_count
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Official Weekly Menu Helpers
# -----------------------------------------------------------------------------

def load_weekly_menu(menu_path: str = DEFAULT_MENU_PATH) -> Dict[str, Any]:
    """Load the official weekly mess menu from JSON configuration."""
    if os.path.exists(menu_path):
        try:
            with open(menu_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading menu from {menu_path}: {e}")
    return {"menu": {}, "metadata": {}}


def save_weekly_menu(menu_data: Dict[str, Any], menu_path: str = DEFAULT_MENU_PATH) -> bool:
    """Save the official weekly mess menu back to JSON configuration."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(menu_path)), exist_ok=True)
        with open(menu_path, "w", encoding="utf-8") as f:
            json.dump(menu_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving menu to {menu_path}: {e}")
        return False


def get_menu_for_meal(day_of_week: str, meal_type: str, menu_path: str = DEFAULT_MENU_PATH) -> Dict[str, Any]:
    """
    Retrieve the official menu for a given day and meal.
    Returns dictionary with:
      - 'dishes': list of primary dishes cooked
      - 'display_text': formatted full menu string (including accompaniments)
      - 'primary_dish': dominant item
    """
    data = load_weekly_menu(menu_path)
    days_dict = data.get("menu", {})
    day_info = days_dict.get(day_of_week, {})
    meal_info = day_info.get(meal_type, None)

    if meal_info:
        return meal_info

    # Fallback if unconfigured day/meal
    fallback_dishes = {
        "Breakfast": ["Idli", "Vada", "Chutney"],
        "Lunch": ["Rice", "Dal", "Vegetable Curry", "Chapati"],
        "Snacks": ["Samosa", "Tea"],
        "Dinner": ["Rice", "Dal", "Paneer Curry", "Chapati"]
    }
    return {
        "dishes": fallback_dishes.get(meal_type, ["Rice", "Dal"]),
        "display_text": f"Official {meal_type} Menu ({day_of_week})",
        "primary_dish": fallback_dishes.get(meal_type, ["Rice"])[0]
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized with attendance_intent support.")
    menu = get_menu_for_meal("Monday", "Lunch")
    print(f"Monday Lunch Official Menu: {menu['display_text']}")
