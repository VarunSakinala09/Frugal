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
import hashlib
import secrets
import hmac
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

        # 4. User accounts with RBAC (Manager vs Student)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('manager', 'student')),
                full_name TEXT NOT NULL,
                roll_no TEXT,
                hostel TEXT DEFAULT 'Hostel-1 (Mess 1)',
                room_number TEXT DEFAULT 'B-204',
                dietary_preference TEXT DEFAULT 'Vegetarian',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
        """)
        # Ensure room_number and dietary_preference exist in existing database schemas
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN room_number TEXT DEFAULT 'B-204'")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN dietary_preference TEXT DEFAULT 'Vegetarian'")
        except Exception:
            pass

        # 5. Dynamic and published menu items table (Manager CRUD & publish control)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS menu_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                dish_name TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'Main Course',
                cost_per_serving REAL NOT NULL DEFAULT 15.0,
                is_published INTEGER NOT NULL DEFAULT 1,
                specific_date TEXT,
                created_by TEXT DEFAULT 'manager',
                updated_at TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_menu_lookup 
            ON menu_items(day_of_week, meal_type, is_published);
        """)

        # 6. Student meal attendance intents (RSVP per meal period)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_meal_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                student_roll TEXT NOT NULL,
                meal_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                intent_status TEXT NOT NULL CHECK(intent_status IN ('Attending', 'Not Attending', 'Not sure')),
                dietary_preference TEXT DEFAULT 'Standard Veg',
                updated_at TEXT NOT NULL,
                CONSTRAINT uq_student_intent UNIQUE (student_roll, meal_date, meal_type)
            );
        """)

        # 7. Student food preference and feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                student_roll TEXT NOT NULL,
                meal_date TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                dish_name TEXT,
                rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                feedback_text TEXT,
                skip_reason TEXT,
                created_at TEXT NOT NULL
            );
        """)

        # 8. Multi-plate OpenCV batch intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vision_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_code TEXT UNIQUE NOT NULL,
                service_date TEXT NOT NULL,
                meal_period TEXT NOT NULL,
                dish_name TEXT NOT NULL,
                total_images INTEGER NOT NULL DEFAULT 0,
                valid_images INTEGER NOT NULL DEFAULT 0,
                failed_images INTEGER NOT NULL DEFAULT 0,
                avg_coverage REAL NOT NULL DEFAULT 0.0,
                median_coverage REAL NOT NULL DEFAULT 0.0,
                min_coverage REAL NOT NULL DEFAULT 0.0,
                max_coverage REAL NOT NULL DEFAULT 0.0,
                high_leftover_count INTEGER NOT NULL DEFAULT 0,
                low_leftover_count INTEGER NOT NULL DEFAULT 0,
                dominant_fingerprint TEXT NOT NULL DEFAULT 'Scattered Residue',
                actual_discarded_kg REAL,
                notes TEXT,
                created_at TEXT NOT NULL
            );
        """)

        # 9. Individual plate inspection results within a batch
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vision_plate_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                plate_id TEXT NOT NULL,
                coverage_pct REAL NOT NULL,
                contour_count INTEGER NOT NULL,
                cluster_dispersion REAL NOT NULL DEFAULT 0.0,
                fingerprint TEXT NOT NULL,
                adaptive_strategy TEXT NOT NULL,
                quality_score REAL NOT NULL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'Success',
                notes TEXT,
                created_at TEXT NOT NULL
            );
        """)

        # 10. Mess configuration settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mess_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # 11. Campus announcements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS announcements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                priority TEXT DEFAULT 'Normal',
                target_role TEXT DEFAULT 'all',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
        """)

        # 12. Security & administrative audit logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                entity TEXT NOT NULL,
                details TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)

        conn.commit()
    finally:
        conn.close()

    # Seed initial data if tables are fresh
    seed_default_users(db_path)
    seed_default_settings(db_path)
    seed_default_announcements(db_path)
    seed_menu_items_from_json(db_path)


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

        # Check student_meal_intents table first (live student votes)
        cursor.execute("""
            SELECT intent_status FROM student_meal_intents
            WHERE meal_date = ? AND meal_type = ?
        """, (meal_date, meal_type))
        live_rows = cursor.fetchall()

        yes_count = 0
        no_count = 0
        not_sure_count = 0

        if live_rows:
            for r in live_rows:
                st = r["intent_status"]
                if st == "Attending":
                    yes_count += 1
                elif st == "Not Attending":
                    no_count += 1
                elif st == "Not sure":
                    not_sure_count += 1
            total_responses = len(live_rows)
        else:
            # Fallback to student_registrations table
            cursor.execute("""
                SELECT attendance_intent, intention_to_attend FROM student_registrations
                WHERE meal_date = ? AND meal_type = ?
            """, (meal_date, meal_type))
            rows = cursor.fetchall()
            for r in rows:
                intent = r["attendance_intent"]
                if intent == "Yes":
                    yes_count += 1
                elif intent == "No":
                    no_count += 1
                elif intent == "Not sure":
                    not_sure_count += 1
                else:
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
            "Attending": yes_count,
            "Not Attending": no_count,
            "Not sure": not_sure_count,
            "attending_count": yes_count,
            "not_attending_count": no_count,
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
    meal_date: Optional[str] = None,
    meal_type: Optional[str] = None,
    dish_name: Optional[str] = None,
    quantity_prepared: Optional[int] = None,
    quantity_served: Optional[int] = None,
    discarded_quantity: Optional[int] = None,
    cost_per_serving: Optional[float] = None,
    is_synthetic: bool = False,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> Tuple[bool, str]:
    """
    Record or update mess staff food preparation and waste logs.
    Strictly validates business rules and supports alias kwargs.
    """
    if "date_str" in kwargs and not meal_date:
        meal_date = kwargs["date_str"]
    if "meal_period" in kwargs and not meal_type:
        meal_type = kwargs["meal_period"]
    if "prepared" in kwargs and quantity_prepared is None:
        quantity_prepared = int(kwargs["prepared"])
    if "served" in kwargs and quantity_served is None:
        quantity_served = int(kwargs["served"])
    if "discarded" in kwargs and discarded_quantity is None:
        discarded_quantity = int(kwargs["discarded"])

    meal_date = str(meal_date or datetime.now().strftime("%Y-%m-%d"))
    meal_type = str(meal_type or "Lunch")
    dish_name = str(dish_name or "General Dish")
    quantity_prepared = int(quantity_prepared if quantity_prepared is not None else 0)
    quantity_served = int(quantity_served if quantity_served is not None else 0)
    discarded_quantity = int(discarded_quantity if discarded_quantity is not None else 0)
    cost_per_serving = float(cost_per_serving if cost_per_serving is not None else 15.0)

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
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> Tuple[pd.DataFrame, int]:
    """Retrieve meal records matching specific search filters, returning (df, total_matching_count)."""
    if "dish_name" in kwargs and not search_dish:
        search_dish = kwargs["dish_name"]
    if "meal_period" in kwargs and not meal_type:
        meal_type = kwargs["meal_period"]

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
    limit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve operational meal records with optional filters and limit."""
    init_db(db_path)
    conditions = []
    params = []

    if start_date:
        conditions.append("meal_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("meal_date <= ?")
        params.append(end_date)
    if meal_type and meal_type not in ["All", "All Meals", ""]:
        conditions.append("meal_type = ?")
        params.append(meal_type)
    if dish_name and dish_name not in ["All", "All Dishes", ""]:
        conditions.append("dish_name = ?")
        params.append(dish_name)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        SELECT id, meal_date, meal_date AS date, meal_type, dish_name, quantity_prepared,
               quantity_served, leftover_quantity, discarded_quantity,
               usable_leftover, cost_per_serving, estimated_discarded_food_cost,
               estimated_discarded_food_cost AS discarded_food_cost_inr,
               is_synthetic, recorded_timestamp
        FROM meal_records
        {where_clause}
        ORDER BY meal_date DESC, meal_type ASC, dish_name ASC
        {limit_clause}
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


# =============================================================================
# AUTHENTICATION & ROLE-BASED ACCESS CONTROL (RBAC)
# =============================================================================

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Securely hash password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return key.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored salt and PBKDF2 hash using timing-safe comparison."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


def create_user(
    username: str,
    email: str,
    password: str,
    role: str,
    full_name: str,
    roll_no: Optional[str] = None,
    hostel: str = "Hostel-1 (Mess 1)",
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[bool, str]:
    """Create a new user account with hashed credentials and role enforcement."""
    clean_username = username.strip().lower()
    clean_email = email.strip().lower()
    role_clean = role.strip().lower()

    if role_clean not in ["manager", "student"]:
        return False, "Invalid role. Role must be 'manager' or 'student'."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    pwd_hash, salt = hash_password(password)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, salt, role, full_name, roll_no, hostel, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (clean_username, clean_email, pwd_hash, salt, role_clean, full_name.strip(), roll_no, hostel, now_str))
        conn.commit()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        conn.close()


def authenticate_user(
    username_or_email: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """
    Authenticate user by username or email.
    Returns user dictionary if credentials match and account is active, else None.
    """
    ident = username_or_email.strip().lower()
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, email, password_hash, salt, role, full_name, roll_no, hostel, is_active
            FROM users
            WHERE username = ? OR email = ?
        """, (ident, ident))
        row = cursor.fetchone()
        if not row:
            return None

        if not row["is_active"]:
            return {"error": "Account is deactivated. Contact mess administration."}

        if verify_password(password, row["password_hash"], row["salt"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "role": row["role"],
                "full_name": row["full_name"],
                "roll_no": row["roll_no"] or row["username"].upper(),
                "hostel": row["hostel"]
            }
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve user dictionary by integer user_id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, role, full_name, roll_no, hostel, is_active, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_users(
    role: Optional[str] = None,
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve DataFrame of users, optionally filtered by role, search, and active status."""
    conn = get_connection(db_path)
    try:
        where = ["1=1"]
        params = []
        if role:
            where.append("role = ?")
            params.append(role)
        if is_active is not None:
            where.append("is_active = ?")
            params.append(1 if is_active else 0)
        if search and search.strip():
            s = f"%{search.strip()}%"
            where.append("(username LIKE ? OR full_name LIKE ? OR roll_no LIKE ? OR hostel LIKE ?)")
            params.extend([s, s, s, s])

        query = f"""
            SELECT id, username, email, role, full_name, roll_no, hostel,
                   room_number, dietary_preference, is_active, created_at
            FROM users
            WHERE {' AND '.join(where)}
            ORDER BY id ASC
        """
        return pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()


def update_user_status(user_id: int, is_active: bool, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Activate or deactivate a student account."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_user_profile(
    user_id: int,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    hostel: Optional[str] = None,
    dietary_preference: Optional[str] = None,
    room_number: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> bool:
    """Allow students/managers to update permissible profile attributes, supporting partial updates."""
    if "dietary_preference" in kwargs and dietary_preference is None:
        dietary_preference = kwargs["dietary_preference"]

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        updates = []
        params = []
        if full_name is not None:
            updates.append("full_name = ?")
            params.append(full_name.strip())
        if email is not None:
            updates.append("email = ?")
            params.append(email.strip().lower())
        if hostel is not None:
            updates.append("hostel = ?")
            params.append(hostel.strip())
        if dietary_preference is not None:
            updates.append("dietary_preference = ?")
            params.append(dietary_preference.strip())
        if room_number is not None:
            updates.append("room_number = ?")
            params.append(room_number.strip())

        if not updates:
            return True

        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, tuple(params))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def seed_default_users(db_path: str = DEFAULT_DB_PATH):
    """Seed initial demonstration accounts for college demonstration & viva testing."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM users")
        if cursor.fetchone()["cnt"] == 0:
            default_accounts = [
                # Manager account
                ("manager", "manager@iiitdwd.ac.in", "Manager@123", "manager", "Mess Catering Supervisor", "STAFF-01", "Hostel-1 (Mess 1)"),
                # Student demonstration accounts
                ("24bcs042", "student@iiitdwd.ac.in", "Student@123", "student", "Varun Sakinala", "24BCS042", "Hostel-1 (Mess 1)"),
                ("24bcs015", "aravind@iiitdwd.ac.in", "Student@123", "student", "Aravind Kumar", "24BCS015", "Hostel-1 (Mess 1)"),
                ("24bcs088", "priya@iiitdwd.ac.in", "Student@123", "student", "Priya Sharma", "24BCS088", "Hostel-2 (Girls Mess)")
            ]
            for u_name, u_email, u_pwd, u_role, u_full, u_roll, u_hostel in default_accounts:
                pwd_h, s = hash_password(u_pwd)
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, salt, role, full_name, roll_no, hostel, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """, (u_name, u_email, pwd_h, s, u_role, u_full, u_roll, u_hostel, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
    finally:
        conn.close()


# =============================================================================
# MESS SETTINGS & CAMPUS ANNOUNCEMENTS
# =============================================================================

def seed_default_settings(db_path: str = DEFAULT_DB_PATH):
    """Seed initial operational parameters for IIIT Dharwad Mess 1."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM mess_settings")
        if cursor.fetchone()["cnt"] == 0:
            defaults = [
                ("mess_name", "IIIT Dharwad Hostel Mess 1"),
                ("mess_code", "MESS-01-MAIN"),
                ("breakfast_time", "07:30 AM - 09:30 AM"),
                ("lunch_time", "12:30 PM - 02:30 PM"),
                ("snacks_time", "05:00 PM - 06:15 PM"),
                ("dinner_time", "07:45 PM - 09:45 PM"),
                ("rsvp_deadline_hours", "2.0"),
                ("safety_buffer_pct", "5.0"),
                ("waste_weight_unit", "kg"),
                ("max_capacity", "550")
            ]
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for k, v in defaults:
                cursor.execute("INSERT OR REPLACE INTO mess_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)", (k, v, now_str))
            conn.commit()
    finally:
        conn.close()


def get_mess_settings(db_path: str = DEFAULT_DB_PATH) -> Dict[str, str]:
    """Retrieve key-value configuration dictionary of mess operational parameters."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT setting_key, setting_value FROM mess_settings")
        return {r["setting_key"]: r["setting_value"] for r in cursor.fetchall()}
    finally:
        conn.close()


def update_mess_setting(key: str, value: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Update a specific mess setting."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT OR REPLACE INTO mess_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)", (key, str(value), now_str))
        conn.commit()
        return True
    finally:
        conn.close()


def seed_default_announcements(db_path: str = DEFAULT_DB_PATH):
    """Seed initial mess announcements."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM announcements")
        if cursor.fetchone()["cnt"] == 0:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO announcements (title, content, priority, target_role, is_active, created_at)
                VALUES 
                ('Welcome to MessMind AI Portal', 'Hostel mess operations are now digitized! Check the daily menu, register your dining attendance, and submit recipe feedback.', 'Normal', 'all', 1, ?),
                ('Special Sunday Dinner Notice', 'Paneer Butter Masala and Gulab Jamun will be served this Sunday. Please confirm your RSVP intent 2 hours in advance.', 'High', 'student', 1, ?)
            """, (now_str, now_str))
            conn.commit()
    finally:
        conn.close()


def get_announcements(target_role: str = "all", active_only: bool = True, db_path: str = DEFAULT_DB_PATH, **kwargs: Any) -> pd.DataFrame:
    """Retrieve announcements visible to target role (e.g. 'student', 'manager', or 'all')."""
    if "only_active" in kwargs:
        active_only = kwargs["only_active"]
    conn = get_connection(db_path)
    try:
        where = ["1=1"]
        params = []
        if active_only:
            where.append("is_active = 1")
        if target_role != "manager":
            where.append("(target_role = 'all' OR target_role = ?)")
            params.append(target_role)

        query = f"SELECT id, title, content, priority, target_role, is_active, created_at FROM announcements WHERE {' AND '.join(where)} ORDER BY id DESC"
        return pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()


def add_announcement(
    title: str,
    content: str,
    priority: str = "Normal",
    target_role: str = "all",
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> bool:
    """Manager publishes an announcement."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO announcements (title, content, priority, target_role, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
        """, (title.strip(), content.strip(), priority, target_role, now_str))
        conn.commit()
        return True
    finally:
        conn.close()


def delete_announcement(announcement_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Manager deletes an announcement."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM announcements WHERE id = ?", (announcement_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# =============================================================================
# DYNAMIC MENU MANAGEMENT (MANAGER CRUD & PUBLISH CONTROL)
# =============================================================================

def seed_menu_items_from_json(db_path: str = DEFAULT_DB_PATH, menu_path: str = DEFAULT_MENU_PATH):
    """Seed initial menu items table from weekly_menu.json if table is empty."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM menu_items")
        if cursor.fetchone()["cnt"] == 0:
            menu_data = load_weekly_menu(menu_path)
            days = menu_data.get("menu", {})
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            dish_categories = {
                "Rice": "Rice Item", "Curd Rice": "Rice Item", "Veg Pulav": "Rice Item",
                "Tomato Rice": "Rice Item", "Jeera Rice": "Rice Item", "Puliogare": "Rice Item",
                "Dal": "Dal & Gravy", "Dal Fry": "Dal & Gravy", "Dal Tadka": "Dal & Gravy",
                "Dal Makhani": "Dal & Gravy", "Sambar": "Dal & Gravy", "Rasam": "Dal & Gravy",
                "Paneer Curry": "Special Curry", "Kadai Paneer": "Special Curry", "Paneer Kholapuri": "Special Curry",
                "Vegetable Curry": "Veg Sabji", "Aloo Gobi": "Veg Sabji", "Mix Veg Curry": "Veg Sabji",
                "Chapati": "Breads & Roti", "Puri": "Breads & Roti", "Paratha": "Breads & Roti", "Bhature": "Breads & Roti",
                "Idli": "Breakfast Core", "Dosa": "Breakfast Core", "Upma": "Breakfast Core", "Pongal": "Breakfast Core",
                "Samosa": "Evening Snack", "Pani Puri": "Evening Snack", "Pav Bhaji": "Evening Snack",
                "Curd": "Dairy & Desserts", "Gulab Jamun": "Dairy & Desserts", "Fruit Salad": "Dairy & Desserts"
            }
            default_costs = {
                "Paneer Curry": 25.0, "Kadai Paneer": 26.0, "Paneer Kholapuri": 25.0,
                "Vegetable Curry": 15.0, "Dal": 14.0, "Dal Fry": 15.0, "Dal Tadka": 15.0,
                "Rice": 15.0, "Curd Rice": 16.0, "Veg Pulav": 18.0, "Chapati": 12.0,
                "Idli": 15.0, "Dosa": 18.0, "Samosa": 15.0, "Curd": 10.0, "Gulab Jamun": 12.0
            }

            for day_name, meals in days.items():
                for meal_name, m_data in meals.items():
                    dishes = m_data.get("dishes", [])
                    for dish in dishes:
                        cat = dish_categories.get(dish, "Main Course")
                        cost = default_costs.get(dish, 15.0)
                        cursor.execute("""
                            INSERT INTO menu_items (day_of_week, meal_type, dish_name, category, cost_per_serving, is_published, created_by, updated_at)
                            VALUES (?, ?, ?, ?, ?, 1, 'system', ?)
                        """, (day_name, meal_name, dish, cat, cost, now_str))
            conn.commit()
    finally:
        conn.close()


def get_menu_items(
    day_of_week: Optional[str] = None,
    meal_type: Optional[str] = None,
    specific_date: Optional[str] = None,
    published_only: bool = False,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve menu items matching filter criteria with alias support."""
    if "meal_period" in kwargs and not meal_type:
        meal_type = kwargs["meal_period"]
    if "only_published" in kwargs:
        published_only = kwargs["only_published"]

    conn = get_connection(db_path)
    try:
        where = ["1=1"]
        params = []
        if day_of_week and day_of_week not in ["All", "All Days"]:
            where.append("day_of_week = ?")
            params.append(day_of_week)
        if meal_type and meal_type not in ["All", "All Meals"]:
            where.append("meal_type = ?")
            params.append(meal_type)
        if specific_date:
            where.append("(specific_date = ? OR specific_date IS NULL)")
            params.append(specific_date)
        if published_only:
            where.append("is_published = 1")

        query = f"""
            SELECT id, day_of_week, meal_type, meal_type AS meal_period, dish_name, category, cost_per_serving, is_published, specific_date, updated_at
            FROM menu_items
            WHERE {' AND '.join(where)}
            ORDER BY id ASC
        """
        return pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()


def add_menu_item(
    day_of_week: str,
    meal_type: str = "Lunch",
    dish_name: str = "",
    category: str = "Main Course",
    cost_per_serving: float = 15.0,
    is_published: int = 1,
    specific_date: Optional[str] = None,
    created_by: str = "manager",
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> Tuple[bool, str]:
    """Manager adds a dish to the official mess schedule."""
    if "meal_period" in kwargs:
        meal_type = kwargs["meal_period"]

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO menu_items (day_of_week, meal_type, dish_name, category, cost_per_serving, is_published, specific_date, created_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (day_of_week, meal_type, dish_name.strip(), category.strip(), float(cost_per_serving), int(is_published), specific_date, created_by, now_str))
        conn.commit()
        return True, f"'{dish_name}' successfully added to {day_of_week} {meal_type} menu."
    except Exception as e:
        return False, f"Failed to add menu item: {e}"
    finally:
        conn.close()


def update_menu_item(
    item_id: int,
    dish_name: Optional[str] = None,
    category: Optional[str] = None,
    cost_per_serving: Optional[float] = None,
    is_published: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
    **kwargs: Any
) -> bool:
    """Manager updates an existing menu item, supporting partial updates (e.g. only is_published)."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        updates = ["updated_at = ?"]
        params = [now_str]

        if dish_name is not None:
            updates.append("dish_name = ?")
            params.append(dish_name.strip())
        if category is not None:
            updates.append("category = ?")
            params.append(category.strip())
        if cost_per_serving is not None:
            updates.append("cost_per_serving = ?")
            params.append(float(cost_per_serving))
        if is_published is not None:
            updates.append("is_published = ?")
            params.append(int(is_published))

        params.append(item_id)
        query = f"UPDATE menu_items SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, tuple(params))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_menu_item(item_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Manager removes a menu item."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM menu_items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def publish_all_day_menu(day_of_week: str, is_published: int = 1, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Publish or unpublish all menu items for a specific weekday."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE menu_items SET is_published = ?, updated_at = ? WHERE day_of_week = ?", (is_published, now_str, day_of_week))
        conn.commit()
        return True
    finally:
        conn.close()


# =============================================================================
# STUDENT MEAL REGISTRATION (RSVP) & FEEDBACK
# =============================================================================

def submit_student_meal_intent(
    *args,
    **kwargs
) -> Tuple[bool, str]:
    """Record student dining intent for a specific meal shift. Supports positional or keyword arguments."""
    db_path = kwargs.get("db_path", DEFAULT_DB_PATH)
    
    if len(args) >= 5:
        user_id = args[0]
        student_roll = str(args[1])
        meal_date = str(args[2])
        meal_type = str(args[3])
        intent_status = str(args[4])
        dietary_pref = args[5] if len(args) > 5 else kwargs.get("dietary_pref", "Standard Veg")
    else:
        user_id = kwargs.get("user_id", 1)
        student_roll = str(kwargs.get("student_roll", kwargs.get("student_id", kwargs.get("username", "STUDENT"))))
        meal_date = str(kwargs.get("meal_date", kwargs.get("service_date", datetime.now().strftime("%Y-%m-%d"))))
        meal_type = str(kwargs.get("meal_type", kwargs.get("meal_period", "Lunch")))
        intent_status = str(kwargs.get("intent_status", kwargs.get("intent", "Attending")))
        dietary_pref = str(kwargs.get("dietary_pref", kwargs.get("dietary_preference", "Standard Veg")))

    if intent_status not in ["Attending", "Not Attending", "Not sure"]:
        return False, "Invalid attendance status."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO student_meal_intents (user_id, student_roll, meal_date, meal_type, intent_status, dietary_preference, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_roll, meal_date, meal_type) DO UPDATE SET
                intent_status = excluded.intent_status,
                dietary_preference = excluded.dietary_preference,
                updated_at = excluded.updated_at
        """, (user_id, student_roll.strip().upper(), meal_date, meal_type, intent_status, dietary_pref, now_str))

        # Also register in legacy student_registrations table for backwards compatibility
        intent_code = "Yes" if intent_status == "Attending" else ("No" if intent_status == "Not Attending" else "Not sure")
        cursor.execute("""
            INSERT INTO student_registrations (meal_date, meal_type, selected_dishes, registration_code, intention_to_attend, attendance_intent, submission_timestamp)
            VALUES (?, ?, 'Official Menu', ?, ?, ?, ?)
        """, (meal_date, meal_type, student_roll.strip().upper(), 1 if intent_status == "Attending" else 0, intent_code, now_str))

        conn.commit()
        return True, f"Intent for {meal_date} {meal_type} saved as '{intent_status}'."
    except Exception as e:
        return False, f"Error saving registration: {e}"
    finally:
        conn.close()


def get_student_meal_intents(
    *args,
    **kwargs
) -> pd.DataFrame:
    """Retrieve student meal attendance intent records. Supports student_roll or student_id, meal_date or service_date."""
    db_path = kwargs.get("db_path", DEFAULT_DB_PATH)
    student_roll = kwargs.get("student_roll", kwargs.get("student_id", kwargs.get("username", args[0] if len(args) > 0 else None)))
    meal_date = kwargs.get("meal_date", kwargs.get("service_date", args[1] if len(args) > 1 else None))
    meal_type = kwargs.get("meal_type", kwargs.get("meal_period", args[2] if len(args) > 2 else None))

    conn = get_connection(db_path)
    try:
        where = ["1=1"]
        params = []
        if student_roll:
            where.append("student_roll = ?")
            params.append(str(student_roll).strip().upper())
        if meal_date:
            where.append("meal_date = ?")
            params.append(str(meal_date))
        if meal_type:
            where.append("meal_type = ?")
            params.append(str(meal_type))

        query = f"""
            SELECT id, student_roll, meal_date, meal_date AS service_date,
                   meal_type, meal_type AS meal_period, intent_status, intent_status AS intent,
                   dietary_preference, updated_at, updated_at AS timestamp
            FROM student_meal_intents
            WHERE {' AND '.join(where)}
            ORDER BY meal_date DESC, id DESC
        """
        return pd.read_sql_query(query, conn, params=tuple(params))
    finally:
        conn.close()


def get_aggregated_intent_counts_for_date(
    meal_date: str,
    meal_type: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """Calculate attendance breakdowns ('Attending', 'Not Attending', 'Not sure') for kitchen planning."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        where = ["meal_date = ?"]
        params = [meal_date]
        if meal_type:
            where.append("meal_type = ?")
            params.append(meal_type)

        query = f"SELECT intent_status, COUNT(*) as cnt FROM student_meal_intents WHERE {' AND '.join(where)} GROUP BY intent_status"
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        counts = {"Attending": 0, "Not Attending": 0, "Not sure": 0}
        for r in rows:
            st = r["intent_status"]
            if st in counts:
                counts[st] = r["cnt"]

        # If zero records in student_meal_intents, check legacy registrations
        if sum(counts.values()) == 0:
            legacy = get_attendance_intent_counts(meal_date, meal_type or "Lunch", db_path)
            counts["Attending"] = legacy.get("yes_count", 0)
            counts["Not Attending"] = legacy.get("no_count", 0)
            counts["Not sure"] = legacy.get("not_sure_count", 0)

        total = sum(counts.values())
        return {
            "Attending": counts["Attending"],
            "Not Attending": counts["Not Attending"],
            "Not sure": counts["Not sure"],
            "attending_count": counts["Attending"],
            "not_attending_count": counts["Not Attending"],
            "not_sure_count": counts["Not sure"],
            "total_polled": total,
            "effective_predicted_turnout": counts["Attending"] + int(round(counts["Not sure"] * 0.5))
        }
    finally:
        conn.close()


def submit_student_feedback(
    *args,
    **kwargs
) -> Tuple[bool, str]:
    """Store student meal review or reason for skipping dining shift. Supports positional or keyword arguments."""
    db_path = kwargs.get("db_path", DEFAULT_DB_PATH)
    if len(args) >= 4:
        user_id = args[0]
        student_roll = str(args[1])
        meal_date = str(args[2])
        meal_type = str(args[3])
        dish_name = args[4] if len(args) > 4 else kwargs.get("dish_name", "Overall Meal")
        rating = args[5] if len(args) > 5 else kwargs.get("rating", 5)
        feedback_text = args[6] if len(args) > 6 else kwargs.get("feedback_text", kwargs.get("comments", ""))
        skip_reason = args[7] if len(args) > 7 else kwargs.get("skip_reason", "")
    else:
        user_id = kwargs.get("user_id", 1)
        student_roll = str(kwargs.get("student_roll", kwargs.get("student_id", "STUDENT")))
        meal_date = str(kwargs.get("meal_date", kwargs.get("service_date", datetime.now().strftime("%Y-%m-%d"))))
        meal_type = str(kwargs.get("meal_type", kwargs.get("meal_period", "Lunch")))
        dish_name = kwargs.get("dish_name", "Overall Meal")
        rating = int(kwargs.get("rating", 5))
        feedback_text = str(kwargs.get("feedback_text", kwargs.get("comments", "")))
        skip_reason = str(kwargs.get("skip_reason", ""))

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO student_feedback (user_id, student_roll, meal_date, meal_type, dish_name, rating, feedback_text, skip_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, student_roll.strip().upper(), meal_date, meal_type, dish_name or "Overall Meal", int(rating), feedback_text.strip(), skip_reason.strip(), now_str))
        conn.commit()
        return True, "Feedback submitted successfully! Thank you for helping improve the mess."
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        conn.close()


def get_student_feedback_records(
    student_roll: Optional[str] = None,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieve student satisfaction ratings and comments."""
    conn = get_connection(db_path)
    try:
        if student_roll:
            query = "SELECT id, student_roll, meal_date, meal_type, dish_name, rating, feedback_text, skip_reason, created_at FROM student_feedback WHERE student_roll = ? ORDER BY id DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(student_roll.strip().upper(), limit))
        else:
            query = "SELECT id, student_roll, meal_date, meal_type, dish_name, rating, feedback_text, skip_reason, created_at FROM student_feedback ORDER BY id DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


def get_feedback_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Generate aggregate satisfaction metrics for Manager Dashboard."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback FROM student_feedback")
        row = cursor.fetchone()
        avg_r = round(float(row["avg_rating"]), 2) if row and row["avg_rating"] is not None else 4.2
        total_f = row["total_feedback"] if row else 0

        # Most cited skip reasons
        cursor.execute("SELECT skip_reason, COUNT(*) as cnt FROM student_feedback WHERE skip_reason != '' GROUP BY skip_reason ORDER BY cnt DESC LIMIT 3")
        reasons = [f"{r['skip_reason']} ({r['cnt']})" for r in cursor.fetchall()]

        return {
            "average_rating": avg_r,
            "total_reviews": total_f,
            "top_skip_reasons": reasons if reasons else ["Out with Friends", "Class/Exam Preparation", "Ordered Online"]
        }
    finally:
        conn.close()


# =============================================================================
# MULTI-PLATE OPENCV BATCHES & HISTORICAL WASTE MEMORY
# =============================================================================

def save_vision_batch(
    *args,
    **kwargs
) -> int:
    """Save an entire multi-plate inspection batch into Historical Waste Memory.
    Can be called either as save_vision_batch(batch_dict, ...) or with individual kwargs.
    """
    db_path = kwargs.get("db_path", DEFAULT_DB_PATH)
    
    # If first positional argument is a dict (from analyze_plate_batch)
    if args and isinstance(args[0], dict):
        b = args[0]
        batch_code = b.get("batch_code", f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        service_date = b.get("service_date", datetime.now().strftime("%Y-%m-%d"))
        meal_period = b.get("meal_period", "Lunch")
        dish_name = b.get("dish_name", "General Dish")
        total_images = int(b.get("total_images", 0))
        valid_images = int(b.get("valid_images", total_images))
        failed_images = int(b.get("failed_images", 0))
        avg_coverage = float(b.get("average_coverage", b.get("avg_coverage", 0.0)))
        median_coverage = float(b.get("median_coverage", avg_coverage))
        min_coverage = float(b.get("min_coverage", 0.0))
        max_coverage = float(b.get("max_coverage", avg_coverage))
        high_leftover_count = int(b.get("high_leftover_count", 0))
        low_leftover_count = int(b.get("low_leftover_count", 0))
        dominant_fingerprint = str(b.get("dominant_fingerprint", "Scattered Residue"))
        actual_discarded_kg = kwargs.get("actual_discarded_kg", b.get("actual_discarded_kg"))
        notes = kwargs.get("notes", b.get("notes", ""))
        plate_results = b.get("plate_results", [])
    else:
        batch_code = kwargs.get("batch_code", args[0] if len(args) > 0 else f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        service_date = kwargs.get("service_date", args[1] if len(args) > 1 else datetime.now().strftime("%Y-%m-%d"))
        meal_period = kwargs.get("meal_period", args[2] if len(args) > 2 else "Lunch")
        dish_name = kwargs.get("dish_name", args[3] if len(args) > 3 else "General Dish")
        total_images = int(kwargs.get("total_images", args[4] if len(args) > 4 else 0))
        valid_images = int(kwargs.get("valid_images", args[5] if len(args) > 5 else total_images))
        failed_images = int(kwargs.get("failed_images", args[6] if len(args) > 6 else 0))
        avg_coverage = float(kwargs.get("avg_coverage", kwargs.get("average_coverage", args[7] if len(args) > 7 else 0.0)))
        median_coverage = float(kwargs.get("median_coverage", args[8] if len(args) > 8 else avg_coverage))
        min_coverage = float(kwargs.get("min_coverage", args[9] if len(args) > 9 else 0.0))
        max_coverage = float(kwargs.get("max_coverage", args[10] if len(args) > 10 else avg_coverage))
        high_leftover_count = int(kwargs.get("high_leftover_count", args[11] if len(args) > 11 else 0))
        low_leftover_count = int(kwargs.get("low_leftover_count", args[12] if len(args) > 12 else 0))
        dominant_fingerprint = str(kwargs.get("dominant_fingerprint", args[13] if len(args) > 13 else "Scattered Residue"))
        actual_discarded_kg = kwargs.get("actual_discarded_kg", args[14] if len(args) > 14 else None)
        notes = kwargs.get("notes", args[15] if len(args) > 15 else "")
        plate_results = kwargs.get("plate_results", args[16] if len(args) > 16 else [])

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO vision_batches (
                batch_code, service_date, meal_period, dish_name, total_images, valid_images, failed_images,
                avg_coverage, median_coverage, min_coverage, max_coverage, high_leftover_count, low_leftover_count,
                dominant_fingerprint, actual_discarded_kg, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_code, service_date, meal_period, dish_name, total_images, valid_images, failed_images,
            float(avg_coverage), float(median_coverage), float(min_coverage), float(max_coverage),
            high_leftover_count, low_leftover_count, dominant_fingerprint,
            float(actual_discarded_kg) if actual_discarded_kg is not None else None, notes, now_str
        ))
        batch_id = cursor.lastrowid

        # Insert individual plate metrics
        if plate_results:
            for p in plate_results:
                cov_pct = float(p.get("coverage_percentage", p.get("coverage_pct", 0.0)))
                c_count = int(p.get("contour_count", 0))
                
                fp = p.get("fingerprint", "Scattered Residue")
                if isinstance(fp, dict):
                    fp_name = fp.get("fingerprint_type", "Scattered Residue")
                    disp = float(fp.get("dispersion_index", 0.0))
                else:
                    fp_name = str(fp)
                    disp = float(p.get("cluster_dispersion", 0.0))
                
                strat = p.get("adaptive_strategy", "Standard Dual-Space")
                if isinstance(strat, dict):
                    strat_name = strat.get("name", "Standard Dual-Space")
                else:
                    strat_name = str(strat)
                
                q_info = p.get("quality_info", {})
                if isinstance(q_info, dict):
                    q_score = float(q_info.get("sharpness", 1.0))
                else:
                    q_score = float(p.get("quality_score", 1.0))

                cursor.execute("""
                    INSERT INTO vision_plate_results (
                        batch_id, plate_id, coverage_pct, contour_count, cluster_dispersion,
                        fingerprint, adaptive_strategy, quality_score, status, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_id,
                    p.get("plate_id", "PLATE"),
                    cov_pct,
                    c_count,
                    disp,
                    fp_name,
                    strat_name,
                    q_score,
                    p.get("status", "Success"),
                    p.get("notes", ""),
                    now_str
                ))

        conn.commit()
        return batch_id
    finally:
        conn.close()


def get_vision_batches(
    limit: int = 50,
    dish_name: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieve historical multi-plate vision inspection batches."""
    conn = get_connection(db_path)
    try:
        if dish_name:
            query = "SELECT * FROM vision_batches WHERE dish_name = ? ORDER BY id DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(dish_name, limit))
        else:
            query = "SELECT * FROM vision_batches ORDER BY id DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


def get_vision_batch_plates(batch_id: int, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve individual plate inspection records for a specific batch."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM vision_plate_results WHERE batch_id = ? ORDER BY id ASC"
        return pd.read_sql_query(query, conn, params=(batch_id,))
    finally:
        conn.close()


def get_dish_historical_vision_memory(dish_name: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve longitudinal multi-plate vision intelligence for a specific recipe."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT service_date, meal_period, total_images, avg_coverage, dominant_fingerprint, actual_discarded_kg
            FROM vision_batches
            WHERE dish_name = ?
            ORDER BY service_date DESC LIMIT 10
        """, (dish_name,))
        rows = cursor.fetchall()
        if not rows:
            return {
                "has_memory": False,
                "dish_name": dish_name,
                "batches_recorded": 0,
                "chronic_waste": False,
                "insight": f"No multi-plate vision batches recorded yet for '{dish_name}'. Run a batch scan to build memory."
            }

        coverages = [r["avg_coverage"] for r in rows]
        mean_cov = round(float(sum(coverages) / len(coverages)), 1)
        chronic = mean_cov >= 28.0

        if chronic:
            insight = f"🚨 CHRONIC SURPLUS: '{dish_name}' has consistently shown {mean_cov}% average plate leftover across {len(rows)} batches. Strong candidate to reduce ladle serving size by 15–20%."
        elif mean_cov >= 15.0:
            insight = f"⚠️ MODERATE LEFTOVERS: '{dish_name}' exhibits {mean_cov}% surface residue. Normal cafeteria scraping; monitor on subsequent shifts."
        else:
            insight = f"🟢 CLEAN CLEARANCE: '{dish_name}' exhibits minimal plate leftovers ({mean_cov}%). Recipe acceptance is high."

        history_list = []
        for r in rows:
            history_list.append({
                "Date": r["service_date"],
                "Shift": r["meal_period"],
                "Plates Analyzed": r["total_images"],
                "Avg Leftover %": f"{r['avg_coverage']:.1f}%",
                "Waste Fingerprint": r["dominant_fingerprint"],
                "Weighed Discarded (kg)": f"{r['actual_discarded_kg']:.1f} kg" if r["actual_discarded_kg"] is not None else "Not weighed"
            })

        return {
            "has_memory": True,
            "dish_name": dish_name,
            "batches_recorded": len(rows),
            "mean_coverage_pct": mean_cov,
            "chronic_waste": chronic,
            "insight": insight,
            "history": history_list
        }
    finally:
        conn.close()


def record_actual_discarded_weight(
    dish_name: str,
    service_date: str,
    meal_period: str,
    weight_kg: float,
    unit: str = "kg",
    batch_code: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Record physical weighing scale measurement of discarded food."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        if batch_code:
            cursor.execute("UPDATE vision_batches SET actual_discarded_kg = ? WHERE batch_code = ?", (float(weight_kg), batch_code))
        # Also update meal_records if matching date, meal, and dish exists
        cost_row = cursor.execute("SELECT cost_per_serving FROM meal_records WHERE meal_date = ? AND meal_type = ? AND dish_name = ?", (service_date, meal_period, dish_name)).fetchone()
        cost_val = cost_row["cost_per_serving"] if cost_row else 15.0
        est_servings = int(round(weight_kg / 0.22))
        est_cost = round(est_servings * cost_val, 2)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE meal_records 
            SET discarded_quantity = ?, estimated_discarded_food_cost = ?, recorded_timestamp = ?
            WHERE meal_date = ? AND meal_type = ? AND dish_name = ?
        """, (est_servings, est_cost, now_str, service_date, meal_period, dish_name))
        conn.commit()
        return True
    finally:
        conn.close()


# =============================================================================
# AUDIT LOGGING (ACCOUNTABILITY FOR SENSITIVE MANAGER ACTIONS)
# =============================================================================

def log_audit_event(
    *args,
    **kwargs
) -> bool:
    """Record security and administrative events into audit_logs table. Supports flexible argument orders."""
    db_path = kwargs.get("db_path", DEFAULT_DB_PATH)
    if len(args) == 5:
        username, role, action, entity, details = args
    elif len(args) == 3:
        # e.g. log_audit_event("USER_LOGIN", username, details)
        action, username, details = args
        role = "manager" if username == "manager" else ("student" if username != "system" else "system")
        entity = "APPLICATION"
    elif len(args) == 4:
        action, username, entity, details = args
        role = "manager" if username == "manager" else "student"
    else:
        username = kwargs.get("username", "system")
        role = kwargs.get("role", "system")
        action = kwargs.get("action", "EVENT")
        entity = kwargs.get("entity", "APPLICATION")
        details = kwargs.get("details", "")

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO audit_logs (username, role, action, entity, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(username), str(role), str(action), str(entity), str(details), now_str))
        conn.commit()
        return True
    except Exception as e:
        print(f"Audit log error: {e}")
        return False
    finally:
        conn.close()


def get_audit_logs(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve chronological audit trail of administrative operations."""
    conn = get_connection(db_path)
    try:
        query = "SELECT id, timestamp, username, role, action, entity, details FROM audit_logs ORDER BY id DESC LIMIT ?"
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized with full RBAC, multi-plate vision memory, and student portals.")
    print("Default accounts:")
    print("  Manager: manager / Manager@123")
    print("  Student: 24bcs042 / Student@123")
