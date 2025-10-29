"""
Add emergency contact columns to users table (SQLite version)
"""
from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        # For SQLite, we need to check columns differently
        result = db.session.execute(text("PRAGMA table_info(users)"))
        columns = [row[1] for row in result]

        print("Current columns in users table:", columns)

        # Add emergency_contact_phone if it doesn't exist
        if 'emergency_contact_phone' not in columns:
            db.session.execute(text("""
                ALTER TABLE users
                ADD COLUMN emergency_contact_phone VARCHAR(20)
            """))
            print("[OK] Added column: emergency_contact_phone")
        else:
            print("[INFO] Column emergency_contact_phone already exists")

        # Add emergency_contact_name if it doesn't exist
        if 'emergency_contact_name' not in columns:
            db.session.execute(text("""
                ALTER TABLE users
                ADD COLUMN emergency_contact_name VARCHAR(100)
            """))
            print("[OK] Added column: emergency_contact_name")
        else:
            print("[INFO] Column emergency_contact_name already exists")

        db.session.commit()
        print("\n[SUCCESS] Database migration completed successfully!")

    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Error during migration: {e}")
        raise
