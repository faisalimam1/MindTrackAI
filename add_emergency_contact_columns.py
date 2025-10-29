"""
Add emergency contact columns to users table
"""
from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        # Check if columns already exist
        result = db.session.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='users' AND column_name IN ('emergency_contact_phone', 'emergency_contact_name')
        """))
        existing_columns = [row[0] for row in result]

        # Add emergency_contact_phone if it doesn't exist
        if 'emergency_contact_phone' not in existing_columns:
            db.session.execute(text("""
                ALTER TABLE users
                ADD COLUMN emergency_contact_phone VARCHAR(20)
            """))
            print("✅ Added column: emergency_contact_phone")
        else:
            print("ℹ️  Column emergency_contact_phone already exists")

        # Add emergency_contact_name if it doesn't exist
        if 'emergency_contact_name' not in existing_columns:
            db.session.execute(text("""
                ALTER TABLE users
                ADD COLUMN emergency_contact_name VARCHAR(100)
            """))
            print("✅ Added column: emergency_contact_name")
        else:
            print("ℹ️  Column emergency_contact_name already exists")

        db.session.commit()
        print("\n✅ Database migration completed successfully!")

    except Exception as e:
        db.session.rollback()
        print(f"❌ Error during migration: {e}")
        raise
