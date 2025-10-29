"""
Migration script using Flask app context
"""
from app import app, db

def add_phone_column():
    """Add phone_number column to users table if it doesn't exist"""
    with app.app_context():
        try:
            # Execute raw SQL to add the column
            with db.engine.connect() as conn:
                # For SQLite, check if column exists using PRAGMA
                result = conn.execute(db.text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result]

                if 'phone_number' in columns:
                    print("[OK] phone_number column already exists")
                    return

                # Add the column
                print("Adding phone_number column to users table...")
                conn.execute(db.text("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)"))
                conn.commit()

                print("[OK] Successfully added phone_number column to users table")

        except Exception as e:
            print(f"[ERROR] Failed to add column: {e}")

if __name__ == '__main__':
    add_phone_column()
