"""
Migration script to add phone_number column to users table (PostgreSQL)
"""
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/mindmate')

def add_phone_column():
    """Add phone_number column to users table if it doesn't exist"""
    try:
        # Connect to the database
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Check if phone_number column already exists
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='users' AND column_name='phone_number'
        """)

        if cursor.fetchone():
            print("[OK] phone_number column already exists")
            conn.close()
            return

        # Add the column
        print("Adding phone_number column to users table...")
        cursor.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
        conn.commit()

        print("[OK] Successfully added phone_number column to users table")

    except psycopg2.Error as e:
        print(f"[ERROR] Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    try:
        add_phone_column()
    except Exception as e:
        print(f"[ERROR] Failed to add column: {e}")
