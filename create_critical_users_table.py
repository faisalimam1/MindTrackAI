"""
Database migration script to create critical_users table
Run this script to add the critical_users table to your existing database
"""
from app import app, db
from models import CriticalUser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_critical_users_table():
    """Create the critical_users table"""
    try:
        with app.app_context():
            logger.info("Creating critical_users table...")

            # Create all tables (will only create missing ones)
            db.create_all()

            logger.info("✓ critical_users table created successfully!")

            # Verify the table was created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()

            if 'critical_users' in tables:
                logger.info("✓ Verified: critical_users table exists in database")
                columns = [col['name'] for col in inspector.get_columns('critical_users')]
                logger.info(f"  Table columns: {', '.join(columns)}")
            else:
                logger.error("✗ Error: critical_users table was not created")

            return True

    except Exception as e:
        logger.error(f"Error creating critical_users table: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("DATABASE MIGRATION: Create critical_users table")
    print("=" * 70)
    print()

    success = create_critical_users_table()

    print()
    if success:
        print("✓ Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Restart your Flask application")
        print("2. Log in as admin (username: admin)")
        print("3. Navigate to /admin to view the admin dashboard")
        print("4. Test by completing an assessment with a score ≥ 80%")
    else:
        print("✗ Migration failed. Please check the errors above.")

    print("=" * 70)
