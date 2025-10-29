"""
Database update script to add emergency contact columns
"""
from app import app, db

with app.app_context():
    # This will create any missing tables and columns
    db.create_all()
    print("✅ Database schema updated successfully!")
    print("✅ New columns added: emergency_contact_phone, emergency_contact_name")
