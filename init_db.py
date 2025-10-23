#!/usr/bin/env python3
"""
Database initialization script for MindTrackAI
"""

import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///mindtrack_ai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Import all models to ensure they're registered

def init_database():
    """Initialize the database with all tables."""
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Check if tables exist
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"📊 Created tables: {', '.join(tables)}")
            
            return True
    except Exception as e:
        print(f"❌ Error creating database: {str(e)}")
        return False

if __name__ == "__main__":
    success = init_database()
    if success:
        print("🎉 Database initialization complete!")
        sys.exit(0)
    else:
        print("💥 Database initialization failed!")
        sys.exit(1)
