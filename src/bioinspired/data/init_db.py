"""
Database initialization script.
Run this script to create the database tables.
"""

import sys

from .database import create_tables, engine

def init_database():
    """Initialize the database with all tables."""
    print("Creating database tables...")
    try:
        create_tables()
        print("[OK] Database tables created successfully!")
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"[INFO] Created tables: {', '.join(tables)}")
        
    except Exception as e:
        print(f"[ERROR] Error creating database tables: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
