#!/usr/bin/env python3
"""
HTTPS version of the MindTrackAI application for camera/microphone permissions
"""

import os
import ssl
from app import app, db
from models import User, JournalEntry, MoodEntry, Task, Goal
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize the database with tables"""
    print("Initializing database...")
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            print("Database tables created successfully")
            
            # Check if admin user exists
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                # Create admin user
                admin_user = User(
                    username='admin',
                    email='admin@ai-mental-health.com',
                    password_hash=generate_password_hash('admin123')
                )
                db.session.add(admin_user)
                db.session.commit()
                print("Admin user created (username: admin, password: admin123)")
            
            print("Database initialization complete!")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    return True

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MindTrackAI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate and key to files
        with open("cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print("✅ Self-signed certificate created successfully!")
        return True
        
    except ImportError:
        print("❌ cryptography library not found. Installing...")
        os.system("pip install cryptography")
        return create_self_signed_cert()
    except Exception as e:
        print(f"❌ Error creating certificate: {e}")
        return False

def main():
    """Main startup function with HTTPS support"""
    print("AI Powered Mental Health Prediction and Personalized Assistance System (HTTPS)")
    print("=" * 70)
    
    # Initialize database
    if not init_database():
        print("\n❌ Database initialization failed.")
        return
    
    # Create self-signed certificate
    if not create_self_signed_cert():
        print("\n❌ Certificate creation failed.")
        return
    
    print("\nStarting AI Powered Mental Health Prediction and Personalized Assistance System...")
    print("🌐 HTTPS Web application will be available at: https://localhost:5000")
    print("🔐 Admin login: admin / admin123")
    print("📹 Camera/Microphone permissions should work with HTTPS")
    print("\n⚠️  Note: You may see a security warning - click 'Advanced' and 'Proceed to localhost'")
    print("\nTo stop the application, press Ctrl+C")
    print("=" * 70)
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    
    # Start the Flask application with HTTPS
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=context)
    except KeyboardInterrupt:
        print("\n\nAI Powered Mental Health Prediction and Personalized Assistance System stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

if __name__ == '__main__':
    main()
