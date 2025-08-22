from typing import Optional, Dict
from pydantic_settings import BaseSettings
import os
import hashlib

class AuthSettings(BaseSettings):
    CONNECT_BASE_URL: str = "https://connect.bracu.ac.bd"
    CONNECT_LOGIN_URL: str = "https://connect.bracu.ac.bd/login"
    CONNECT_AUTH_INIT_URL: str = "https://connect.bracu.ac.bd/api/auth/init"
    CONNECT_USER_INFO_URL: str = "https://connect.bracu.ac.bd/api/adv/v1/advising/student/info"
    CONNECT_SCHEDULE_URL: str = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections/student/{student_id}/schedules"
    FRONTEND_URL: str = "https://connapi.vercel.app"
    BACKEND_URL: str = "https://connapi.vercel.app"
    SECRET_PASSWORD: str = "default-not-used"
    
    # Admin authentication settings
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH: str = os.getenv("ADMIN_PASSWORD_HASH", hashlib.sha256("admin123".encode()).hexdigest())
    ADMIN_SESSION_DURATION: int = 3600  # 1 hour in seconds
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-jwt-tokens")
    
    # Admin permissions and features
    ADMIN_FEATURES: Dict[str, bool] = {
        "manage_users": True,
        "view_all_sessions": True,
        "clear_cache": True,
        "system_settings": True
    }

settings = AuthSettings() 

def verify_admin_password(password: str) -> bool:
    """Verify if the provided password matches the admin password hash"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == settings.ADMIN_PASSWORD_HASH