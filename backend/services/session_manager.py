"""
session_manager.py — JWT session management with secure token handling.

This module implements the SessionManager class that handles:
- JWT token generation and validation with RS256 signing
- Session lifecycle management with proper expiration
- Token refresh automation with secure refresh tokens
- Concurrent session support with cleanup

Requirements: 1.4, 2.1, 2.2, 2.3
"""

import jwt
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import uuid

from pydantic import BaseModel, Field


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Configuration for JWT session management."""
    jwt_secret_key: str
    jwt_algorithm: str = "RS256"
    token_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    max_concurrent_sessions: int = 5
    issuer: str = "ai-recommendation-agent"
    audience: str = "ai-recommendation-users"


class JWTToken(BaseModel):
    """JWT token response with access and refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds until expiration
    expires_at: datetime  # absolute expiration time
    session_id: str


class SessionContext(BaseModel):
    """Validated session context extracted from JWT token."""
    user_id: str
    session_id: str
    email: str
    display_name: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    privacy_settings: Dict[str, Any] = Field(default_factory=dict)
    issued_at: datetime
    expires_at: datetime
    is_valid: bool = True


class RefreshTokenData(BaseModel):
    """Refresh token data structure."""
    token_hash: str
    user_id: str
    session_id: str
    expires_at: datetime
    created_at: datetime
    last_used: datetime


# ── Session Manager ────────────────────────────────────────────────────────────

class SessionManager:
    """
    JWT session manager with secure token handling.
    
    This service handles:
    1. JWT token generation with RS256 signing
    2. Token validation with signature verification
    3. Automatic token refresh with secure refresh tokens
    4. Session lifecycle management and cleanup
    5. Concurrent session support with limits
    """
    
    def __init__(self, config: SessionConfig):
        """
        Initialize the session manager.
        
        Args:
            config: Session management configuration
        """
        self.config = config
        self._private_key = None
        self._public_key = None
        self._refresh_tokens: Dict[str, RefreshTokenData] = {}  # In-memory storage for now
        self._active_sessions: Dict[str, SessionContext] = {}  # Session cache
        
        # Generate or load RSA key pair for JWT signing
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize RSA or EC key pair for JWT signing."""
        if self.config.jwt_algorithm in ["RS256", "RS384", "RS512"]:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            self._private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            self._public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        elif self.config.jwt_algorithm in ["ES256", "ES384", "ES512"]:
            # Generate EC key pair
            from cryptography.hazmat.primitives.asymmetric import ec
            
            # Choose curve based on algorithm
            if self.config.jwt_algorithm == "ES256":
                curve = ec.SECP256R1()
            elif self.config.jwt_algorithm == "ES384":
                curve = ec.SECP384R1()
            else:  # ES512
                curve = ec.SECP521R1()
            
            private_key = ec.generate_private_key(curve, default_backend())
            
            self._private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            self._public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            # For HMAC algorithms, use the secret key directly
            self._private_key = self.config.jwt_secret_key.encode('utf-8')
            self._public_key = self._private_key
    
    def create_session(self, user_profile) -> JWTToken:
        """
        Create a new user session with JWT tokens.
        
        Args:
            user_profile: UserProfile object containing user information
            
        Returns:
            JWTToken containing access token, refresh token, and metadata
            
        Requirements: 1.4, 2.1, 2.2
        """
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        
        # Calculate expiration times
        access_expires_at = now + timedelta(hours=self.config.token_expiry_hours)
        refresh_expires_at = now + timedelta(days=self.config.refresh_token_expiry_days)
        
        # Create JWT payload
        payload = {
            "sub": user_profile.user_id,  # Subject (user ID)
            "iat": int(now.timestamp()),  # Issued at
            "exp": int(access_expires_at.timestamp()),  # Expiration
            "iss": self.config.issuer,  # Issuer
            "aud": self.config.audience,  # Audience
            "session_id": session_id,
            "email": user_profile.email,
            "display_name": user_profile.display_name,
            "preferences": user_profile.preferences,
            "privacy_settings": user_profile.privacy_settings,
            "jti": str(uuid.uuid4())  # JWT ID for uniqueness
        }
        
        # Generate access token
        access_token = jwt.encode(
            payload,
            self._private_key,
            algorithm=self.config.jwt_algorithm
        )
        
        # Generate secure refresh token
        refresh_token = self._generate_refresh_token(
            user_profile.user_id,
            session_id,
            refresh_expires_at
        )
        
        # Create session context for caching
        session_context = SessionContext(
            user_id=user_profile.user_id,
            session_id=session_id,
            email=user_profile.email,
            display_name=user_profile.display_name,
            preferences=user_profile.preferences,
            privacy_settings=user_profile.privacy_settings,
            issued_at=now,
            expires_at=access_expires_at,
            is_valid=True
        )
        
        # Cache the session
        self._active_sessions[session_id] = session_context
        
        # Clean up old sessions if user exceeds concurrent session limit
        self._enforce_session_limits(user_profile.user_id)
        
        return JWTToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int((access_expires_at - now).total_seconds()),
            expires_at=access_expires_at,
            session_id=session_id
        )
    
    def validate_session(self, token: str) -> Optional[SessionContext]:
        """
        Validate JWT token and extract session context.
        
        Args:
            token: JWT access token to validate
            
        Returns:
            SessionContext if token is valid, None otherwise
            
        Requirements: 1.5, 2.1
        """
        try:
            # Decode and validate JWT token
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[self.config.jwt_algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # Extract session information
            session_id = payload.get("session_id")
            if not session_id:
                return None
            
            # Check if session is in cache and still valid
            if session_id in self._active_sessions:
                session_context = self._active_sessions[session_id]
                if session_context.is_valid and session_context.expires_at > datetime.now(timezone.utc):
                    return session_context
                else:
                    # Remove invalid session from cache
                    del self._active_sessions[session_id]
                    return None
            
            # Check if session was explicitly invalidated by checking if refresh tokens exist
            # If no refresh tokens exist for this session, it was likely invalidated
            session_has_refresh_token = any(
                refresh_data.session_id == session_id 
                for refresh_data in self._refresh_tokens.values()
            )
            
            if not session_has_refresh_token:
                # Session was invalidated, don't recreate it
                return None
            
            # Create session context from token payload
            session_context = SessionContext(
                user_id=payload["sub"],
                session_id=session_id,
                email=payload.get("email", ""),
                display_name=payload.get("display_name", ""),
                preferences=payload.get("preferences", {}),
                privacy_settings=payload.get("privacy_settings", {}),
                issued_at=datetime.fromtimestamp(payload["iat"], timezone.utc),
                expires_at=datetime.fromtimestamp(payload["exp"], timezone.utc),
                is_valid=True
            )
            
            # Cache the session
            self._active_sessions[session_id] = session_context
            
            return session_context
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    def refresh_session(self, refresh_token: str) -> Optional[JWTToken]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token to validate and use
            
        Returns:
            New JWTToken if refresh is successful, None otherwise
            
        Requirements: 2.3
        """
        try:
            # Hash the refresh token to find stored data
            token_hash = self._hash_refresh_token(refresh_token)
            
            # Find refresh token data
            refresh_data = None
            for stored_hash, data in self._refresh_tokens.items():
                if stored_hash == token_hash:
                    refresh_data = data
                    break
            
            if not refresh_data:
                return None
            
            # Check if refresh token is expired
            if refresh_data.expires_at <= datetime.now(timezone.utc):
                # Remove expired refresh token
                del self._refresh_tokens[token_hash]
                return None
            
            # Update last used timestamp
            refresh_data.last_used = datetime.now(timezone.utc)
            
            # Get user profile information (in a real implementation, this would come from database)
            # For now, create a minimal profile from session data
            from backend.services.authentication import UserProfile
            
            # Find the session context to get user information
            session_context = self._active_sessions.get(refresh_data.session_id)
            if not session_context:
                return None
            
            # Create a minimal user profile for token generation
            user_profile = UserProfile(
                user_id=refresh_data.user_id,
                google_id="",  # Not needed for refresh
                email=session_context.email,
                display_name=session_context.display_name,
                preferences=session_context.preferences,
                privacy_settings=session_context.privacy_settings,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_active=True
            )
            
            # Generate new access token (reuse existing session_id)
            now = datetime.now(timezone.utc)
            access_expires_at = now + timedelta(hours=self.config.token_expiry_hours)
            
            payload = {
                "sub": user_profile.user_id,
                "iat": int(now.timestamp()),
                "exp": int(access_expires_at.timestamp()),
                "iss": self.config.issuer,
                "aud": self.config.audience,
                "session_id": refresh_data.session_id,
                "email": user_profile.email,
                "display_name": user_profile.display_name,
                "preferences": user_profile.preferences,
                "privacy_settings": user_profile.privacy_settings,
                "jti": str(uuid.uuid4())
            }
            
            access_token = jwt.encode(
                payload,
                self._private_key,
                algorithm=self.config.jwt_algorithm
            )
            
            # Update session context in cache
            session_context.expires_at = access_expires_at
            session_context.issued_at = now
            
            return JWTToken(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep the same refresh token
                expires_in=int((access_expires_at - now).total_seconds()),
                expires_at=access_expires_at,
                session_id=refresh_data.session_id
            )
            
        except Exception:
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a specific session.
        
        Args:
            session_id: Session identifier to invalidate
            
        Returns:
            True if session was invalidated successfully
        """
        try:
            # Remove from active sessions cache
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            # Remove associated refresh tokens
            tokens_to_remove = []
            for token_hash, refresh_data in self._refresh_tokens.items():
                if refresh_data.session_id == session_id:
                    tokens_to_remove.append(token_hash)
            
            for token_hash in tokens_to_remove:
                del self._refresh_tokens[token_hash]
            
            return True
            
        except Exception:
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions and refresh tokens.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now(timezone.utc)
        cleaned_count = 0
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, session_context in self._active_sessions.items():
            if session_context.expires_at <= now or not session_context.is_valid:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._active_sessions[session_id]
            cleaned_count += 1
        
        # Clean up expired refresh tokens
        expired_tokens = []
        for token_hash, refresh_data in self._refresh_tokens.items():
            if refresh_data.expires_at <= now:
                expired_tokens.append(token_hash)
        
        for token_hash in expired_tokens:
            del self._refresh_tokens[token_hash]
            cleaned_count += 1
        
        return cleaned_count
    
    def get_user_sessions(self, user_id: str) -> List[SessionContext]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active SessionContext objects for the user
        """
        user_sessions = []
        for session_context in self._active_sessions.values():
            if session_context.user_id == user_id and session_context.is_valid:
                user_sessions.append(session_context)
        
        return user_sessions
    
    def _generate_refresh_token(self, user_id: str, session_id: str, expires_at: datetime) -> str:
        """
        Generate a secure refresh token.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            expires_at: Token expiration time
            
        Returns:
            Secure refresh token string
        """
        # Generate cryptographically secure random token
        token = secrets.token_urlsafe(32)
        
        # Hash the token for storage
        token_hash = self._hash_refresh_token(token)
        
        # Store refresh token data
        refresh_data = RefreshTokenData(
            token_hash=token_hash,
            user_id=user_id,
            session_id=session_id,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc)
        )
        
        self._refresh_tokens[token_hash] = refresh_data
        
        return token
    
    def _hash_refresh_token(self, token: str) -> str:
        """
        Hash refresh token for secure storage.
        
        Args:
            token: Raw refresh token
            
        Returns:
            Hashed token for storage
        """
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    def _enforce_session_limits(self, user_id: str):
        """
        Enforce concurrent session limits for a user.
        
        Args:
            user_id: User identifier
        """
        user_sessions = self.get_user_sessions(user_id)
        
        if len(user_sessions) > self.config.max_concurrent_sessions:
            # Sort by issued_at to find oldest sessions
            user_sessions.sort(key=lambda s: s.issued_at)
            
            # Remove oldest sessions to stay within limit
            sessions_to_remove = len(user_sessions) - self.config.max_concurrent_sessions
            for i in range(sessions_to_remove):
                session_to_remove = user_sessions[i]
                self.invalidate_session(session_to_remove.session_id)
    
    def should_refresh_token(self, session_context: SessionContext) -> bool:
        """
        Check if a token should be automatically refreshed.
        
        Args:
            session_context: Current session context
            
        Returns:
            True if token should be refreshed (expires within 2 hours)
            
        Requirements: 2.3
        """
        if not session_context or not session_context.is_valid:
            return False
        
        now = datetime.now(timezone.utc)
        time_until_expiry = session_context.expires_at - now
        
        # Refresh if token expires within 2 hours
        return time_until_expiry <= timedelta(hours=2)


# ── Utility Functions ──────────────────────────────────────────────────────────

def create_session_manager(
    jwt_secret_key: str,
    jwt_algorithm: str = "RS256",
    token_expiry_hours: int = 24,
    refresh_token_expiry_days: int = 30,
    max_concurrent_sessions: int = 5
) -> SessionManager:
    """
    Factory function to create SessionManager with configuration.
    
    Args:
        jwt_secret_key: JWT signing secret key
        jwt_algorithm: JWT signing algorithm (RS256, ES256, HS256)
        token_expiry_hours: Access token expiry in hours
        refresh_token_expiry_days: Refresh token expiry in days
        max_concurrent_sessions: Maximum concurrent sessions per user
        
    Returns:
        Configured SessionManager instance
    """
    config = SessionConfig(
        jwt_secret_key=jwt_secret_key,
        jwt_algorithm=jwt_algorithm,
        token_expiry_hours=token_expiry_hours,
        refresh_token_expiry_days=refresh_token_expiry_days,
        max_concurrent_sessions=max_concurrent_sessions
    )
    
    return SessionManager(config)