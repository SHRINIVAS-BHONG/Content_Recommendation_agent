"""
authentication.py — OAuth 2.0 Google authentication service.

This module implements the AuthenticationService class that handles:
- OAuth 2.0 flow with Google using PKCE
- User profile creation and management
- Token exchange and validation
- Security policy enforcement
- JWT session token creation

Requirements: 1.1, 1.2, 1.3, 1.4
"""

import secrets
import hashlib
import base64
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import httpx
from urllib.parse import urlencode, parse_qs

from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.session_manager import JWTToken


# ── Data Models ────────────────────────────────────────────────────────────────

@dataclass
class AuthConfig:
    """Configuration for OAuth 2.0 authentication."""
    google_client_id: str
    google_client_secret: str
    redirect_uri: str
    jwt_secret_key: str
    jwt_algorithm: str = "RS256"
    token_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    max_concurrent_sessions: int = 5


class OAuthRedirectResponse(BaseModel):
    """Response containing OAuth redirect URL and state."""
    redirect_url: str
    state: str
    code_verifier: str  # Store this securely for PKCE


class GoogleUserInfo(BaseModel):
    """Google user information from OAuth response."""
    model_config = {"populate_by_name": True}  # Allow both 'id' and 'sub' field names
    
    id: str = Field(alias="sub")
    email: str
    name: str
    picture: Optional[str] = None
    email_verified: bool = False


class UserProfile(BaseModel):
    """User profile data structure."""
    user_id: str
    google_id: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    privacy_settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class AuthResult(BaseModel):
    """Result of authentication process."""
    success: bool
    user_profile: Optional[UserProfile] = None
    jwt_tokens: Optional[Any] = None  # Use Any to avoid circular import
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class TokenResponse(BaseModel):
    """OAuth token response."""
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str = "Bearer"


# ── Authentication Service ─────────────────────────────────────────────────────

class AuthenticationService:
    """
    OAuth 2.0 Google authentication service with PKCE support.
    
    This service handles the complete OAuth flow:
    1. Generate OAuth redirect URL with PKCE
    2. Handle OAuth callback and exchange authorization code for tokens
    3. Retrieve user information from Google
    4. Create or update user profiles
    5. Validate credentials and manage sessions
    """
    
    def __init__(self, config: AuthConfig):
        """
        Initialize the authentication service.
        
        Args:
            config: Authentication configuration including Google OAuth credentials
        """
        self.config = config
        self.google_oauth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.google_token_url = "https://oauth2.googleapis.com/token"
        self.google_userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        
        # OAuth 2.0 scopes for Google
        self.scopes = [
            "openid",
            "email", 
            "profile"
        ]
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def _generate_state(self) -> str:
        """Generate secure random state parameter."""
        return secrets.token_urlsafe(32)
    
    def initiate_oauth(self) -> OAuthRedirectResponse:
        """
        Initiate OAuth 2.0 flow with Google.
        
        Generates a secure OAuth redirect URL with PKCE support.
        The client should redirect the user to this URL and store the
        code_verifier securely for the callback.
        
        Returns:
            OAuthRedirectResponse containing redirect URL, state, and code verifier
            
        Requirements: 1.1
        """
        # Generate PKCE parameters
        code_verifier = self._generate_code_verifier()
        code_challenge = self._generate_code_challenge(code_verifier)
        state = self._generate_state()
        
        # Build OAuth parameters
        params = {
            "client_id": self.config.google_client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.scopes),
            "response_type": "code",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",  # Request refresh token
            "prompt": "consent"  # Force consent to get refresh token
        }
        
        redirect_url = f"{self.google_oauth_url}?{urlencode(params)}"
        
        return OAuthRedirectResponse(
            redirect_url=redirect_url,
            state=state,
            code_verifier=code_verifier
        )
    
    async def handle_oauth_callback(
        self, 
        code: str, 
        state: str, 
        code_verifier: str,
        expected_state: str
    ) -> AuthResult:
        """
        Handle OAuth callback and exchange authorization code for tokens.
        
        Args:
            code: Authorization code from Google
            state: State parameter from OAuth response
            code_verifier: PKCE code verifier stored during initiation
            expected_state: Expected state value for validation
            
        Returns:
            AuthResult with user profile or error information
            
        Requirements: 1.2, 1.3
        """
        try:
            # Validate state parameter
            if state != expected_state:
                return AuthResult(
                    success=False,
                    error_code="INVALID_STATE",
                    error_message="Invalid state parameter"
                )
            
            # Exchange authorization code for tokens
            token_response = await self._exchange_code_for_tokens(code, code_verifier)
            if not token_response:
                return AuthResult(
                    success=False,
                    error_code="TOKEN_EXCHANGE_FAILED",
                    error_message="Failed to exchange authorization code for tokens"
                )
            
            # Get user information from Google
            user_info = await self._get_user_info(token_response.access_token)
            if not user_info:
                return AuthResult(
                    success=False,
                    error_code="USER_INFO_FAILED",
                    error_message="Failed to retrieve user information from Google"
                )
            
            # Create or update user profile
            user_profile = await self._create_or_update_user_profile(
                user_info, 
                token_response.refresh_token
            )
            
            # Create JWT session tokens
            jwt_tokens = self.create_jwt_session(user_profile)
            
            return AuthResult(
                success=True,
                user_profile=user_profile,
                jwt_tokens=jwt_tokens  # Add JWT tokens to result
            )
            
        except Exception as e:
            return AuthResult(
                success=False,
                error_code="OAUTH_CALLBACK_ERROR",
                error_message=f"OAuth callback processing failed: {str(e)}"
            )
    
    async def _exchange_code_for_tokens(self, code: str, code_verifier: str) -> Optional[TokenResponse]:
        """
        Exchange authorization code for access and refresh tokens.
        
        Args:
            code: Authorization code from Google
            code_verifier: PKCE code verifier
            
        Returns:
            TokenResponse or None if exchange fails
        """
        token_data = {
            "client_id": self.config.google_client_id,
            "client_secret": self.config.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.config.redirect_uri,
            "code_verifier": code_verifier
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.google_token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()
                
                token_json = response.json()
                return TokenResponse(
                    access_token=token_json["access_token"],
                    refresh_token=token_json.get("refresh_token"),
                    expires_in=token_json.get("expires_in", 3600),
                    token_type=token_json.get("token_type", "Bearer")
                )
                
            except httpx.HTTPError as e:
                print(f"Token exchange failed: {e}")
                return None
    
    async def _get_user_info(self, access_token: str) -> Optional[GoogleUserInfo]:
        """
        Retrieve user information from Google using access token.
        
        Args:
            access_token: Google OAuth access token
            
        Returns:
            GoogleUserInfo or None if request fails
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.google_userinfo_url, headers=headers)
                response.raise_for_status()
                
                user_data = response.json()
                return GoogleUserInfo(**user_data)
                
            except httpx.HTTPError as e:
                print(f"User info retrieval failed: {e}")
                return None
    
    async def _create_or_update_user_profile(
        self, 
        user_info: GoogleUserInfo, 
        refresh_token: Optional[str]
    ) -> UserProfile:
        """
        Create new user profile or update existing one.
        
        Args:
            user_info: Google user information
            refresh_token: OAuth refresh token
            
        Returns:
            UserProfile object
            
        Requirements: 1.3
        """
        # For now, create a basic user profile
        # In a real implementation, this would interact with the database
        now = datetime.now(timezone.utc)
        
        user_profile = UserProfile(
            user_id=f"user_{user_info.id}",
            google_id=user_info.id,
            email=user_info.email,
            display_name=user_info.name,
            avatar_url=user_info.picture,
            preferences={
                "categories": ["anime", "manga"],
                "privacy_level": "standard"
            },
            privacy_settings={
                "data_sharing": True,
                "analytics": True
            },
            created_at=now,
            updated_at=now,
            last_login=now,
            is_active=True
        )
        
        # TODO: Store refresh token securely in database
        # This would be handled by the database adapter
        
        return user_profile
    
    def create_jwt_session(self, user_profile: UserProfile) -> 'JWTToken':
        """
        Create JWT session tokens for authenticated user.
        
        Args:
            user_profile: User profile to create session for
            
        Returns:
            JWTToken containing access and refresh tokens
            
        Requirements: 1.4, 2.1, 2.2
        """
        # Import here to avoid circular imports
        from backend.services.session_manager import create_session_manager
        
        # Create session manager with same config
        # For production, this should be a singleton or injected dependency
        if not hasattr(self, '_session_manager'):
            self._session_manager = create_session_manager(
                jwt_secret_key=self.config.jwt_secret_key,
                jwt_algorithm=self.config.jwt_algorithm,
                token_expiry_hours=self.config.token_expiry_hours,
                refresh_token_expiry_days=self.config.refresh_token_expiry_days,
                max_concurrent_sessions=self.config.max_concurrent_sessions
            )
        
        # Create session and return JWT tokens
        return self._session_manager.create_session(user_profile)
    
    async def refresh_token(self, refresh_token: str) -> Optional[TokenResponse]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: OAuth refresh token
            
        Returns:
            New TokenResponse or None if refresh fails
            
        Requirements: 1.6
        """
        token_data = {
            "client_id": self.config.google_client_id,
            "client_secret": self.config.google_client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.google_token_url,
                    data=token_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()
                
                token_json = response.json()
                return TokenResponse(
                    access_token=token_json["access_token"],
                    refresh_token=token_json.get("refresh_token", refresh_token),  # Keep old if not provided
                    expires_in=token_json.get("expires_in", 3600),
                    token_type=token_json.get("token_type", "Bearer")
                )
                
            except httpx.HTTPError as e:
                print(f"Token refresh failed: {e}")
                return None
    
    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """
        Revoke a user session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if session was revoked successfully
            
        Requirements: 1.7
        """
        # TODO: Implement session revocation with database
        # This would invalidate the session in the database
        # and optionally revoke the refresh token with Google
        return True
    
    def validate_credentials(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT credentials and extract user context.
        
        Args:
            token: JWT token to validate
            
        Returns:
            User context dictionary or None if invalid
            
        Requirements: 1.5
        """
        # Import here to avoid circular imports
        from backend.services.session_manager import create_session_manager
        
        # Use the same session manager instance for consistency
        if not hasattr(self, '_session_manager'):
            self._session_manager = create_session_manager(
                jwt_secret_key=self.config.jwt_secret_key,
                jwt_algorithm=self.config.jwt_algorithm,
                token_expiry_hours=self.config.token_expiry_hours,
                refresh_token_expiry_days=self.config.refresh_token_expiry_days,
                max_concurrent_sessions=self.config.max_concurrent_sessions
            )
        
        # Validate token and get session context
        session_context = self._session_manager.validate_session(token)
        if not session_context:
            return None
        
        # Convert to dictionary format
        return {
            "user_id": session_context.user_id,
            "session_id": session_context.session_id,
            "email": session_context.email,
            "display_name": session_context.display_name,
            "preferences": session_context.preferences,
            "privacy_settings": session_context.privacy_settings,
            "issued_at": session_context.issued_at,
            "expires_at": session_context.expires_at,
            "is_valid": session_context.is_valid
        }


# ── Utility Functions ──────────────────────────────────────────────────────────

def create_auth_service(
    google_client_id: str,
    google_client_secret: str,
    redirect_uri: str,
    jwt_secret_key: str
) -> AuthenticationService:
    """
    Factory function to create AuthenticationService with configuration.
    
    Args:
        google_client_id: Google OAuth client ID
        google_client_secret: Google OAuth client secret
        redirect_uri: OAuth redirect URI
        jwt_secret_key: JWT signing secret key
        
    Returns:
        Configured AuthenticationService instance
    """
    config = AuthConfig(
        google_client_id=google_client_id,
        google_client_secret=google_client_secret,
        redirect_uri=redirect_uri,
        jwt_secret_key=jwt_secret_key
    )
    
    return AuthenticationService(config)