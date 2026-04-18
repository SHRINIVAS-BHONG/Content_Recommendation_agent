"""
auth_example.py — Example configuration for authentication service.

This file shows how to configure and use the AuthenticationService.
Copy this file to auth_config.py and update with your actual credentials.
"""

import os
from backend.services.authentication import create_auth_service

# Example configuration - replace with your actual values
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-change-this-in-production")

def get_auth_service():
    """
    Create and return configured authentication service.
    
    Returns:
        AuthenticationService: Configured authentication service instance
    """
    return create_auth_service(
        google_client_id=GOOGLE_CLIENT_ID,
        google_client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        jwt_secret_key=JWT_SECRET_KEY
    )

# Example usage:
if __name__ == "__main__":
    # Create authentication service
    auth_service = get_auth_service()
    
    # Initiate OAuth flow
    oauth_response = auth_service.initiate_oauth()
    print(f"Redirect URL: {oauth_response.redirect_url}")
    print(f"State: {oauth_response.state}")
    print(f"Code Verifier: {oauth_response.code_verifier}")
    
    print("\nTo complete OAuth flow:")
    print("1. Redirect user to the redirect_url")
    print("2. User authorizes with Google")
    print("3. Google redirects back with authorization code")
    print("4. Call handle_oauth_callback() with the code and stored parameters")