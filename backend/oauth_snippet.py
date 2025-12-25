from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.requests import Request as StarletteRequest

# ... (Add this near imports) ...
# Ensure these are imported
from google_auth_oauthlib.flow import Flow

# Update constant
REDIRECT_URI = "http://localhost:9091/auth/google/callback"
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' # Allow http for local dev

# Add Middleware for session (required for oauth flow state)
app.add_middleware(SessionMiddleware, secret_key="secret-key-walnut-saas")

@app.get("/auth/login")
def login(request: StarletteRequest):
    if not os.path.exists(CREDS_FILE):
        raise HTTPException(status_code=500, detail="SaaS Server Credentials not configured.")

    flow = Flow.from_client_secrets_file(
        CREDS_FILE, 
        scopes=SCOPES, 
        redirect_uri=REDIRECT_URI
    )
    
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    
    request.session['state'] = state
    return RedirectResponse(authorization_url)

@app.get("/auth/google/callback")
def auth_callback(request: StarletteRequest):
    state = request.session.get('state')
    
    if not state:
         raise HTTPException(status_code=400, detail="State missing in session")

    flow = Flow.from_client_secrets_file(
        CREDS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI
    )
    
    # Use the authorization server's response to fetch the OAuth 2.0 token.
    authorization_response = str(request.url)
    flow.fetch_token(authorization_response=authorization_response)

    creds = flow.credentials
    
    # Save credentials (per user in real app, global for now)
    with open(TOKEN_FILE, 'wb') as token:
        pickle.dump(creds, token)
        
    # Redirect back to frontend
    return RedirectResponse("http://localhost:5173/")
