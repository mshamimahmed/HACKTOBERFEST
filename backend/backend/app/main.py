from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, doctor, research, illinois
from app.api import researcher

app = FastAPI(title="Symptom Repurpose API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    """
    Append security-related HTTP headers to all responses.
    Currently sets `Referrer-Policy: strict-origin-when-cross-origin`.
    """
    response = await call_next(request)
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response

app.include_router(auth.router, prefix="/auth", tags=["auth"])  # mock login
app.include_router(doctor.router, prefix="/doctor", tags=["doctor"])  # differential + analyzer + IL search
app.include_router(research.router, prefix="/research", tags=["research"])  # demo candidates/combos
app.include_router(illinois.router, prefix="/illinois", tags=["illinois"])  # IDPH sync/search
app.include_router(researcher.router, prefix="/api/researcher", tags=["researcher"])  # semantic search + composition

@app.get("/")
async def root():
    """Health/status endpoint for quick liveness checks."""
    return {"status": "ok"}
