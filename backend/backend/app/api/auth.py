from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class LoginReq(BaseModel):
    email: str
    password: str

@router.post('/login')
async def login(req: LoginReq):
    # Mock token for now
    return {"access_token": "mock-token", "token_type": "bearer"}
