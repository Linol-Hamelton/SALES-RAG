"""Authentication endpoints: register, login, me."""
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from app.database import get_connection
from app.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=128)
    email: str = ""
    full_name: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (req.username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(400, "Username already taken")
    hashed = hash_password(req.password)
    cur = conn.execute(
        "INSERT INTO users (username, hashed_pw, email, full_name) VALUES (?, ?, ?, ?)",
        (req.username, hashed, req.email, req.full_name),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    token = create_access_token(user_id, req.username)
    return TokenResponse(access_token=token, user_id=user_id, username=req.username)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    conn = get_connection()
    row = conn.execute("SELECT id, username, hashed_pw, is_active FROM users WHERE username = ?", (req.username,)).fetchone()
    conn.close()
    if not row or not verify_password(req.password, row["hashed_pw"]):
        raise HTTPException(401, "Invalid username or password")
    if not row["is_active"]:
        raise HTTPException(403, "Account disabled")
    token = create_access_token(row["id"], row["username"])
    return TokenResponse(access_token=token, user_id=row["id"], username=row["username"])


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)):
    return UserResponse(**user)
