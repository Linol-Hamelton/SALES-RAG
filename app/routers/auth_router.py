"""Authentication endpoints: register, login, me, admin user management."""
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


class PendingUserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    is_active: int
    created_at: str


@router.post("/register")
def register(req: RegisterRequest):
    """Register a new user. Account is inactive until admin approves."""
    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (req.username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(400, "Username already taken")
    hashed = hash_password(req.password)
    cur = conn.execute(
        "INSERT INTO users (username, hashed_pw, email, full_name, is_active) VALUES (?, ?, ?, ?, 0)",
        (req.username, hashed, req.email, req.full_name),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {
        "status": "pending",
        "message": "Регистрация отправлена. Ожидайте одобрения администратора.",
        "user_id": user_id,
        "username": req.username,
    }


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    conn = get_connection()
    row = conn.execute("SELECT id, username, hashed_pw, is_active FROM users WHERE username = ?", (req.username,)).fetchone()
    conn.close()
    if not row or not verify_password(req.password, row["hashed_pw"]):
        raise HTTPException(401, "Invalid username or password")
    if not row["is_active"]:
        raise HTTPException(403, "Аккаунт ожидает одобрения администратором")
    token = create_access_token(row["id"], row["username"])
    return TokenResponse(access_token=token, user_id=row["id"], username=row["username"])


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)):
    return UserResponse(**user)


# --- Admin: user management ---

@router.get("/users/pending", response_model=list[PendingUserResponse])
def list_pending_users(user: dict = Depends(get_current_user)):
    """List users awaiting approval (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, email, full_name, is_active, created_at FROM users WHERE is_active = 0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/users/{user_id}/approve")
def approve_user(user_id: int, user: dict = Depends(get_current_user)):
    """Activate a pending user (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    target = conn.execute("SELECT id, username, is_active FROM users WHERE id = ?", (user_id,)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")
    conn.execute("UPDATE users SET is_active = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "approved", "user_id": user_id, "username": target["username"]}


@router.post("/users/{user_id}/reject")
def reject_user(user_id: int, user: dict = Depends(get_current_user)):
    """Delete a pending user (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    target = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")
    if target["role"] == "admin":
        conn.close()
        raise HTTPException(400, "Cannot reject admin user")
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "rejected", "user_id": user_id, "username": target["username"]}


@router.get("/users", response_model=list[PendingUserResponse])
def list_all_users(user: dict = Depends(get_current_user)):
    """List all users (admin only)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, email, full_name, is_active, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
