import os
from fastapi import FastAPI, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, Float, select
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import numpy as np
from sklearn.linear_model import LinearRegression
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware

# ✅ Load environment variables
load_dotenv()

# ✅ Get DATABASE_URL securely
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables!")

# ✅ Convert URL for asyncpg (if needed)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# ✅ Create async database engine
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# ✅ Define Base model
Base = declarative_base()

# ✅ Example Data model
class DataPoint(Base):
    __tablename__ = "data_points"
    id = Column(Integer, primary_key=True, index=True)
    x_value = Column(Float, index=True)
    y_value = Column(Float, index=True)

# ✅ Use `lifespan` for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

# ✅ Initialize FastAPI with `lifespan`
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ✅ Add Rate Limiter (3 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ✅ Dependency to get the async DB session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ✅ Root endpoint
@app.get("/")
@limiter.limit("10/minute")  # Limit root endpoint to 10 requests per minute
async def read_root(request: Request):
    return {"message": "Welcome to used car price prediction"}

# ✅ Fetch all data
@app.get("/data/")
@limiter.limit("5/minute")  # Limit data fetching
async def get_data(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DataPoint))
    data = result.scalars().all()
    return [{"x": d.x_value, "y": d.y_value} for d in data]

# ✅ Predict price based on miles driven
@app.get("/predict/")
@limiter.limit("10/minute")  # Limit predictions
async def predict_price(
    request: Request,
    miles: float = Query(100000, description="Miles driven (default: 100000)"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(DataPoint))
    data = result.scalars().all()

    X = np.array([d.x_value for d in data]).reshape(-1, 1)
    y = np.array([d.y_value for d in data])

    model = LinearRegression()
    model.fit(X, y)

    predicted_price = model.predict([[miles]])[0]

    return {"miles": miles, "predicted_price": predicted_price}
