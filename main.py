import os
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, Float, select
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi import Query

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

# ✅ Create an async session factory
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# ✅ Define Base model (fixing the previous error)
Base = declarative_base()

# ✅ Example Data model (Modify based on your actual table structure)
class DataPoint(Base):
    __tablename__ = "data_points"  # Replace with your actual table name
    id = Column(Integer, primary_key=True, index=True)
    x_value = Column(Float, index=True)
    y_value = Column(Float, index=True)

# ✅ Use `lifespan` for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # Create tables if not exist
    yield  # Hand control to the application
    await engine.dispose()  # Clean up resources on shutdown

# ✅ Initialize FastAPI with `lifespan`
app = FastAPI(lifespan=lifespan)

# ✅ Dependency to get the async DB session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ✅ Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Neon PostgreSQL Connected!"}

# ✅ Fetch all data for analysis (e.g., linear regression)
@app.get("/data/")
async def get_data(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DataPoint))
    data = result.scalars().all()

    # Convert to dictionary for easy JSON response
    return [{"x": d.x_value, "y": d.y_value} for d in data]

@app.get("/predict/")
async def predict_price(miles: float = Query(100000, description="Miles driven (default: 100000)"), db: AsyncSession = Depends(get_db)):
    # Fetch data from the database
    result = await db.execute(select(DataPoint))
    data = result.scalars().all()

    # Convert data to NumPy arrays
    X = np.array([d.x_value for d in data]).reshape(-1, 1)  # Miles driven
    y = np.array([d.y_value for d in data])  # Price

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make a prediction
    predicted_price = model.predict([[miles]])[0]

    return {"miles": miles, "predicted_price": predicted_price}
