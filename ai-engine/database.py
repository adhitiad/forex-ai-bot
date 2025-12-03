from datetime import timezone

from config import settings
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def get_engine():
    return create_engine(settings.DATABASE_URL)


def get_session_local():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


class TradeLog(Base):
    __tablename__ = "trade_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=timezone.utc)
    symbol = Column(String)
    action = Column(String)
    entry_price = Column(Float)
    tp_price = Column(Float)
    sl_price = Column(Float)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    status = Column(String, default="OPEN")
    pnl = Column(Float, nullable=True)


class ModelLog(Base):
    __tablename__ = "model_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=timezone.utc)
    version = Column(String)
    status = Column(String)


def init_db():
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Continue without database for now


def get_db():
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()
