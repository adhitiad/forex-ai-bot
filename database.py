import datetime
import logging

# Definisi Tabel
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")

# --- FIX DATABASE CRASH ---
connect_args = {}

# Cek apakah pakai SQLite?
if "sqlite" in settings.DATABASE_URL:
    # Hanya SQLite yang butuh ini
    connect_args = {"check_same_thread": False}
    logger.info("📂 Using SQLite Database")
else:
    # PostgreSQL tidak boleh ada check_same_thread
    logger.info("🐘 Using PostgreSQL/Remote Database")

# Buat Engine dengan argumen yang sudah disesuaikan
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    # pool_pre_ping berguna agar koneksi Postgres tidak putus saat idle lama
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String, index=True)
    action = Column(String)
    entry_price = Column(Float)
    tp_price = Column(Float)
    sl_price = Column(Float)
    status = Column(String)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database Tables Ready")
    except Exception as e:
        logger.error(f"❌ Database Init Failed: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
