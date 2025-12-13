import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


# 1. Tabel Market Data (Optimized Time-Series)
class MarketTick(Base):
    __tablename__ = "market_ticks"

    # Di TimescaleDB, Time adalah primary index utama
    # Kita gunakan server_default agar DB yang handle waktu jika kosong
    time = Column(DateTime, primary_key=True, default=datetime.datetime.utcnow)
    symbol = Column(String, primary_key=True)
    price = Column(Float)
    volume = Column(Float)


# 2. Tabel User Preference
class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, unique=True, index=True)
    active_symbol = Column(String, default="EURUSD=X")
    risk_factor = Column(Float, default=1.0)
    is_active = Column(Boolean, default=True)


# 3. Tabel Trade Logs (Opsional, buat jaga-jaga)
class TradeLog(Base):
    __tablename__ = "trade_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String)
    action = Column(String)
    entry_price = Column(Float)
    tp_price = Column(Float)
    sl_price = Column(Float)
    status = Column(String)
    pnl = Column(Float, default=0.0)


# Setup Engine
# pool_pre_ping=True penting untuk Cloud DB agar koneksi tidak putus (Disconnect Handling)
engine = create_engine(
    settings.DATABASE_URL, pool_size=20, max_overflow=0, pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    inspector = inspect(engine)

    # 1. CEK DULU: Apakah tabel market_ticks sudah ada?
    # Ini mencegah error "UniqueViolation" pada pg_type
    if not inspector.has_table("market_ticks"):
        print("🛠️ Creating table 'market_ticks'...")
        # Buat tabel secara spesifik
        MarketTick.__table__.create(bind=engine)

        # Konversi ke Hypertable (Hanya dijalankan SEKALI saat tabel baru dibuat)
        with engine.connect() as conn:
            conn.commit()
            try:
                conn.execute(text("SELECT create_hypertable('market_ticks', 'time');"))
                print("✅ TimescaleDB Hypertable initialized.")
            except Exception as e:
                print(f"⚠️ Hypertable creation skipped: {e}")
    else:
        print("⚡ Table 'market_ticks' already exists. Skipping creation.")

    # 2. Buat tabel sisanya (UserPreference, TradeLog) jika belum ada
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        print(f"⚠️ Table creation warning: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
