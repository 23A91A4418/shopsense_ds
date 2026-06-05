import pandas as pd
import sqlalchemy
from sqlalchemy import text

def create_schema_and_tables(engine) -> None:
    """
    Create the shopsense schema and all 4 tables with primary/foreign keys and indexes.
    """
    with engine.begin() as conn:
        # Drop schema if exists to ensure we get the updated columns
        conn.execute(text("DROP SCHEMA IF EXISTS shopsense CASCADE;"))
        conn.execute(text("CREATE SCHEMA shopsense;"))

        # Create products table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shopsense.products (
                product_id VARCHAR(50) PRIMARY KEY,
                product_name VARCHAR(255) NOT NULL,
                category VARCHAR(100) NOT NULL,
                base_price DOUBLE PRECISION NOT NULL,
                brand_tier VARCHAR(50) NOT NULL,
                avg_rating DOUBLE PRECISION NOT NULL
            );
        """))

        # Create customers table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shopsense.customers (
                customer_id VARCHAR(50) PRIMARY KEY,
                signup_date TIMESTAMP NOT NULL,
                age INTEGER NOT NULL,
                gender VARCHAR(10) NOT NULL,
                city VARCHAR(100) NOT NULL,
                acquisition_channel VARCHAR(50) NOT NULL,
                is_premium BOOLEAN NOT NULL,
                customer_profile VARCHAR(50),
                churn_label INTEGER NOT NULL
            );
        """))

        # Create transactions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shopsense.transactions (
                transaction_id VARCHAR(50) PRIMARY KEY,
                customer_id VARCHAR(50) REFERENCES shopsense.customers(customer_id) ON DELETE CASCADE,
                transaction_date TIMESTAMP NOT NULL,
                product_id VARCHAR(50) REFERENCES shopsense.products(product_id) ON DELETE CASCADE,
                category VARCHAR(100) NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price DOUBLE PRECISION NOT NULL,
                discount_pct DOUBLE PRECISION NOT NULL,
                payment_method VARCHAR(50) NOT NULL,
                return_flag INTEGER NOT NULL,
                revenue DOUBLE PRECISION
            );
        """))

        # Create events table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shopsense.events (
                event_id VARCHAR(50) PRIMARY KEY,
                customer_id VARCHAR(50) REFERENCES shopsense.customers(customer_id) ON DELETE CASCADE,
                event_date TIMESTAMP NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                session_duration_sec INTEGER NOT NULL,
                device_type VARCHAR(50) NOT NULL,
                page_category VARCHAR(100) NOT NULL
            );
        """))

        # Create indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON shopsense.transactions(customer_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_transactions_transaction_date ON shopsense.transactions(transaction_date);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_events_customer_id ON shopsense.events(customer_id);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_events_event_date ON shopsense.events(event_date);
        """))

def load_dataframe_to_db(df: pd.DataFrame, table_name: str, engine, schema: str = "shopsense", if_exists: str = "replace") -> int:
    """
    Load a pandas DataFrame to PostgreSQL.
    """
    if if_exists == "replace":
        with engine.begin() as conn:
            # Check if table exists
            check_query = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table_name}'
                );
            """)
            table_exists = conn.execute(check_query).scalar()
            if table_exists:
                # Truncate rather than drop to keep keys, schema, and indexes intact
                conn.execute(text(f'TRUNCATE TABLE "{schema}"."{table_name}" CASCADE;'))
                # Set if_exists to append since table is now empty but schema is preserved
                if_exists = "append"

    # Write dataframe to SQL
    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=5000
    )
    return len(df)

def execute_query(query: str, engine) -> pd.DataFrame:
    """
    Execute a SELECT SQL query and return a pandas DataFrame.
    """
    return pd.read_sql_query(query, con=engine)
