import psycopg

DB_CONFIG = {
    "dbname": "skate_analysis",
    "user": "postgres",
    "password": "qazwsx89",
    "host": "localhost",
    "port": 5432,
}


def create_tables():
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT NOW(),
            jump_type TEXT,
            confidence REAL,
            rotations REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_probs (
            id SERIAL PRIMARY KEY,
            session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
            class_name TEXT NOT NULL,
            probability REAL NOT NULL
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Таблицы созданы")


if __name__ == "__main__":
    create_tables()