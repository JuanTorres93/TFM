import sqlite3 as sql

from src.modules import filesystem_utils as fs

db_directory = fs.home_directory() + "/.local/TFM/"
db_name = "db.sqlite"
db_path = db_directory + db_name


def create_connection(path=db_path):
    """Original function from https://realpython.com/python-sql-libraries/#understanding-the-database-schema"""
    fs.create_directory(db_directory)

    connection = None

    try:
        connection = sql.connect(path)
        print("Connection to SQLite DB successful")
    except sql.Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    """Original function from https://realpython.com/python-sql-libraries/#understanding-the-database-schema"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except sql.Error as e:
        print(f"The error '{e}' occurred")

def generate_initial_database():
    connection = create_connection()

    create_materials_table = """
    CREATE TABLE IF NOT EXISTS materials (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      e REAL NOT NULL
    );
    """

    execute_query(connection, create_materials_table)

