import sqlite3 as sql

from src.modules import filesystem_utils as fs

db_directory = fs.home_directory() + "/.local/TFM/"
db_name = "db.sqlite"
db_path = db_directory + db_name


def destroy_database():
    fs.remove(db_path)


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


def execute_read_query(connection, query):
    """Original function from https://realpython.com/python-sql-libraries/#understanding-the-database-schema"""
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except sql.Error as e:
        print(f"The error '{e}' occurred")


def regenerate_initial_database(force=False):
    if force or not fs.path_exists(db_path):
        destroy_database()

        connection = create_connection()

        queries = []

        create_materials_table = """
        CREATE TABLE IF NOT EXISTS materials (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          generic_name TEXT NOT NULL,
          name TEXT NOT NULL UNIQUE,
          e REAL NOT NULL
        );
        """
        populate_materials_table = """
        INSERT INTO materials (generic_name, name, e)
        VALUES ('steel', 's275j', 205939650000);
        """

        queries.append(create_materials_table)
        queries.append(populate_materials_table)

        for query in queries:
            execute_query(connection, query)




# TODO BORRAR TODO LO DE ABAJO CUANDO HAYA TERMINADO DEHACER PRUEBAS
regenerate_initial_database(force=True)

conn = create_connection()

result = execute_read_query(conn, """
SELECT * FROM materials;
""")

print(result)