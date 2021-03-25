import sqlite3 as sql

from src.modules import filesystemutils as fs

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
        # print("Connection to SQLite DB successful")
    except sql.Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    """Original function from https://realpython.com/python-sql-libraries/#understanding-the-database-schema"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        # print("Query executed successfully")
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
          e REAL NOT NULL,
          g REAL NOT NULL,
          v REAL NOT NULL,
          a REAL NOT NULL,
          d REAL NOT NULL
        );
        """
        # TODO ser más exhaustivo con los datos de g, v, a, d
        populate_materials_table = """
        INSERT INTO materials (generic_name, name, e, g, v, a, d)
        VALUES ('steel', 's275j', 205939650000, 81000000000, 0.3, 0.000012, 7.85);
        """

        queries.append(create_materials_table)
        queries.append(populate_materials_table)

        for query in queries:
            execute_query(connection, query)

def add_material_to_db(material_info):
    """

    :param material_info: Tuple with the values in the following order (generic_name, name, e)
    :return:
    """
    if type(material_info[0]) not in [str] or type(material_info[1]) not in [str]:
        raise TypeError("material_info[0] and material_info[1] must be str")
    # Maybe the below condition will have to be updated when using numpy
    if type(material_info[2]) not in [int, float] or \
       type(material_info[3]) not in [int, float] or \
       type(material_info[4]) not in [int, float] or \
       type(material_info[5]) not in [int, float] or \
       type(material_info[6]) not in [int, float]:
        raise TypeError("material_info[3] to material_info[6] must be a number")

    conn = create_connection()
    query = """
    INSERT INTO materials (generic_name, name, e, g, v, a, d)
    VALUES (""" + "'" + material_info[0] + "', '" + material_info[1] + "', " + str(material_info[2]) + ");";

    execute_query(conn, query)


# regenerate_initial_database(force=True)
# conn = create_connection()
# x = execute_read_query(conn, "SELECT * FROM materials;")
# print(x)