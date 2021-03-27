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

        queries = [
        # https://www.codigotecnico.org/pdf/Documentos/SE/DBSE-A.pdf
        """
        CREATE TABLE IF NOT EXISTS materials (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          generic_name TEXT NOT NULL,
          name TEXT NOT NULL UNIQUE,
          young_mod REAL NOT NULL, -- Young's modulus
          rig_mod REAL NOT NULL, -- rigidity modulus
          poisson_coef REAL NOT NULL, -- Poisson coeficiente
          thermal_dil_coef REAL NOT NULL, -- thermal dilatation
          density REAL NOT NULL  -- density
        );
        """,
        # More fields in http://prontuarios.info/perfiles/IPE
        # TODO pair name, number_name must be unique
        """
        CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, -- IPE, HEB...
        name_number INTEGER NOT NULL, -- e.g. 300 in IPE 300
        area REAL NOT NULL, 
        weight REAL NOT NULL, -- kg/m
        inertia_moment_x REAL NOT NULL,
        res_mod_x REAL NOT NULL,
        inertia_moment_y REAL NOT NULL,
        res_mod_y REAL NOT NULL
        );
        """,
        # TODO ser más exhaustivo con los datos de g, v, a, d (¿Son EXACTAMENTE iguales para TODOS los aceros?)
        """
        INSERT INTO materials (generic_name, name, young_mod, rig_mod, poisson_coef, 
        thermal_dil_coef, density)
        VALUES ('steel', 's275j', 205939650000, 81000000000, 0.3, 0.000012, 7.85);
        """,
        """
        INSERT INTO profiles (name, name_number, area, weight, inertia_moment_x, 
        res_mod_x, inertia_moment_y, res_mod_y)
        VALUES ('IPE', 80, 0.000764, 6, 0.000000801, 0.000020, 0.0000000849,
        0.00000369);
        """,
        """
        INSERT INTO profiles (name, name_number, area, weight, inertia_moment_x, 
        res_mod_x, inertia_moment_y, res_mod_y)
        VALUES ('IPE', 300, 0.00538, 42.2, 0.00008360, 0.000577, 0.0000000604,
        0.00000335);
        """
        ]

        for query in queries:
            execute_query(connection, query)


def add_material_to_db(material_info):
    """

    :param material_info: Tuple with the values in the following order (generic_name, name, e)
    :return:
    """
    if len(material_info) < 7:
        raise ValueError("Not enough information was provided.")
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
    VALUES (""" + "'" + material_info[0] + "', '" + material_info[1] + "', " + str(material_info[2]) + ");"

    execute_query(conn, query)


regenerate_initial_database(force=True)
conn = create_connection()
x = execute_read_query(conn, "SELECT * FROM materials;")
y = execute_read_query(conn, "SELECT * FROM profiles;")
print(x)
print(y)
