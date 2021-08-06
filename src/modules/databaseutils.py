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


def generate_profile_query(name, name_number, area, inertia_moment_x, res_mod_x, inertia_moment_y, res_mod_y):
    query = f"INSERT INTO profiles (name, name_number, area, inertia_moment_x, res_mod_x, inertia_moment_y, " \
            f"res_mod_y) VALUES ('{str(name)}', {str(name_number)}, {str(area)}, {str(inertia_moment_x)}, " \
            f"{str(res_mod_x)}, {str(inertia_moment_y)}, {str(res_mod_y)}); "

    return query


def generate_material_query(generic_name, name, young_mod, rig_mod, poisson_coef, thermal_dil_coef, density):
    query = f"INSERT INTO materials(generic_name, name, young_mod, rig_mod, poisson_coef, thermal_dil_coef, density) " \
            f"VALUES('{generic_name}', '{name}', {str(young_mod)}, {str(rig_mod)}, {str(poisson_coef)}, " \
            f"{str(thermal_dil_coef)}, {str(density)});"

    return query


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
        #     """
        # CREATE TABLE IF NOT EXISTS profiles (
        # id INTEGER PRIMARY KEY AUTOINCREMENT,
        # name TEXT NOT NULL, -- IPE, HEB...
        # name_number INTEGER NOT NULL, -- e.g. 300 in IPE 300
        # area REAL NOT NULL,
        # inertia_moment_x REAL NOT NULL,
        # res_mod_x REAL NOT NULL,
        # inertia_moment_y REAL NOT NULL,
        # res_mod_y REAL NOT NULL
        # );
        # """
            """
        CREATE TABLE IF NOT EXISTS profiles (
        name TEXT NOT NULL, -- IPE, HEB...
        name_number INTEGER NOT NULL, -- e.g. 300 in IPE 300
        area REAL NOT NULL, 
        inertia_moment_x REAL NOT NULL,
        res_mod_x REAL NOT NULL,
        inertia_moment_y REAL NOT NULL,
        res_mod_y REAL NOT NULL,
        PRIMARY KEY (name, name_number)
        );
        """
        ]
        # generic_name, name, young_mod (E), rig_mod (G), poisson_coef (mu), thermal_dil_coef (alpha), density (rho)
        materials = [('steel', 's275j', 205939650000, 81000000000, 0.3, 0.000012, 7850),
                     ('steel', 's235j', 205939650000, 81000000000, 0.3, 0.000012, 7850),
                     ('steel', 's280', 205939650000, 81000000000, 0.3, 0.000012, 7850),
                     ('steel', 's350', 205939650000, 81000000000, 0.3, 0.000012, 7850),
                     ('steel', 's355', 205939650000, 81000000000, 0.3, 0.000012, 7850),
                     ('steel', 's420', 205939650000, 81000000000, 0.3, 0.000012, 7850)
                     ]

        # name, name_number, area, inertia_moment_x, res_mod_x, inertia_moment_y, res_mod_y
        profiles = [('IPE', 80, 0.000764, 0.000000801, 0.000020, 0.0000000849, 0.00000369),
                    ('IPE', 100, 0.00103239, 1.71024e-6, 34.2049e-6, 159.189e-9, 5.78868e-6),
                    ('IPE', 120, 0.00132109, 3.17772e-6, 52.962e-6, 276.684e-9, 8.64637e-6),
                    ('IPE', 140, 0.00164267, 5.4125e-6, 77.3214e-6, 449.18e-9, 12.3063e-6),
                    ('IPE', 160, 0.00200925, 8.69349e-6, 108.669e-6, 683.15e-9, 16.6622e-6),
                    ('IPE', 180, 0.00239485, 13.1703e-6, 146.337e-6, 1.00851e-6, 22.165e-6),
                    ('IPE', 200, 0.00284862, 19.4333e-6, 194.333e-6, 1.4237e-6, 28.4739e-6),
                    ('IPE', 220, 0.00333726, 27.7203e-6, 252.003e-6, 2.04888e-6, 37.2523e-6),
                    ('IPE', 240, 0.00391194, 38.9198e-6, 324.332e-6, 2.83637e-6, 47.2729e-6),
                    ('IPE', 270, 0.00459482, 57.9024e-6, 428.907e-6, 4.19872e-6, 62.2033e-6),
                    ('IPE', 300, 0.00538, 0.00008356, 0.000577, 0.0000000604, 80.5042e-6),
                    ('IPE', 330, 0.00626109, 117.679e-6, 713.207e-6, 7.88149e-6, 98.5186e-6),
                    ('IPE', 360, 0.00727339, 162.668e-6, 903.713e-6, 10.4346e-6, 122.76e-6),
                    ('IPE', 400, 0.00844699, 231.304e-6, 0.00115652, 13.1784e-6, 146.426e-6),
                    ('IPE', 450, 0.00988271, 337.455e-6, 0.0014998, 16.7587e-6, 176.408e-6),
                    ('IPE', 500, 0.0115528, 482.018e-6, 0.00192807, 21.417e-6, 214.17e-6),
                    ('IPE', 550, 0.0134424, 671.217e-6, 0.00244079, 26.6761e-6, 254.058e-6),
                    ('IPE', 600, 0.0155993, 920.896e-6, 0.00306965, 33.8736e-6, 307.942e-6)]

        for m in materials:
            queries.append(generate_material_query(m[0], m[1], m[2], m[3], m[4], m[5], m[6]))

        for p in profiles:
            queries.append(generate_profile_query(p[0], p[1], p[2], p[3], p[4], p[5], p[6]))

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

# regenerate_initial_database(force=True)
# conn = create_connection()
# x = execute_read_query(conn, "SELECT * FROM materials;")
# y = execute_read_query(conn, "SELECT * FROM profiles;")
# print(x)
# print(y)
