# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import unittest
try:
    import context
except ModuleNotFoundError:
    import tests.context

from src.modules import databaseutils as db



class TestQuerys(unittest.TestCase):
    # TODO Modify support when it is defined
    def test_add_material_to_db(self):

        # TODO add tuple size assert
        self.assertRaises(TypeError, db.add_material_to_db, (9, "foo", 9))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", 2, 9))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", "r"))

if __name__ == '__main__':
    unittest.main()
