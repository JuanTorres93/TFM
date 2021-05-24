# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import unittest
try:
    import context
except ModuleNotFoundError:
    import src.tests.context

from src.modules import databaseutils as db


class TestQuerys(unittest.TestCase):
    def test_add_material_to_db(self):
        # TODO add tuple size assert
        self.assertRaises(ValueError, db.add_material_to_db, ("foo", "bar", 9))
        self.assertRaises(TypeError, db.add_material_to_db, (9, "foo", 9, 8, 8, 8, 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", 2, 9, 8, 8, 8, 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", "r", 8, 8, 8, 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", 8, "2", 8, 8, 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", 8, 2, "3", 8, 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", 8, 2, 4.9, "4", 8))
        self.assertRaises(TypeError, db.add_material_to_db, ("foo", "bar", 8, 2, 4.9, 5.7, "3"))


if __name__ == '__main__':
    unittest.main()
