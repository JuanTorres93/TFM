# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import unittest
from modules import structures as st
# from src.modules import structures as st


class TestNode(unittest.TestCase):
    # TODO Modify support when it is defined
    def test_constructor(self):
        node = st.Node("N1", position=(1, 2, 3), force=(4, 5, 6), momentum=(7, 8, 9), support=0)

        self.assertEqual(node.name, "N1")
        self.assertEqual(node.position, (1, 2, 3))
        self.assertEqual(node.force, (4, 5, 6))
        self.assertEqual(node.momentum, (7, 8, 9))
        self.assertEqual(node.support, 0)
        self.assertRaises(TypeError, st.Node, 3)

    # TODO name should be an str
    def test_set_name(self):
        node = st.Node("N1")

        node.set_name("node")
        self.assertEqual("node", node.name)

    # TODO position should be a tuple of int or float
    def test_set_position(self):
        node = st.Node("N1", position=(1, 1, 1))

        node.set_position((2, 2, 2))
        self.assertEqual((2, 2, 2), node.position)

    # TODO force should be a tuple of int or float
    def test_set_force(self):
        node = st.Node("N1", force=(1, 1, 1))

        node.set_force((2, 2, 2))
        self.assertEqual((2, 2, 2), node.force)

    # TODO momentum should be a tuple of int or float
    def test_set_momentum(self):
        node = st.Node("N1", momentum=(1, 1, 1))

        node.set_momentum((2, 2, 2))
        self.assertEqual((2, 2, 2), node.momentum)

    def test_x(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(1, node.x())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.x())

    def test_y(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(2, node.y())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.y())

    def test_z(self):
        node = st.Node("N1", position=(1, 2, 3))
        self.assertEqual(3, node.z())

        node.set_position((2, 2, 2))
        self.assertEqual(2, node.z())


class TestBar(unittest.TestCase):
    def test_constructor(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end)

        self.assertEqual(bar.name, "B1")
        self.assertEqual(bar.origin, n_ori)
        self.assertEqual(bar.end, n_end)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, (0, 0, 0))
        self.assertRaises(TypeError, st.Bar, "name", (0, 0, 0), n_end)
        self.assertRaises(TypeError, st.Bar, 3, n_ori, n_end)
        self.assertRaises(ValueError, st.Bar, "name", n_ori, n_ori)

    def test_set_name(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        bar.set_name("New name")

        self.assertEqual("New name", bar.name)

    def test_set_origin(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.origin, n_ori)

        new_origin = (1, 2, 3)
        bar.set_origin(new_origin)

        self.assertEqual(bar.origin, (1, 2, 3))

    def test_set_end(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.end, n_end)

        new_end = (1, 2, 3)
        bar.set_end(new_end)

        self.assertEqual(bar.end, (1, 2, 3))

    def test_length(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end)

        self.assertAlmostEqual(bar.length(), 3.741657387)


class TestStructure(unittest.TestCase):
    def test_constructor(self):
        n1 = st.Node("N1")
        n2 = st.Node("N2", position=(0, 2, 3))
        n3 = st.Node("N3", position=(1, 2, 3))

        bar_1 = st.Bar("B1", n1, n2)
        bar_2 = st.Bar("B2", n2, n3)

        bar_list = [bar_2, bar_1]

        self.assertRaises(TypeError, st.Structure, "name", bar_1)
        self.assertRaises(TypeError, st.Structure, "name", bar_list)

        bar_dict = {
            'b1': bar_1,
            'b2': bar_2
        }

        structure = st.Structure("S1", bar_dict)

        self.assertEqual(structure.name, "S1")
        self.assertEqual(structure.bars.get('b1'), bar_1)
        self.assertEqual(structure.bars.get('b2'), bar_2)

        bar_3 = st.Bar("B1", n2, n3)
        bar_dict = {
            'b1': bar_1,
            'b2': bar_3
        }

        self.assertRaises(ValueError, st.Structure, "name", bar_dict)


if __name__ == '__main__':
    unittest.main()
