# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import numpy as np
import unittest
try:
    import context
except ModuleNotFoundError:
    import tests.context

from src.modules import structures as st



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
        self.assertIsInstance(bar.material, st.Material)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, (0, 0, 0))
        self.assertRaises(TypeError, st.Bar, "name", (0, 0, 0), n_end)
        self.assertRaises(TypeError, st.Bar, 3, n_ori, n_end)
        self.assertRaises(ValueError, st.Bar, "name", n_ori, n_ori)
        self.assertRaises(TypeError, st.Bar, "name", n_ori, n_end, material=2)

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

    def test_local_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.local_rigidity_matrix_2d_rigid_nodes(i=8356 * 10 ** (-8))

        expected_matrix = np.array([
            [263798885, 0, 0, -263798885, 0, 0],
            [0, 2787223.38, 5853169.1, 0, -2787223.38, 5853169.10],
            [0, 5853169.1, 16388873.48, 0, -5853169.1, 8194436.74],
            [-263798885, 0, 0, 263798885, 0, 0],
            [0, -2787223.38, -5853169.1, 0, 2787223.38, -5853169.10],
            [0, 5853169.1, 8194436.74, 0, -5853169.1, 16388873.48],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix)


    def test_system_change_matrix_2d_rigid_nodes(angle):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.system_change_matrix_2d_rigid_nodes(angle=0)

        expected_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix)

        calculated_matrix = bar.system_change_matrix_2d_rigid_nodes(angle=1.30899)

        expected_matrix = np.array([
            [0.2588257, -0.9659240, 0],
            [0.9659240, 0.25882571, 0],
            [0, 0, 1],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, rtol=1e-5)


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

class TestMaterial(unittest.TestCase):
    def test_constructor(self):
        mat = st.Material("s275j")

        self.assertEqual(mat.generic_name, "steel")
        self.assertEqual(mat.name, "s275j")
        self.assertEqual(mat.young_mod, 205939650000)
        self.assertEqual(mat.rig_mod, 81000000000)
        self.assertEqual(mat.poisson_coef, 0.3)
        self.assertEqual(mat.thermal_dil_coef, 0.000012)
        self.assertEqual(mat.density, 7.85)

        self.assertRaises(TypeError, st.Material, 4)
        self.assertRaises(LookupError, st.Material, "foobar")


class TestProfile(unittest.TestCase):
    def test_constructor(self):
        pro = st.Profile("IPE", 80)

        self.assertEqual(pro.name, "IPE")
        self.assertEqual(pro.name_number, 80)
        self.assertEqual(pro.area, 0.000764)
        self.assertEqual(pro.weight, 6)
        self.assertEqual(pro.inertia_moment_x, 0.000000801)
        self.assertEqual(pro.res_mod_x, 0.000020)
        self.assertEqual(pro.inertia_moment_y, 0.0000000849)
        self.assertEqual(pro.res_mod_y, 0.00000369)

        self.assertRaises(TypeError, st.Profile, 4, 3)
        self.assertRaises(LookupError, st.Profile, "h", "foobar")

if __name__ == '__main__':
    unittest.main()
