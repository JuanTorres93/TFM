# Run this file with python3 -m unittest
# Otherwise it will fail due to imports
import math

import numpy as np
import unittest
try:
    import context
except ModuleNotFoundError:
    import src.tests.context

from src.modules import structures as st



class TestNode(unittest.TestCase):
    def test_constructor(self):
        node = st.Node("N1", position=(1, 2, 3), force=(4, 5, 6), momentum=(7, 8, 9), support=st.Support.NONE)

        self.assertEqual(node.name, "N1")
        np.testing.assert_array_equal(node.position, np.array((1, 2, 3)))
        np.testing.assert_array_equal(node.force, np.array((4, 5, 6)))
        np.testing.assert_array_equal(node.momentum, np.array((7, 8, 9)))
        self.assertEqual(node.support, st.Support.NONE)
        self.assertRaises(TypeError, st.Node, 3)

        self.assertRaises(TypeError, st.Node, "N1", position=[1, 2, 3])
        self.assertRaises(TypeError, st.Node, "N1", force=[1, 2, 3])
        self.assertRaises(TypeError, st.Node, "N1", momentum=[1, 2, 3])

    def test_set_name(self):
        node = st.Node("N1")

        node.set_name("node")
        self.assertEqual("node", node.name)

        self.assertRaises(TypeError, node.set_name, 0)

    def test_set_position(self):
        node = st.Node("N1", position=(1, 1, 1))

        node.set_position((2, 2, 2))

        np.testing.assert_array_equal(node.position, (2, 2, 2))

        self.assertRaises(TypeError, node.set_position, 0)
        self.assertRaises(TypeError, node.set_position, [0, 0, 0])

    def test_set_force(self):
        node = st.Node("N1", force=(1, 1, 1))

        node.set_force((2, 2, 2))
        np.testing.assert_array_equal(node.force, (2, 2, 2))

        self.assertRaises(TypeError, node.set_force, 0)
        self.assertRaises(TypeError, node.set_force, [0, 0, 0])

    def test_set_momentum(self):
        node = st.Node("N1", momentum=(1, 1, 1))

        node.set_momentum((2, 2, 2))

        np.testing.assert_array_equal(node.momentum, (2, 2, 2))

        self.assertRaises(TypeError, node.set_momentum, 0)
        self.assertRaises(TypeError, node.set_momentum, [0, 0, 0])

    def test_set_support(self):
        node = st.Node("N1", momentum=(1, 1, 1))
        self.assertEqual(node.support, st.Support.NONE)

        node.set_support(st.Support.FIXED)
        self.assertEqual(node.support, st.Support.FIXED)

        self.assertRaises(TypeError, node.set_support, 0)
        self.assertRaises(TypeError, node.set_support, [0, 0, 0])

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

        self.assertRaises(TypeError, bar.set_name, 0)

    def test_set_origin(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.origin, n_ori)

        new_origin = st.Node("N3", position=(1, 2, 3))
        bar.set_origin(new_origin)

        self.assertEqual(bar.origin, new_origin)

        self.assertRaises(TypeError, bar.set_origin, 0)

    def test_set_end(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2")

        bar = st.Bar("B1", n_ori, n_end)
        self.assertEqual(bar.end, n_end)

        new_end = st.Node("N3", (1, 2, 3))
        bar.set_end(new_end)

        self.assertEqual(bar.end, new_end)
        self.assertRaises(TypeError, bar.set_end, 0)
        self.assertRaises(TypeError, bar.set_end, (0, 0, 0))

    def test_length(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(1, 2, 3))

        bar = st.Bar("B1", n_ori, n_end)

        self.assertAlmostEqual(bar.length(), 3.741657387)

    def test_local_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.local_rigidity_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [263798885, 0, 0, -263798885, 0, 0],
            [0, 2787223.38, 5853169.1, 0, -2787223.38, 5853169.10],
            [0, 5853169.1, 16388873.48, 0, -5853169.1, 8194436.74],
            [-263798885, 0, 0, 263798885, 0, 0],
            [0, -2787223.38, -5853169.1, 0, 2787223.38, -5853169.10],
            [0, 5853169.1, 8194436.74, 0, -5853169.1, 16388873.48],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix)


    def test_system_change_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.system_change_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, rtol=1e-5, atol=1e-7)

    def test_angle_from_global_to_local(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.5, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.PINNED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        b1 = st.Bar("B1", n1, n2)

        np.testing.assert_equal(b1._angle_from_global_to_local(), np.pi / 2)
        np.testing.assert_almost_equal(b2._angle_from_global_to_local(), math.radians(8.778), 3)
        np.testing.assert_almost_equal(b3._angle_from_global_to_local(), math.radians(351.222), 3)
        # Differs in 0.03 radians due to trigonometry used to calculate node position
        np.testing.assert_almost_equal(b4._angle_from_global_to_local(), math.radians(351.222), 0)
        np.testing.assert_equal(b5._angle_from_global_to_local(), math.radians(270))

    def test_global_rigidity_matrix_2d_rigid_nodes(self):
        n_ori = st.Node("N1")
        n_end = st.Node("N2", position=(0, 4.2, 0))

        bar = st.Bar("B1", n_ori, n_end)

        calculated_matrix = bar.global_rigidity_matrix_2d_rigid_nodes()

        expected_matrix = np.array([
            [2787223.38, 0, -5853169.10, -2787223.38, 0, -5853169.10],
            [0, 263798885, 0, 0, -263798885, 0],
            [-5853169.1, 0, 16388873.48, 5853169.10, 0, 8194436.74],
            [-2787223.38, 0, 5853169.10, 2787223.38, 0, 5853169.10],
            [0, -263798885, 0, 0, 263798885, 0],
            [-5853169.1, 0, 8194436.74, 5853169.10, 0, 16388873.48],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, rtol=1e-6, atol=1)


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

    def test_assembled_matrix(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_matrix = structure.assembled_matrix()

        expected_matrix = 10000 * np.array([
            [278, 0, -585, -278, 0, -585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 26379, 0, 0, -26379, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-585, 0, 1638, 585, 0, 819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-278, 0, 585, 16007, 2418, 552, -15729, -2428, -33, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -26379, 0, 2418, 26816, 215, -2418, -436, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-585, 0, 819, 552, 215, 2639, 33, -215, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -15729, -2418, 33, 31458, 0, 66, -15729, 2418, 33, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2418, -436, -215, 0, 873, 0, 2418, -436, 215, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -33, 215, 500, 66, 0, 2000, -33, -215, 500, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -15729, 2418, -33, 45725, -6941, 670, -29717, 4522, 118, -278, 0, 585],
            [0, 0, 0, 0, 0, 0, 2418, -436, -215, -6941, 27942, 553, 4522, -1125, 769, 0, -26379, 0],
            [0, 0, 0, 0, 0, 0, 33, 215, 500, 670, 553, 4528, -118, -769, 944, -585, 0, 819],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -29717, 4522, -118, 29717, -4522, -118, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4522, -1125, -769, -4522, 1125, -769, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 769, 944, -118, -769, 1889, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -278, 0, -585, 0, 0, 0, 278, 0, -585],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -26379, 0, 0, 0, 0, 0, 26379, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 585, 0, 819, 0, 0, 0, -585, 0, 1638]
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, atol=90300)

    def test_decoupled_matrix(self):
        n1 = st.Node("N1", position=(0, 0, 0), support=st.Support.PINNED)
        n2 = st.Node("N2", position=(0, 4.2, 0))
        n3 = st.Node("N3", position=(6.8, 5.25, 0))
        n4 = st.Node("N4", position=(13.6, 4.2, 0))
        n5 = st.Node("N5", position=(17.2, 3.644117647, 0))
        n6 = st.Node("N6", position=(13.6, 0, 0), support=st.Support.FIXED)

        b1 = st.Bar("B1", n1, n2)
        b2 = st.Bar("B2", n2, n3)
        b3 = st.Bar("B3", n3, n4)
        b4 = st.Bar("B4", n4, n5)
        b5 = st.Bar("B5", n4, n6)

        bars = {
            b1.name: b1,
            b2.name: b2,
            b3.name: b3,
            b4.name: b4,
            b5.name: b5
        }

        structure = st.Structure("S1", bars)

        calculated_matrix = structure.decoupled_matrix()

        expected_matrix = 10000 * np.array([
            [1638, 585, 0, 819, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [585, 16007, 2418, 552, -15729, -2428, -33, 0, 0, 0, 0, 0, 0],
            [0, 2418, 26816, 215, -2418, -436, 215, 0, 0, 0, 0, 0, 0],
            [819, 552, 215, 2639, 33, -215, 500, 0, 0, 0, 0, 0, 0],
            [0, -15729, -2418, 33, 31458, 0, 66, -15729, 2418, 33, 0, 0, 0],
            [0, -2418, -436, -215, 0, 873, 0, 2418, -436, 215, 0, 0, 0],
            [0, -33, 215, 500, 66, 0, 2000, -33, -215, 500, 0, 0, 0],
            [0, 0, 0, 0, -15729, 2418, -33, 45725, -6941, 670, -29717, 4522, 118],
            [0, 0, 0, 0, 2418, -436, -215, -6941, 27942, 553, 4522, -1125, 769],
            [0, 0, 0, 0, 33, 215, 500, 670, 553, 4528, -118, -769, 944],
            [0, 0, 0, 0, 0, 0, 0, -29717, 4522, -118, 29717, -4522, -118],
            [0, 0, 0, 0, 0, 0, 0, 4522, -1125, -769, -4522, 1125, -769],
            [0, 0, 0, 0, 0, 0, 0, 118, 769, 944, -118, -769, 1889],
        ])

        np.testing.assert_allclose(calculated_matrix, expected_matrix, atol=90300)

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
