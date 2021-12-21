import enum
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import re
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from src.modules import databaseutils as db
from src.modules import filesystemutils as fs
from src.modules import structures as st

# In spite of the fact that pycharm marks this import as not used, it really is used.
# Its function is to provide icons
import qrc_resources

# CONSTANTS
NODE_RADIUS = 10
Z_VALUE_NODES = 2
Z_VALUE_BARS = 1
Z_VALUE_AXIS = -1
# -- Colors
# ---- Normal color of the node
NORMAL_COLOR = QtGui.QColor(0, 0, 0)
ORIGIN_NODE_COLOR = QtGui.QColor(20, 40, 255) # Color used for punctual force distance to origin
# ---- Color of the node when hovering mouse over it
HOVER_COLOR_NORMAL_MODE = QtGui.QColor(49, 204, 55)
HOVER_COLOR_BAR_MODE = QtGui.QColor(203, 39, 23)
# ---- Color of the node when selected
SELECTED_COLOR = QtGui.QColor(235, 204, 55)
# -- Conversion factors
METER_TO_PX = 50
PX_TO_METER = 1 / METER_TO_PX

# Currently selected node or bar
active_structure_element = None

# Initialized from main window
node_properties = None
bar_properties = None


def unset_active_structure_element():
    """
    Sets the active structure element to None
    """
    global active_structure_element

    # If some element is currently active
    if active_structure_element is not None:
        # If it is a node, change its color to normal
        if type(active_structure_element) is Node:
            active_structure_element.change_node_color()
        elif type(active_structure_element) is Bar:
            active_structure_element.change_bar_color()
            active_structure_element.change_origin_node_color()

        # Set active_structure_element to None
        active_structure_element = None

        bar_properties.show()
        node_properties.show()


def set_active_structure_element(item):
    """
    Sets the active structure element to the specified one
    :param item: item that is desired to be the new active structure element
    """
    global active_structure_element
    active_structure_element = item

    if type(active_structure_element) is Bar:
        bar_properties.show()
        node_properties.hide()
    elif type(active_structure_element) is Node:
        bar_properties.hide()
        node_properties.show()


def get_widgets_in_layout(layout):
    """
    This function can be used to retrieve all widgets that have been added to a layout.
    :param layout: Layout to get the widgets from
    :return: list with the widgets contained in the layout
    """
    items = []
    total_elements = layout.count()

    for i in range(total_elements):
        items.append(layout.itemAt(i).widget())

    return items


def make_qline_edit_value_zero(qlineEdit: QtWidgets.QLineEdit):
    qlineEdit.setText("0")
    return 0


@enum.unique
class ApplicationMode(enum.Enum):
    """
    Enumeration for the different types of application modes
    """
    NODE_MODE = 1
    NORMAL_MODE = 2
    BAR_MODE = 3


# Default mode for application is normal mode
application_mode = ApplicationMode.NORMAL_MODE


class Window(QtWidgets.QMainWindow):
    """
    Application main window
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Central Widget
        # This widget is used to display the drawing scene where the user inputs the structure
        self.central_widget = QtWidgets.QGraphicsView()

        # Create actions
        self._create_actions()

        # Menu bar
        self._create_menu_bar()

        # Structure toolbar
        self._create_toolbars_and_docks()

        # Status bar
        self._create_status_bar()

        # Context menus
        self._create_context_menu()

        # Connect actions
        self._connect_actions()

        # Canvas
        self._create_drawing_scene(2000, 2000)

        # Configure central widget
        self.central_widget.setScene(self.scene)
        self.central_widget.setParent(self)

        # TODO el zoom se implementa con el método scale
        # self.central_widget.scale(4, 4)
        self.central_widget.show()

        # Draw axis in canvas
        self._draw_axis_lines()

        # List containing the effort laws windows
        self.effort_laws_windows = []

        # Main window definition
        self.setObjectName("MainWindow")
        self.setWindowTitle("TFM")
        self.setCentralWidget(self.central_widget)

        self.central_widget.setFocus()

    def set_current_mode(self, mode):
        """
        Establish the mode in which the user utilizes the application in a given moment
        :param mode: Mode to which the application is going to change
        :return:
        """
        if type(mode) is not ApplicationMode:
            raise TypeError("Error. mode must be of type ApplicationMode")

        global application_mode
        if application_mode != mode:
            # Change the mode and log it to console
            application_mode = mode
            print("Changed mode to " + str(mode))

            # Show the change in GUI
            message = str(mode).split(".")[1]
            message = message.replace("_", " ")
            self.current_mode_message.setText(message)

    def centered_coordinates(self, x, y):
        """
        Origin point for coordinates systems in image processing is located at the top left corner, this function
        provides a way to work with the origin centered in the scene
        :param x: desired centered x coordinate
        :param y: desired centered y coordinate
        :return: List with x and y values of the point in the new coordinate system
        """
        scene_width = self.scene.sceneRect().width()
        scene_height = self.scene.sceneRect().height()
        distance_point_y_axis = abs(x - scene_width / 2)
        distance_point_x_axis = abs(y - scene_height / 2)

        if x >= scene_width / 2:
            x_converted = distance_point_y_axis
        else:
            x_converted = - distance_point_y_axis

        if y <= scene_height / 2:
            y_converted = distance_point_x_axis
        else:
            y_converted = - distance_point_x_axis

        return [x_converted, y_converted]

    def scene_coordinates(self, x_centered, y_centered):
        """
        Origin point for coordinates systems in image processing is located at the top left corner, this function
        provides a way to convert from center coordinates to scene coordinates
        :param x_centered: desired scene x coordinate
        :param y_centered: desired scene y coordinate
        :return: List with x and y values of the point in the new coordinate system (scene)
        """
        scene_width = self.scene.sceneRect().width()
        scene_height = self.scene.sceneRect().height()

        if x_centered >= 0:
            x_converted = x_centered + scene_width / 2
        else:
            x_converted = scene_width / 2 - abs(x_centered)

        if y_centered >= 0:
            y_converted = scene_height / 2 - y_centered
        else:
            y_converted = scene_height / 2 + abs(y_centered)

        return [x_converted, y_converted]

    def _create_drawing_scene(self, width, height):
        """
        Creates the drawing scene
        :param width: Scene width
        :param height: Scene height
        :return:
        """
        self.scene = GraphicsScene(self, self)
        self.scene.setSceneRect(0, 0, width, height)

    def new_file(self):
        # TODO escribir funcion
        self.scene.addRect(1000, 500, 100, 100)

    def show_error_pop_up(self, title, msg):
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle(title)
        message_box.setText(msg)
        message_box.setIcon(QtWidgets.QMessageBox.Critical)
        message_box.exec_()

    def solve_structure(self):
        """
        This method is called from solve structure action
        """
        bars = self.scene.get_bars()

        bar_dict = {}
        for bar in bars:
            bar_logic = bar.bar_logic
            bar_dict[bar_logic.name] = bar_logic

        # Generate structure
        structure = st.Structure("S1", bar_dict)
        try:
            # First step in structure resolution
            structure.get_nodes_displacements()
        except AttributeError:
            self.show_error_pop_up("ERROR", "Singular matrix. Consider adding more supports.")
            return
        except TypeError:
            self.show_error_pop_up("ERROR", "Some value is not correctly entered. Check them and ensure that are"
                                            " valid numbers.")
            return

        # Calculate nodes reactions
        structure.get_nodes_reactions()

        def create_effort_laws_plots(main_window, bar):
            bar_logic = bar.bar_logic
            # Calculate efforts in bars
            bar_logic.calculate_efforts()
            bar_logic.get_efforts()

            # Points to represent
            number_of_points = 1000
            # Points to calculate efforts laws (per one)
            x_axis_normalized = list(map(lambda x: x / number_of_points,
                                         range(number_of_points))
                                     )

            # Points to show in graph
            x_axis_represented = list(map(lambda x: x * bar_logic.length(),
                                          x_axis_normalized)
                                      )

            # List to store the points of each effort law
            y_axis_axial_force = []
            y_axis_shear_strength = []
            y_axis_bending_moment = []


            # Calculate effort laws
            for x in x_axis_normalized:
                y_axis_axial_force.append(bar_logic.axial_force_law(x))
                y_axis_shear_strength.append(bar_logic.shear_strength_law(x))
                y_axis_bending_moment.append(bar_logic.bending_moment_law(x))

            mplCanvas = MplCanvas(width=7, height=6)
            mplCanvas.fig.suptitle(f"Bar {bar_logic.name}")
            mplCanvas.axes_axile.plot(x_axis_represented, y_axis_axial_force)
            mplCanvas.axes_shear.plot(x_axis_represented, y_axis_shear_strength)
            mplCanvas.axes_bending.plot(x_axis_represented, y_axis_bending_moment)

            main_window.effort_laws_windows.append(mplCanvas)

        # Delete all previously stored effort laws
        for w in self.effort_laws_windows:
            w.deleteLater()

        self.effort_laws_windows.clear()

        # Calculate effort laws in each bar
        for bar in bars:
            create_effort_laws_plots(self, bar)

        self.show_effort_laws()

    def show_effort_laws(self):
        """
        Once the structure has been solved, all plots are stored in a list. If it is attempted to show them at the
        the time of creation, some of them won't appear.
        :return:
        """
        for w in self.effort_laws_windows:
            w.show()

    def hide_effort_laws(self):
        """
        Once the structure has been solved, all plots are stored in a list. If it is attempted to show them at the
        the time of creation, some of them won't appear.
        :return:
        """
        for w in self.effort_laws_windows:
            w.hide()

    def activate_draw_node_mode(self):
        """
        Activates the application mode in which the nodes can be drawn with mouse clicks. The mode can be reverted back
        to normal if this function is called again
        :return:
        """
        global application_mode

        if application_mode != ApplicationMode.NODE_MODE:
            self.set_current_mode(ApplicationMode.NODE_MODE)
        else:
            self.set_current_mode(ApplicationMode.NORMAL_MODE)

    def activate_draw_bar_mode(self):
        """
        Activates the application mode in which the bars can be drawn with mouse clicks. The mode can be reverted back
        to normal if this function is called again
        :return:
        """
        global application_mode

        if application_mode != ApplicationMode.BAR_MODE:
            self.set_current_mode(ApplicationMode.BAR_MODE)
        else:
            self.set_current_mode(ApplicationMode.NORMAL_MODE)

    def draw_node(self, radius):
        """
        Draws a node at the cursor position
        :param radius: radius with which the node is drawn
        """
        # Get global position of the mouse
        view_position = self.central_widget.mapFromGlobal(QtGui.QCursor.pos())
        # Translate it to scene coordinates
        scene_position = self.central_widget.mapToScene(view_position)

        # Coordinates to draw the circle in its center instead of in its top-left corner
        draw_coordinates = [scene_position.x(), scene_position.y()]
        # Create node instance
        node = Node(self, draw_coordinates[0], draw_coordinates[1], radius)
        node.setZValue(Z_VALUE_NODES)

        # Add node to scene
        self.scene.addItem(node)

        # set_active_structure_element(node)

    def draw_bar(self, node_origin, node_end):
        """
        Draws a node at the cursor position
        :param node_origin:
        :param node_end:
        """
        if node_origin is not node_end:
            bar = Bar(self, node_origin, node_end)
            bar.signals.error_convert_value_to_string.connect(lambda: self.show_error_pop_up("ERROR", "Must be introduced"
                                                                                                      " a valid number."))

            bar.setZValue(Z_VALUE_BARS)

            # Add node to scene
            # TODO add the bar only if there is not another occupying the same position
            self.scene.addItem(bar)

    def delete_node(self, node):
        """
        Removes the specified node from the scene
        :param node: Node to be deleted
        """
        # Delete node
        node.label_support_image.deleteLater()
        self.scene.removeItem(node)
        # Unset active structure element in order not to have null references
        unset_active_structure_element()

    def delete_bar(self, bar):
        """
        Removes the specified bar from the scene
        :param bar: Bar to be deleted
        """
        # Delete node
        self.scene.removeItem(bar)
        # Unset active structure element in order not to have null references
        unset_active_structure_element()

    def _draw_axis_lines(self):
        """
        Draws the x and y axis on the scene
        :return:
        """
        # X Axis
        point_x1 = [0, self.scene.sceneRect().height() / 2]
        point_x2 = [self.scene.sceneRect().width(), self.scene.sceneRect().height() / 2]
        color = QtGui.QColor(200, 0, 0)
        # Draw axis
        x_axis = self.scene.addLine(point_x1[0], point_x1[1], point_x2[0], point_x2[1], pen=color)
        # Draw it at the bottom in order not to superpose user drawings
        x_axis.setZValue(Z_VALUE_AXIS)

        # Y Axis
        point_y1 = [self.scene.sceneRect().width() / 2, 0]
        point_y2 = [self.scene.sceneRect().width() / 2, self.scene.sceneRect().height()]
        color = QtGui.QColor(0, 200, 0)
        # Draw axis
        y_axis = self.scene.addLine(point_y1[0], point_y1[1], point_y2[0], point_y2[1], pen=color)
        # Draw it at the bottom in order not to superpose user drawings
        y_axis.setZValue(Z_VALUE_AXIS)

    def _create_menu_bar(self):
        """
        Creates the menu bar in the main window
        """
        menu_bar = self.menuBar()
        # ========== FILE MENU ==========
        # The parent is self (the main window) because, according documentation, the parent typically is the window
        # in which the menu is going to be used
        file_menu = QtWidgets.QMenu("&File", self)

        file_menu.addAction(self.new_file_action)
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.save_file_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        menu_bar.addMenu(file_menu)

        # ========== EDIT MENU ==========
        edit_menu = QtWidgets.QMenu("&Edit", self)

        # Structure
        edit_menu.addAction(self.enable_node_mode_action)
        edit_menu.addAction(self.enable_bar_mode_action)
        edit_menu.addAction(self.solve_structure_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.copy_action)
        edit_menu.addAction(self.cut_action)
        edit_menu.addAction(self.paste_action)

        menu_bar.addMenu(edit_menu)

        # ========== HELP MENU ==========
        help_menu = QtWidgets.QMenu("&Help", self)

        help_menu.addAction(self.help_content_action)
        help_menu.addAction(self.about_action)

        # TODO borrar
        test_borrar_menu = help_menu.addMenu("Test Borrar")
        test_borrar_menu.addAction("Test")
        test_borrar_menu.addAction("Borrar")

        menu_bar.addMenu(help_menu)

    def _update_selected_node_position(self, line_edit, text, axis):
        """
        This function is connected to the textboxes that represent the coordinates of the nodes
        :param text: current text of the textbox
        :param axis: axis in which the coordinate is going to change
        """
        try:
            new_pos = float(text)
        except ValueError:
            valid_number_pattern = re.compile("-?[0-9]*\.?[0-9]*")
            matches = valid_number_pattern.match(text).group()
            if len(matches) != len(text):
                line_edit.setText(matches)

            try:
                new_pos = float(matches)
            except ValueError:
                new_pos = 0
                new_text = str(new_pos)
                if text.strip() == "-":
                    new_text = "-"
                line_edit.setText(new_text)

        global active_structure_element
        if type(active_structure_element) is Node:
            if axis == "x":
                active_structure_element.update_position(new_pos, None)
            elif axis == "y":
                active_structure_element.update_position(None, new_pos)

    def _update_selected_node_support(self, text):
        """
        This function is connected to the support combobox
        :param text: current text of the textbox
        """
        if type(active_structure_element) is Node:
            active_structure_element.update_support(text)

    def _update_selected_bar_material(self, material):
        """
        This function is connected to the material combobox that represent the material of the bar
        :param material: material name to update to
        """
        if type(active_structure_element) is Bar:
            active_structure_element.update_material(material)

    def _update_selected_bar_profile(self, profile):
        """
        This function is connected to the profile combobox that represent the profile of the bar
        :param profile: tuple with profile name and number
        """
        if type(active_structure_element) is Bar:
            active_structure_element.update_profile(profile)

    def _add_distributed_charge_to_selected_bar(self):
        """
        This function is connected to the button distributed charge.
        """
        if type(active_structure_element) is Bar:
            active_structure_element.add_distributed_charge()

    def _add_punctual_force_to_selected_bar(self):
        """
        This function is connected to the button punctual force.
        """
        if type(active_structure_element) is Bar:
            active_structure_element.add_punctual_force()

    def _create_toolbars_and_docks(self):
        """
        Creates the toolbars and docks in the main window
        """
        # ========== STRUCTURE TOOLBAR ==========
        structure_toolbar = QtWidgets.QToolBar("Structure", self)

        structure_toolbar.addAction(self.enable_node_mode_action)
        structure_toolbar.addAction(self.enable_bar_mode_action)
        structure_toolbar.addAction(self.solve_structure_action)

        self.addToolBar(QtCore.Qt.LeftToolBarArea, structure_toolbar)

        # ========== PROPERTIES DOCK ==========
        def create_layout_and_container(layout_type, widget_type):
            """
            QSplitter class doesn't allow to add layouts directly, so a workaround is needed.
            This function returns a layout to which widgets can be added and a single widget that
            holds that layout.
            :param widget_type: layout to  be used inside widget
            :param layout_type: tyoe of widget to be returned
            :return: layout and container widget
            """
            layout = layout_type
            container = widget_type
            container.setLayout(layout)

            return layout, container

        properties_dock = QtWidgets.QDockWidget("Properties", self)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(splitter)
        properties_dock.setWidget(scroll_area)
        # properties_dock.setWidget(splitter)

        # Bar properties
        bar_properties_layout, bar_properties_container = create_layout_and_container(
            QtWidgets.QVBoxLayout(),
            QtWidgets.QGroupBox()
        )

        bar_properties_layout.addWidget(QtWidgets.QLabel("Bar properties"))
        global bar_properties
        bar_properties = bar_properties_container

        # -- Material
        material_layout, mat_container = create_layout_and_container(QtWidgets.QHBoxLayout(), QtWidgets.QWidget())

        # ---- Label material
        label_material = QtWidgets.QLabel("Material:")
        material_layout.addWidget(label_material)

        # ---- Material comboBox
        self.material_combo_box = QtWidgets.QComboBox()
        self.material_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self._populate_material_combobox()
        self.material_combo_box.currentTextChanged.connect(lambda: self._update_selected_bar_material(
            self.get_currently_selected_material()
        ))

        material_layout.addWidget(self.material_combo_box)

        bar_properties_layout.addWidget(mat_container)

        # -- Profile
        profile_layout, profile_container = create_layout_and_container(QtWidgets.QHBoxLayout(), QtWidgets.QWidget())
        # ---- Label profile
        label_profile = QtWidgets.QLabel("Profile:")
        profile_layout.addWidget(label_profile)

        # ---- ComboBox profile
        self.profile_combo_box = QtWidgets.QComboBox()
        self.profile_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)
        profile_layout.addWidget(self.profile_combo_box)
        self._populate_profile_combobox()
        self.profile_combo_box.currentTextChanged.connect(lambda: self._update_selected_bar_profile(
            self.get_currently_selected_profile()
        ))

        bar_properties_layout.addWidget(profile_container)
        splitter.addWidget(bar_properties_container)

        # -- Charges
        self.bar_charges_layout, bar_charges_container = create_layout_and_container(QtWidgets.QVBoxLayout(),
                                                                                     QtWidgets.QWidget())

        self.bar_charges_layout.addWidget(QtWidgets.QLabel("Charges:"))
        add_bar_charge_button = QtWidgets.QPushButton("New charge")
        add_bar_charge_button.pressed.connect(lambda: self._add_distributed_charge_to_selected_bar())

        self.bar_charges_layout.addWidget(add_bar_charge_button)
        bar_properties_layout.addWidget(bar_charges_container)

        # -- Punctual Force
        self.bar_punctual_forces_layout, bar_punctual_forces_container = create_layout_and_container(
            QtWidgets.QVBoxLayout(),
            QtWidgets.QWidget())

        self.bar_punctual_forces_layout.addWidget(QtWidgets.QLabel("Punctual Forces:"))
        add_punctual_force_button = QtWidgets.QPushButton("New punctual force")
        add_punctual_force_button.pressed.connect(
            lambda: self._add_punctual_force_to_selected_bar()
        )

        self.bar_punctual_forces_layout.addWidget(add_punctual_force_button)
        bar_properties_layout.addWidget(bar_punctual_forces_container)

        # Node properties
        node_properties_layout, node_properties_container = create_layout_and_container(
            QtWidgets.QVBoxLayout(),
            QtWidgets.QGroupBox()
        )

        global node_properties
        node_properties = node_properties_container

        node_properties_layout.addWidget(QtWidgets.QLabel("Node properties"))
        node_properties_layout.addSpacing(5)
        # -- Coordinates
        node_coords_layout, node_coords_container = create_layout_and_container(QtWidgets.QHBoxLayout(),
                                                                                QtWidgets.QWidget())

        def create_coordinate(self, label_text, node_coords_layout):
            """
            Creates label and textbox for coordinates
            :param self: main_window
            :param label_text: text to display in the label
            :param node_coords_layout: layout to add the components to
            :return: PlainTextBox
            """
            # Label to display the coordinate attached to the textbox
            label = QtWidgets.QLabel(label_text)
            # Textbox to hold the coordinate value
            text_item = LineEdit()
            # Can be resized in width, but not in height
            text_item.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                    QtWidgets.QSizePolicy.Maximum)

            # associated axis to update the node coordinates qhen text_item text changes
            associated_axis = None
            if label_text.startswith("x"):
                associated_axis = "x"
            elif label_text.startswith("y"):
                associated_axis = "y"
            else:
                raise ValueError(
                    f"Error: label_text must begin with the letter of an axis. Current value is {label_text}")

            # Update node position when text chages
            text_item.textChanged.connect(lambda:
                                          self._update_selected_node_position(text_item, text_item.text(),
                                                                              associated_axis))

            # Resize ratio
            node_coords_layout.addWidget(label, 1)
            node_coords_layout.addWidget(text_item, 4)

            return text_item

        # ---- x coordinate
        self.x_coordinate = create_coordinate(self, "x:", node_coords_layout)
        # ---- y coordinate
        self.y_coordinate = create_coordinate(self, "y:", node_coords_layout)
        # ---- z coordinate
        # self.z_coordinate = create_coordinate("z:", node_coords_layout)

        # Initialize textboxes
        self.update_coordinates("---", "---", "---")

        node_properties_layout.addWidget(QtWidgets.QLabel("Coordinates"))
        node_properties_layout.addWidget(node_coords_container)

        # -- Support
        support_layout, support_container = create_layout_and_container(QtWidgets.QHBoxLayout(), QtWidgets.QWidget())

        self.support_combo_box = QtWidgets.QComboBox()
        self.support_combo_box.addItems([
            "NONE", "ROLLER_X", "ROLLER_Y", "PINNED", "FIXED"
        ])
        self.support_combo_box.currentTextChanged.connect(lambda: self._update_selected_node_support(
            self.support_combo_box.currentText()
        ))

        support_layout.addWidget(QtWidgets.QLabel("Support: "))
        support_layout.addWidget(self.support_combo_box)

        node_properties_layout.addWidget(support_container)

        splitter.addWidget(node_properties_container)

        # -- TextBox Info
        self.node_info_text_box = QtWidgets.QPlainTextEdit()
        self.node_info_text_box.setReadOnly(True)
        node_properties_layout.addWidget(self.node_info_text_box)

        # Widget to act as a separator. It allows the widget in the bars to be compact
        separator = QtWidgets.QWidget()
        separator.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                QtWidgets.QSizePolicy.Minimum)

        splitter.addWidget(separator)

        # Add the dock to the main window
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, properties_dock)

    def update_coordinates(self, x, y, z):
        """
        Sets the corresponding textboxes to the specified values.
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :return:
        """
        self.x_coordinate.setText(str(x))
        self.x_coordinate.setCursorPosition(0)

        self.y_coordinate.setText(str(y))
        self.y_coordinate.setCursorPosition(0)
        # self.z_coordinate.setPlainText(str(z))

    def get_currently_selected_material(self):
        """
        Provides the material information for creating bars
        :return: material name
        """
        material_text = self.material_combo_box.currentText()
        material_text = material_text.replace(" ", "")
        return material_text

    def get_currently_selected_profile(self):
        """
        Provides the profile information for creating bars
        :return: tuple of profile name and its number
        """
        profile_text = self.profile_combo_box.currentText()
        return tuple(profile_text.split(" "))

    def _populate_material_combobox(self):
        """
        Provides the items to populate the material combobox with the information in the database
        """
        # Clear items in the combobox
        self.material_combo_box.clear()
        # Establish connection to the database
        db_connection = db.create_connection()
        # Select information from database
        query = """
        SELECT name FROM materials ORDER BY name;
        """
        # Execute query
        query_results = db.execute_read_query(db_connection, query)
        # Add elements to a list that will hold the user options
        items = []
        for tup in query_results:
            # query_results is a list of tuples, retrieve the wanted information
            material = tup[0]
            if material not in items:
                items.append(material)

        # Add items to the combobox
        self.material_combo_box.addItems(items)

    def _populate_profile_combobox(self):
        """
        Provides the items to populate the profile combobox with the information in the database
        """
        # Clear items in the combobox
        self.profile_combo_box.clear()
        # Establish connection to the database
        db_connection = db.create_connection()
        # Select information from database
        query = """
        SELECT name, name_number FROM profiles ORDER BY name, name_number;
        """
        # Execute query
        query_results = db.execute_read_query(db_connection, query)
        # Add elements to a list that will hold the user options
        items = []
        for tup in query_results:
            # query_results is a list of tuples, retrieve the wanted information
            profile = f"{tup[0]} {tup[1]}"
            if profile not in items:
                items.append(profile)

        # Add items to the combobox
        self.profile_combo_box.addItems(items)

    def _create_status_bar(self):
        """
        Creates the status bar in the main window
        """
        self.status_bar = self.statusBar()
        # Temporary message
        self.status_bar.showMessage("Ready", 3000)

        # Permanent message
        self.current_mode_message = QtWidgets.QLabel("NORMAL MODE")
        self.status_bar.addPermanentWidget(self.current_mode_message)

    def _create_context_menu(self):
        """
        Creates right-click menus
        :return:
        """
        # ========== CENTRAL WIDGET ==========
        # Set contextMenuPolicy
        self.central_widget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        # Widget to act as a separator in context menus, since the method .addSeparator can't be used
        separator = QtWidgets.QAction(self)
        separator.setSeparator(True)

        # Populate widget with action
        self.central_widget.addAction(self.enable_node_mode_action)
        self.central_widget.addAction(self.enable_bar_mode_action)
        self.central_widget.addAction(self.solve_structure_action)
        # TODO Borrar, este separador, solo esta para motivos de documentacion
        self.central_widget.addAction(separator)
        self.central_widget.addAction(self.enable_bar_mode_action)

    def _connect_actions(self):
        """
        Connects the actions to functions
        """
        self.new_file_action.triggered.connect(self.new_file)
        self.exit_action.triggered.connect(self.close)

        self.enable_node_mode_action.triggered.connect(self.activate_draw_node_mode)
        self.enable_bar_mode_action.triggered.connect(self.activate_draw_bar_mode)
        self.solve_structure_action.triggered.connect(self.solve_structure)

    def _create_actions(self):
        """
        Creates actions for menus and toolbars
        """

        def _add_tip(item, tip):
            """
            Adds help tips
            :param item: item to which the help tip is going to be added
            :param tip: help tip to be shown
            :return:
            """
            # Status tip shows the tip in the status bar
            item.setStatusTip(tip)
            # Tool tip shows the tip when hovering mouse over widget in a toolbar
            item.setToolTip(tip)

        # ========== FILE ACTIONS ==========
        # Create actions
        self.new_file_action = QtWidgets.QAction("&New", self)
        self.open_file_action = QtWidgets.QAction("&Open", self)
        self.save_file_action = QtWidgets.QAction("&Save", self)
        self.exit_action = QtWidgets.QAction("&Exit", self)
        # Add help tips
        _add_tip(self.new_file_action, "Create a new file")
        _add_tip(self.open_file_action, "Open a file")
        _add_tip(self.save_file_action, "Save the current file")
        _add_tip(self.exit_action, "Close the application")

        # ========== EDIT ACTIONS ==========
        # Create actions
        self.copy_action = QtWidgets.QAction("&Copy", self)
        self.paste_action = QtWidgets.QAction("&Paste", self)
        self.cut_action = QtWidgets.QAction("Cu&t", self)
        # Add help tips

        # ========== STRUCTURE ACTIONS ==========
        # Create actions
        self.enable_node_mode_action = QtWidgets.QAction("&Node mode", self)
        self.enable_bar_mode_action = QtWidgets.QAction("&Bar mode", self)
        self.solve_structure_action = QtWidgets.QAction("&Solve", self)
        # Add shortcuts
        self.enable_node_mode_action.setShortcut("N")
        self.enable_bar_mode_action.setShortcut("B")
        self.solve_structure_action.setShortcut("Shift+S")
        # Add help tips
        _add_tip(self.enable_node_mode_action, "Create a new node")
        _add_tip(self.enable_bar_mode_action, "Create a new bar")
        _add_tip(self.solve_structure_action, "Solves the structure")

        # ========== HELP ACTIONS ==========
        # Create actions
        self.help_content_action = QtWidgets.QAction("&Help content", self)
        self.about_action = QtWidgets.QAction("&About", self)
        # Add help tips


class Bar(QtWidgets.QGraphicsLineItem):
    error_convert_value_to_string = QtCore.pyqtSignal()

    def __init__(self, main_window, node_origin, node_end):
        self.main_window = main_window

        # Origin point
        self.node_origin = node_origin
        self.x1_scene = node_origin.draw_x_pos + node_origin.radius / 2
        self.y1_scene = node_origin.draw_y_pos + node_origin.radius / 2
        self.node_origin.signals.position_changed.connect(lambda: self.update_point_position(
            "origin",
            self.node_origin.x_scene,
            self.node_origin.y_scene
        ))
        # End Point
        self.node_end = node_end
        self.x2_scene = node_end.draw_x_pos + node_end.radius / 2
        self.y2_scene = node_end.draw_y_pos + node_end.radius / 2
        self.node_end.signals.position_changed.connect(lambda: self.update_point_position(
            "end",
            self.node_end.x_scene,
            self.node_end.y_scene
        ))

        super().__init__(self.x1_scene, self.y1_scene, self.x2_scene, self.y2_scene)

        # Appearance
        self.color = NORMAL_COLOR
        self.drawn_thickness = 4
        pen = QtGui.QPen(NORMAL_COLOR, self.drawn_thickness)
        self.setPen(pen)

        self.setAcceptHoverEvents(True)

        # Distributed charges
        self.distributed_charges_container = QtWidgets.QGroupBox()
        self.distributed_charges_layout = QtWidgets.QVBoxLayout()
        self.distributed_charges_container.setLayout(self.distributed_charges_layout)
        # Do NOT change the name, it is hardcoded in other function to be able to retrieve this object
        self.distributed_charges_container.setObjectName("distributed_charges_container")

        # Punctual forces
        self.punctual_forces_container = QtWidgets.QGroupBox()
        self.punctual_forces_layout = QtWidgets.QVBoxLayout()
        self.punctual_forces_container.setLayout(self.punctual_forces_layout)
        # Do NOT change the name, it is hardcoded in other function to be able to retrieve this object
        self.punctual_forces_container.setObjectName("punctual_forces_container")

        # Bar logic
        bar_name = f"B_{self.x1_scene}_{self.y1_scene}_{self.x2_scene}_{self.y2_scene}"
        origin_node_logic = node_origin.node_logic
        end_node_logic = node_end.node_logic
        material = self.main_window.get_currently_selected_material()
        profile = self.main_window.get_currently_selected_profile()
        self.bar_logic = st.Bar(bar_name, origin_node_logic, end_node_logic,
                                material, profile)

        # Signals
        self.signals = BarSignals()

    def change_bar_color(self, color="normal"):
        """
        Changes the color of the node in a standardized way
        :param color: string representing the color to set
        """
        if color == "hover":
            new_color = HOVER_COLOR_NORMAL_MODE
        elif color == "selected":
            new_color = SELECTED_COLOR
        elif color == "hover_bar":
            new_color = HOVER_COLOR_BAR_MODE
        else:
            new_color = NORMAL_COLOR

        new_pen = QtGui.QPen(new_color, self.drawn_thickness)
        # Actually change the color
        self.setPen(new_pen)
        # Store logically the currently used color
        self.color = new_color

    def change_origin_node_color(self, color="normal"):
        if color == "origin_node":
            new_color = ORIGIN_NODE_COLOR
        else:
            new_color = NORMAL_COLOR

        # Actually change the node color
        self.node_origin.setBrush(new_color)
        # Store logically the currently used color
        self.node_origin.color = new_color

    def update_point_position(self, point_reference, new_x_scene, new_y_scene):
        """
        Updates the position of one of the points of the line. This function is called when origin or end points
        change position
        :param point_reference: Point to update. "origin" for origin node, something else for end node
        :param new_x_scene: new x position in scene coordinates
        :param new_y_scene: new y position in scene coordinates
        :return:
        """
        if point_reference == "origin":
            self.x1_scene = new_x_scene
            self.y1_scene = new_y_scene
        else:
            self.x2_scene = new_x_scene
            self.y2_scene = new_y_scene

        self.setLine(self.x1_scene, self.y1_scene, self.x2_scene, self.y2_scene)

    def add_distributed_charge(self):
        """
        Adds the graphical representation of a distributed charge. Called from a button
        :return:
        """
        # Give a unique name in order to be able to retrieve its values
        base_name = "dc"
        dc_name = fs.get_random_name(base_name)
        current_charges = get_widgets_in_layout(self.distributed_charges_layout)

        current_charges_dict_names = list(
            map(lambda x: x.dc_name,
                current_charges)
        )

        while dc_name in current_charges_dict_names:
            dc_name = fs.get_random_name(base_name)

        # Create the graphical component for distributed charges and add it to the GUI
        dc = BarDistributedCharge(self, dc_name).get_widget()
        self.distributed_charges_layout.addWidget(dc)
        print(f"New charge added to bar {self.bar_logic.name}")

    def add_punctual_force(self):
        """
        Adds the graphical representation of a punctual force. Called from a button
        :return:
        """
        # Give a unique name in order to be able to retrieve its values
        base_name = "pf"
        pf_name = fs.get_random_name(base_name)
        current_punctual_forces = get_widgets_in_layout(self.punctual_forces_layout)

        current_forces_dict_names = list(
            map(lambda x: x.pf_name,
                current_punctual_forces)
        )

        while pf_name in current_forces_dict_names:
            pf_name = fs.get_random_name(base_name)

        # Create the graphical component for punctual force and add it to the GUI
        pf = BarPunctualForce(self, pf_name).get_widget()
        self.punctual_forces_layout.addWidget(pf)
        print(f"New punctual force added to bar {self.bar_logic.name}")

    def update_material(self, new_material):
        """
        Changes the material of the bar
        :param new_material: name of the new material
        :return:
        """
        self.bar_logic.set_material(new_material)

    def update_profile(self, new_profile):
        """
        Changes the profile of the bar
        :param new_profile: tuple containing name and number of the profile
        :return:
        """
        self.bar_logic.set_profile(new_profile[0], new_profile[1])

    def update_distributed_charge(self, dc_name, dc_type, value, direction):
        """
        Function called from update charge in class BarDistributedCharge
        :param direction:
        :param dc_name: name of the distributed charge in the bar logic dictionary
        :param dc_type: distributed charge type to update
        :param value: value to update with
        """

        def update(dc_type, value, direction):
            current_distributed_charges = self.bar_logic.get_distributed_charges()

            none_values = len(
                [x for x in [dc_type, value, direction] if x is None]
            )

            if dc_name not in current_distributed_charges.keys():
                dc = st.DistributedCharge(
                    dc_type=dc_type,
                    max_value=value,
                    direction=direction
                )

                self.bar_logic.add_distributed_charge(dc, dc_name)
            else:
                dc_to_modify = current_distributed_charges.get(dc_name)
                dc_to_modify.set_dc_type(dc_type)
                dc_to_modify.set_max_value(value)
                dc_to_modify.set_direction(direction)

        update(dc_type, value, direction)

    def update_punctual_force(self, pf_name, value, origin_end_factor, direction):
        """
        Function called from update punctual force in class BarPunctualForce
        :param origin_end_factor: distance from origin to the point in which the force is applied. Measured in per unit.
        :param direction: Direction in which the force is applied
        :param pf_name: name of the punctual force in the bar logic dictionary
        :param value: magnitude of the force
        """
        current_punctual_forces = self.bar_logic.get_punctual_forces()

        if pf_name not in current_punctual_forces.keys():
            pf = st.PunctualForceInBar(
                value=value,
                origin_end_factor=origin_end_factor,
                direction=direction
            )

            self.bar_logic.add_punctual_force(pf, pf_name)
        else:
            pf_to_modify = current_punctual_forces.get(pf_name)
            pf_to_modify.set_value(value)
            pf_to_modify.set_origin_end_factor(origin_end_factor)
            pf_to_modify.set_direction(direction)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse enters the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE and \
                (active_structure_element is not self or (
                        active_structure_element is self and self.color is NORMAL_COLOR
                )):
            self.change_bar_color("hover")

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse exits the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE and \
                active_structure_element is not self:
            self.change_bar_color()
        elif application_mode == ApplicationMode.BAR_MODE:
            self.change_bar_color()

    # Mouse clicks need to be handled from GraphicsScene class


class BarDistributedCharge(QtWidgets.QWidget):
    def __init__(self, bar_attached_to, dc_name):
        """
        :param bar_attached_to: bar to which an instance of distributed charge is applied to
        :param dc_name: name to store the distributed charge in the bar logic dictionary
        """
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout()
        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(self.layout)
        self.widget.dc_name = dc_name
        self.widget.bar_attached_to = bar_attached_to

        # CHARGE TYPE
        self.charge_type_combo_box = QtWidgets.QComboBox()
        self.populate_charge_type_combo_box()
        self.charge_type_combo_box.currentTextChanged.connect(
            lambda: self._update_charge()
        )

        # CHARGE VALUE
        self.charge_value_text_box = LineEdit()
        # Can be resized in width, but not in height
        self.charge_value_text_box.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                                 QtWidgets.QSizePolicy.Maximum)

        self.charge_value_text_box.textChanged.connect(
            lambda: self._update_charge()
        )

        # REMOVE BUTTON
        self.remove_charge_button = SmallButton("X")
        self.remove_charge_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Minimum)
        self.remove_charge_button.pressed.connect(self.widget.deleteLater)

        self.layout.addWidget(self.charge_type_combo_box)
        self.layout.addWidget(self.charge_value_text_box)
        self.layout.addWidget(self.remove_charge_button)

    def get_widget(self):
        """
        Simple instanciating this class didn't provide a widget. This function is meant to fix that problem
        :return: instance of the class widget
        """
        return self.widget

    def populate_charge_type_combo_box(self):
        # Clear all previous items, if any
        self.charge_type_combo_box.clear()
        # Get full name of distributed charge types
        distributed_charge_types = list(map(str, st.DistributedChargeType))
        # Get only specific name of distributed charge type
        useful_distributed_charges = list(map(lambda x: x.split(".")[1], distributed_charge_types))
        # This is just a workaround because I couldn't connect the creation of the object to the update method
        # This forces the user to, at least, change the value of the combobox once  and, hence, call the update method
        distributed_charges = [""]

        distributed_charges.extend(
            useful_distributed_charges
        )

        # Populate combobox
        self.charge_type_combo_box.addItems(distributed_charges)

    def _update_charge(self):
        """
        This function is connected to the charge type combobox
        """
        if self.charge_type_combo_box.itemText(0) == "":
            self.charge_type_combo_box.removeItem(0)

        dc_type_string = self.charge_type_combo_box.currentText()

        if dc_type_string == "SQUARE":
            dc_type = st.DistributedChargeType.SQUARE
            direction = (0, 1, 0)
        elif dc_type_string == "PARALLEL_TO_BAR":
            dc_type = st.DistributedChargeType.PARALLEL_TO_BAR
            direction = (0, 1, 0)
        else:
            raise ValueError(f"Error: Distributed charge type {dc_type_string} has not been implemented")

        value = self.charge_value_text_box.text()

        if self.charge_value_text_box.text().strip() == "":
            value = make_qline_edit_value_zero(self.charge_value_text_box)

        try:
            float(value)
        except ValueError:
            value = make_qline_edit_value_zero(self.charge_value_text_box)

        self.widget.bar_attached_to.update_distributed_charge(self.widget.dc_name,
                                                              dc_type,
                                                              value,
                                                              direction)


class BarPunctualForce(QtWidgets.QWidget):
    def __init__(self, bar_attached_to, pf_name):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout()
        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(self.layout)
        self.widget.pf_name = pf_name
        self.widget.bar_attached_to = bar_attached_to

        # X Component
        self.layout.addWidget(QtWidgets.QLabel("Fx: "), 1)
        self.x_component_text = LineEdit()
        self.layout.addWidget(self.x_component_text, 4)
        self.x_component_text.textChanged.connect(
            lambda: self._update_punctual_force()
        )

        # Y Component
        self.layout.addWidget(QtWidgets.QLabel("Fy: "), 1)
        self.y_component_text = LineEdit()
        self.layout.addWidget(self.y_component_text, 4)
        self.y_component_text.textChanged.connect(
            lambda: self._update_punctual_force()
        )

        # Origin to end factor
        self.layout.addWidget(QtWidgets.QLabel("m: "), 1)
        self.origin_end_factor_in_meters = LineEdit()
        self.layout.addWidget(self.origin_end_factor_in_meters, 4)
        self.origin_end_factor_in_meters.textChanged.connect(
            lambda: self._update_punctual_force()
        )

        # REMOVE BUTTON
        self.remove_punctual_force_button = SmallButton("X")
        self.remove_punctual_force_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                                        QtWidgets.QSizePolicy.Minimum)
        self.remove_punctual_force_button.pressed.connect(self.widget.deleteLater)
        self.layout.addWidget(self.remove_punctual_force_button, 1)

    def get_widget(self):
        """
        Simple instantiating this class didn't provide a widget. This function is meant to fix that problem
        :return: instance of the class widget
        """
        return self.widget

    def _update_punctual_force(self):
        """
        This function is connected to the textboxes
        """

        def _parse_float_qlineedit(qLineEdit: QtWidgets.QLineEdit):
            txt = qLineEdit.text()
            if txt.strip() == "":
                parsed_value = make_qline_edit_value_zero(qLineEdit)

            try:
                parsed_value = float(txt)
            except ValueError:
                parsed_value = make_qline_edit_value_zero(qLineEdit)

            return parsed_value

        x_component = _parse_float_qlineedit(self.x_component_text)
        y_component = _parse_float_qlineedit(self.y_component_text)
        origin_end_meters = _parse_float_qlineedit(self.origin_end_factor_in_meters)

        bar_length = self.widget.bar_attached_to.bar_logic.length()

        if origin_end_meters > bar_length:
            origin_end_meters = bar_length
            self.origin_end_factor_in_meters.setText(str(origin_end_meters))
        elif origin_end_meters < 0:
            origin_end_meters = 0
            self.origin_end_factor_in_meters.setText(str(origin_end_meters))

        origin_end_factor = origin_end_meters / self.widget.bar_attached_to.bar_logic.length()

        if origin_end_factor < 0 or origin_end_factor > 1:
            return

        force_vector = np.array([x_component, y_component, 0])
        value = np.linalg.norm(force_vector)

        direction = tuple(force_vector / value)

        self.widget.bar_attached_to.update_punctual_force(pf_name=self.widget.pf_name,
                                                          value=value,
                                                          origin_end_factor=origin_end_factor,
                                                          direction=direction)


class NodeSignals(QtWidgets.QGraphicsObject):
    """
    The class from which inherits Node cannot emit signals. This class is aimed to hold the needed custom signals
    of the Node class and emit them when necessary
    """
    position_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()


class BarSignals(QtWidgets.QGraphicsObject):
    """
    The class from which inherits Bar cannot emit signals. This class is aimed to hold the needed custom signals
    of the Bar class and emit them when necessary
    """
    error_convert_value_to_string = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()


class Node(QtWidgets.QGraphicsEllipseItem):
    """
    Class that holds all structure node information, as well as its graphic behavior
    """

    def __init__(self, main_window, x_scene, y_scene, radius):
        """

        :param main_window: main_window of the application
        :param x_scene: x position in scene coordinates
        :param y_scene: y position in scene coordinates
        :param radius: radius of the graphical representation
        """
        # 0, 0 are x and y coordinates in ITEM COORDINATES
        super().__init__(0, 0, radius, radius)

        self.main_window = main_window

        # Radius of the node
        self.radius = radius
        # x and y coordinates in scene reference system
        self.x_scene, self.y_scene = x_scene, y_scene
        # x and y coordinates in centered reference system
        self.x_centered, self.y_centered = self.main_window.centered_coordinates(x_scene, y_scene)

        # x and y coordinates in scene reference system to draw the node by its center
        self.draw_x_pos, self.draw_y_pos = x_scene - self.radius, self.y_scene - self.radius

        self.setPos(self.draw_x_pos, self.draw_y_pos)

        # Currently used color
        self.color = NORMAL_COLOR

        # Change the color to the normal one
        self.change_node_color("normal")

        # Needed for hover events to take place
        self.setAcceptHoverEvents(True)

        # Node logic
        # Position based on user click input
        x_meter = self.x_centered * PX_TO_METER
        y_meter = self.y_centered * PX_TO_METER

        node_name = "N_" + str(x_scene) + "_" + str(y_scene)
        # Structure node object
        self.node_logic = st.Node(node_name, (x_meter, y_meter, 0))

        # Support
        self.label_support_image = QtWidgets.QLabel()
        self.label_support_image.setStyleSheet("background-color:#00FFFFFF")
        self.main_window.scene.addWidget(self.label_support_image)
        self.update_support("NONE")

        # Signal
        self.signals = NodeSignals()

    def update_position(self, new_x_centered_in_meters=None, new_y_centered_in_meters=None):
        """
        This method is called from the coordinate textBoxes. Changes the position of the node the the specified one.
        If one of the parameters is omitted or set to None, then the current position is applied.
        :param new_x_centered_in_meters: New x position, in centered coordinates and in meters
        :param new_y_centered_in_meters: New y position, in centered coordinates and in meters
        :return:
        """
        # If no x coordinate is specified, then use the current value
        if new_x_centered_in_meters is None:
            new_x_centered_in_meters = self.node_logic.x()
            new_x_centered = self.x_centered
        # Otherwise, get the pixel position in centered coordinates
        else:
            new_x_centered = int(new_x_centered_in_meters * METER_TO_PX)
            self.x_centered = new_x_centered

        # If no y coordinate is specified, then use the current value
        if new_y_centered_in_meters is None:
            new_y_centered_in_meters = self.node_logic.y()
            new_y_centered = self.y_centered
        # Otherwise, get the pixel position in centered coordinates
        else:
            new_y_centered = int(new_y_centered_in_meters * METER_TO_PX)
            self.y_centered = new_y_centered

        if new_x_centered_in_meters != self.node_logic.x() or new_y_centered_in_meters != self.node_logic.y():
            # Convert centered coordinates to scene ones in order to be able to draw them correctly
            new_x_scene, new_y_scene = self.main_window.scene_coordinates(new_x_centered, new_y_centered)

            # Pixel coordinates must be integer
            new_x_scene = int(new_x_scene)
            new_y_scene = int(new_y_scene)

            # Scene coordinates
            self.x_scene, self.y_scene = new_x_scene, new_y_scene
            # Update meter coordinates in node logic
            self.node_logic.set_position((new_x_centered_in_meters, new_y_centered_in_meters, 0))

            # Modify scene coordinates to draw the node at its center
            self.draw_x_pos = int(new_x_scene - self.radius / 2)
            self.draw_y_pos = int(new_y_scene - self.radius / 2)

            try:
                # Pack into a point the draw coordinates
                draw_pos = QtCore.QPoint(self.draw_x_pos, self.draw_y_pos)
            except OverflowError:
                # If the coordinates are to big, then cancel the operation
                return

            # Change node position
            self.setPos(draw_pos)
            # Change support position
            current_support = str(self.node_logic.support).split(".")[1]
            self.update_support(current_support)
            # This signal communicates with Bar to change its position
            self.signals.position_changed.emit()

    def update_support(self, support_name):
        if support_name == "ROLLER_X":
            support = st.Support.ROLLER_X
            support_image = QtGui.QPixmap(":roller_x_support.svg")
            image_x_coordinate = int(self.draw_x_pos - support_image.width() / 2 + self.radius / 2)
            image_y_coordinate = int(self.draw_y_pos + self.radius)
        elif support_name == "ROLLER_Y":
            support = st.Support.ROLLER_Y
            support_image = QtGui.QPixmap(":roller_y_support.svg")
            image_x_coordinate = int(self.draw_x_pos + self.radius)
            image_y_coordinate = int(self.draw_y_pos - self.radius)
        elif support_name == "PINNED":
            support = st.Support.PINNED
            support_image = QtGui.QPixmap(":pinned_support.svg")
            image_x_coordinate = int(self.draw_x_pos - support_image.width() / 2 + self.radius / 2)
            image_y_coordinate = int(self.draw_y_pos + self.radius)
        elif support_name == "FIXED":
            support = st.Support.FIXED
            support_image = QtGui.QPixmap(":fixed_support.svg")
            image_x_coordinate = int(self.draw_x_pos - support_image.width() / 2 + self.radius / 2)
            image_y_coordinate = int(self.draw_y_pos + self.radius)
        else:
            support = st.Support.NONE
            support_image = QtGui.QPixmap()
            image_x_coordinate = int(self.draw_x_pos - support_image.width() / 2 + self.radius / 2)
            image_y_coordinate = int(self.draw_y_pos + self.radius)

        self.label_support_image.setPixmap(support_image)
        self.label_support_image.adjustSize()
        self.label_support_image.move(image_x_coordinate, image_y_coordinate)

        self.node_logic.set_support(support)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse enters the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE and \
                (active_structure_element is not self or (
                        active_structure_element is self and self.color is NORMAL_COLOR
                )):
            self.change_node_color("hover")
        elif application_mode == ApplicationMode.BAR_MODE:
            self.change_node_color("hover_bar")

    def change_node_color(self, color="normal"):
        """
        Changes the color of the node in a standarized way
        :param color: string representing the color to set
        """
        if color == "hover":
            new_color = HOVER_COLOR_NORMAL_MODE
        elif color == "selected":
            new_color = SELECTED_COLOR
        elif color == "hover_bar":
            new_color = HOVER_COLOR_BAR_MODE
        else:
            new_color = NORMAL_COLOR

        # Actually change the node color
        self.setBrush(new_color)
        # Store logically the currently used color
        self.color = new_color

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse exits the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE and \
                active_structure_element is not self:
            self.change_node_color()
        elif application_mode == ApplicationMode.BAR_MODE:
            self.change_node_color()

    # Mouse clicks need to be handled from GraphicsScene class


class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Reimplementation of QGraphicsScene in order to be able to handle the mouse events
    """

    def __init__(self, main_window, parent=None):
        """

        :param main_window: main window of the application
        :param parent:
        """
        super().__init__(parent)
        self.main_window = main_window

        # Dictionary to track the mouse clicks when in bar mode in order to store
        # the points of the bar
        self.nodes_for_bar_creation = None

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        """
        Function that is triggered every time a key is released
        :param event:
        :return:
        """
        global application_mode
        global active_structure_element

        # Go back to normal mode
        if event.key() == QtCore.Qt.Key_Escape:
            self.main_window.set_current_mode(ApplicationMode.NORMAL_MODE)
        # Delete active element
        elif event.key() == QtCore.Qt.Key_Delete and application_mode == ApplicationMode.NORMAL_MODE:
            if type(active_structure_element) is Node:
                self.main_window.delete_node(active_structure_element)
            elif type(active_structure_element) is Bar:
                self.main_window.delete_bar(active_structure_element)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        """
        Function that is triggered every time the left mouse button is released
        :param event:
        :return:
        """
        # NODE MODE functionality
        if application_mode == ApplicationMode.NODE_MODE:
            self.main_window.draw_node(NODE_RADIUS)
        # BAR MODE functionality
        elif application_mode == ApplicationMode.BAR_MODE:
            # Get the item under the cursor
            item = self._get_item_at_mouse_position(event)

            if type(item) is Node:
                if self.nodes_for_bar_creation is None:
                    self.nodes_for_bar_creation = {}
                    self.nodes_for_bar_creation["node_origin"] = item
                else:
                    self.nodes_for_bar_creation["node_end"] = item
            else:
                self.nodes_for_bar_creation = None

            # Create bar
            if self.nodes_for_bar_creation is not None and len(self.nodes_for_bar_creation) >= 2:
                node_origin = self.nodes_for_bar_creation.get("node_origin")
                node_end = self.nodes_for_bar_creation.get("node_end")

                self.main_window.draw_bar(node_origin, node_end)
                self.nodes_for_bar_creation = None

        # NORMAL MODE functionality
        elif application_mode == ApplicationMode.NORMAL_MODE:
            # Get the item under the cursor
            item = self._get_item_at_mouse_position(event)

            unset_active_structure_element()
            # Select current element as active
            set_active_structure_element(item)

            self._update_active_element_info()

    def _update_active_element_info(self):
        """
        Updates the interface elements to show information about the active element
        :return:
        """
        # NODE item
        if type(active_structure_element) is Node:
            # Change node color
            active_structure_element.change_node_color("selected")
            # Get coordinates of the node
            node_logic = active_structure_element.node_logic
            x = node_logic.x()
            y = node_logic.y()
            z = node_logic.z()
            # Show coordinates in the textboxes
            self.main_window.update_coordinates(x, y, z)
            # Show support
            support_name = active_structure_element.node_logic.support
            support_name = str(support_name).split(".")
            self.main_window.support_combo_box.setCurrentText(
                support_name[1]
            )
            # Show displacement
            node_displacement = node_logic.get_displacement()
            self.main_window.node_info_text_box.clear()
            self.main_window.node_info_text_box.setPlainText(
                "DISPLACEMENTS" + "\n"
                                  "==============" + "\n"
                                                     f"x: {node_displacement.get('x')} [m]" + "\n"
                                                                                              f"y: {node_displacement.get('y')} [m]" + "\n"
                                                                                                                                       f"ang.: {node_displacement.get('angle')} [rad.]" + "\n"
            )

            # Show reactions
            node_reactions = node_logic.get_reactions()
            reaction_label = "REACTIONS"
            if not node_logic.has_support():
                reaction_label = "FORCES"

            self.main_window.node_info_text_box.appendPlainText(
                reaction_label + "\n"
                                 "==========" + "\n"
                                 f"x: {node_reactions.get('x')} [m]" + "\n"
                                 f"y: {node_reactions.get('y')} [m]" + "\n"
                                 f"M.: {node_reactions.get('momentum')} [N/m]" + "\n"
            )

            self._hide_last_bar_charges_info_from_gui()
            self._hide_last_bar_punctual_forces_info_from_gui()
        # BAR item
        elif type(active_structure_element) is Bar:
            # Change bar color
            active_structure_element.change_bar_color("selected")
            active_structure_element.change_origin_node_color("origin_node")
            material_name = active_structure_element.bar_logic.get_material().name
            profile_name = " ".join([active_structure_element.bar_logic.get_profile().name,
                                     str(active_structure_element.bar_logic.get_profile().name_number)])

            self.main_window.material_combo_box.setCurrentText(material_name)
            self.main_window.profile_combo_box.setCurrentText(profile_name)
            self.main_window.update_coordinates("---", "---", "---")
            self.main_window.support_combo_box.setCurrentText("NONE")

            # Distributed charges
            # Hide currently shown
            self._hide_last_bar_charges_info_from_gui()
            # Add active element
            if active_structure_element.distributed_charges_container not in get_widgets_in_layout(
                    self.main_window.bar_charges_layout):
                self.main_window.bar_charges_layout.addWidget(active_structure_element.distributed_charges_container)
            else:
                active_structure_element.distributed_charges_container.show()

            # Punctual forces
            # Hide currently shown
            self._hide_last_bar_punctual_forces_info_from_gui()
            # Add active element
            if active_structure_element.punctual_forces_container not in get_widgets_in_layout(
                    self.main_window.bar_punctual_forces_layout):
                self.main_window.bar_punctual_forces_layout.addWidget(
                    active_structure_element.punctual_forces_container)
            else:
                active_structure_element.punctual_forces_container.show()

        # No item
        elif active_structure_element is None:
            self.main_window.update_coordinates("---", "---", "---")
            self.main_window.support_combo_box.setCurrentText("NONE")
            self._hide_last_bar_charges_info_from_gui()
            self._hide_last_bar_punctual_forces_info_from_gui()

    def _hide_last_bar_charges_info_from_gui(self):
        """
        This function is used to update the active element information shown in the GUI. It
        HIDES the already included distributed charges widgets in the dock.

        The original idea was to remove them from the dock, but its children caused visual problems and,
        when deleted from the layout, they disappeared also in the Bar object.
        """
        # Get elements that are currently contained in bar_charges_layout
        elements_in_bar_charges_layout = get_widgets_in_layout(self.main_window.bar_charges_layout)
        # Use the name of the widget to filter the ones that must be hidden
        elements_to_hide = list(filter(lambda x: x.objectName() == "distributed_charges_container",
                                       elements_in_bar_charges_layout))

        # Hide the filtered elements
        if len(elements_to_hide) > 0:
            for element in elements_to_hide:
                element.hide()

    def _hide_last_bar_punctual_forces_info_from_gui(self):
        """
        This function is used to update the active element information shown in the GUI. It
        HIDES the already included punctual forces widgets in the dock.

        The original idea was to remove them from the dock, but its children caused visual problems and,
        when deleted from the layout, they disappeared also in the Bar object.
        """
        # Get elements that are currently contained in bar_charges_layout
        elements_in_punctual_forces_layout = get_widgets_in_layout(self.main_window.bar_punctual_forces_layout)
        # Use the name of the widget to filter the ones that must be hidden
        elements_to_hide = list(filter(lambda x: x.objectName() == "punctual_forces_container",
                                       elements_in_punctual_forces_layout))

        # Hide the filtered elements
        if len(elements_to_hide) > 0:
            for element in elements_to_hide:
                element.hide()

    def _get_item_at_mouse_position(self, event):
        """

        :param event: Mouse event
        :return: item under the cursor
        """
        # Get position where the release has happened
        position = event.scenePos()
        # Transform matrix is needed for itemAt method.
        # It is used the identity matrix in order not to change anything
        transform = QtGui.QTransform(1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1)

        # Get the item at event position
        item = self.itemAt(position, transform)

        return item

    def get_bars(self):
        bars = []
        for item in self.items():
            if type(item) is Bar:
                bars.append(item)

        return bars


class LineEdit(QtWidgets.QLineEdit):
    """
    Extends QLineEdit in order to be able to specify a sizeHint lesser than the default one
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 10)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 25)


class SmallButton(QtWidgets.QPushButton):
    """
    Extends QPushButton in order to be able to specify a sizeHint lesser than the default one
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(25, 25)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(25, 25)


class MplCanvas(FigureCanvasQTAgg):
    """
    Source: https://www.pythonguis.com/tutorials/plotting-matplotlib/
    """

    def __init__(self, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        # Create axes for each effort law
        self.axes_axile, self.axes_shear, self.axes_bending = self.fig.subplots(3, 1, sharex=True)

        # Label y axis
        self.axes_axile.set_ylabel("Axile force (N)")
        self.axes_shear.set_ylabel("Shear strength (N)")
        self.axes_bending.set_ylabel("Bending moment (N/m)")

        # Label x axis
        self.axes_bending.set_xlabel("Length (m)")
        super(MplCanvas, self).__init__(self.fig)

        # Create markers for each effort law. They are transparent on startup
        self.axile_marker, = self.axes_axile.plot([0], [0], marker="x", color="#FF000000", zorder=3)
        self.shear_marker, = self.axes_shear.plot([0], [0], marker="x", color="#FF000000", zorder=3)
        self.bending_marker, = self.axes_bending.plot([0], [0], marker="x", color="#FF000000", zorder=3)

        # Flag to signal whether the marker can be moved or not
        self.marker_can_be_moved = False

        # Create value text for each effort law. They are empty on startup
        self.axile_text = self.axes_axile.text(0, 0, "")
        self.shear_text = self.axes_shear.text(0, 0, "")
        self.bending_text = self.axes_bending.text(0, 0, "")

        # Connect self.mouse_movement function to mouse hover event
        self.fig.canvas.mpl_connect('motion_notify_event', self._mouse_movement)
        # Allow marker movement when clicking the mouse
        self.fig.canvas.mpl_connect('button_press_event', self._allow_marker_movement)
        # Deny marker movement when not clicking the mouse
        self.fig.canvas.mpl_connect('button_release_event', self._deny_marker_movement)

    def _allow_marker_movement(self, event):
        """
        Turns to True the flag that allows the marker movement and moves it to the cursor position
        """
        self.marker_can_be_moved = True
        self._move_marker(event)

    def _deny_marker_movement(self, event):
        """
        Turns to False the flag that allows the marker movement
        """
        self.marker_can_be_moved = False

    def _mouse_movement(self, event):
        """
        Snaps the marker to the plotted lines and shows its value when hovering the mouse
        """
        self._move_marker(event)

    def _move_marker(self, event):
        """
        Moves the marker and updates the text using the given mouse event
        """
        event_info = self._get_info_from_mouse_event(event)
        if event_info is not None:
            ax = event_info['ax']
            x_data = event_info['x_data']
            y_data = event_info['y_data']
            self._update_marker_and_text(ax, x_data, y_data)

    def _get_info_from_mouse_event(self, event):
        """
        Original from https://stackoverflow.com/questions/44679473/add-cursor-to-matplotlib
        Extracts the useful information from the given MouseEvent
        """
        # For MouseEvent event
        if isinstance(event, mpl.backend_bases.MouseEvent):
            # If the mouse is hovering over and axes
            if event.inaxes:
                # Get mouse point in data coordinates
                x_mouse_value, y_mouse_value = event.xdata, event.ydata
                # Get the axes over which the mouse is hovering
                ax = event.inaxes.axes
                # Get x and y plotted values
                x_axis = ax.lines[1].get_xdata()    # The effort information is the second line since the marker is the first one
                y_axis = ax.lines[1].get_ydata()    # The effort information is the second line since the marker is the first one

                # The x_value of the mouse must be in the plotted range in order for the data to be display correctly
                if x_axis[0] <= x_mouse_value <= x_axis[-1] and self.marker_can_be_moved:
                    # Get the index value for the closest x point in axis to the x cursor value
                    index = np.searchsorted(x_axis, [x_mouse_value])[0]
                    # Get x and y data coordinates
                    x_data_value = x_axis[index]
                    y_data_value = y_axis[index]

                    return {
                        "ax": ax,
                        "x_mouse": x_mouse_value,
                        "y_mouse": y_mouse_value,
                        "x_data": x_data_value,
                        "y_data": y_data_value,
                    }

        return None

    def _update_marker_and_text(self, ax, x_val, y_val):
        """
        Lower level function to update marker and text
        """
        # Get marker and update its position
        marker = ax.lines[0]
        marker.set_data([x_val], [y_val])
        # Stop marker being transparent
        self._make_marker_visible(marker)
        # Get text and update it
        text = ax.texts[0]
        text.set_text(f"x: {x_val:.4f}\ny: {y_val:.4f}")
        text.set_position((x_val, y_val))
        # Redraw the plot with the new marker position
        ax.figure.canvas.draw_idle()

    def _make_marker_visible(self, marker):
        marker.set_color("#FF0000AA")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    # window.showMaximized()
    sys.exit(app.exec_())
