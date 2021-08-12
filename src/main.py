import enum
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from src.modules import databaseutils as db
from src.modules import structures as st

# CONSTANTS
NODE_RADIUS = 10
Z_VALUE_NODES = 2
Z_VALUE_BARS = 1
Z_VALUE_AXIS = -1
# -- Colors
# ---- Normal color of the node
NORMAL_COLOR = QtGui.QColor(0, 0, 0)
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

        # Set active_structure_element to None
        active_structure_element = None


def set_active_structure_element(item):
    """
    Sets the active structure element to the specified one
    :param item: item that is desired to be the new active structure element
    """
    global active_structure_element
    active_structure_element = item


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
        draw_coordinates = [scene_position.x() - radius / 2, scene_position.y() - radius / 2]
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

            bar.setZValue(Z_VALUE_BARS)

            # Add node to scene
            self.scene.addItem(bar)

            # set_active_structure_element(bar)

    def delete_node(self, node):
        """
        Removes the specified node from the scene
        :param node: Node to be deleted
        """
        # Delete node
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
        edit_menu.addAction(self.create_support_action)
        edit_menu.addAction(self.create_charge_action)
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

    def _update_selected_node_position(self, text, axis):
        """
        This function is connected to the textboxes that represent the coordinates of the nodes
        :param text: current text of the textbox
        :param axis: axis in which the coordinate is going to change
        """
        try:
            # Parse the text to float
            new_pos = float(text)
        except ValueError:
            # If it cannot be parsed, then do not continue
            return

        global active_structure_element
        if type(active_structure_element) is Node:
            if axis == "x":
                active_structure_element.update_position(new_pos, None)
            elif axis == "y":
                active_structure_element.update_position(None, new_pos)
            # TODO implement Z if 3D structures

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
        global active_structure_element
        if type(active_structure_element) is Bar:
            active_structure_element.update_profile(profile)

    def _create_toolbars_and_docks(self):
        """
        Creates the toolbars and docks in the main window
        """
        # ========== STRUCTURE TOOLBAR ==========
        structure_toolbar = QtWidgets.QToolBar("Structure", self)

        structure_toolbar.addAction(self.enable_node_mode_action)
        structure_toolbar.addAction(self.enable_bar_mode_action)
        structure_toolbar.addAction(self.create_support_action)
        structure_toolbar.addAction(self.create_charge_action)

        self.addToolBar(QtCore.Qt.LeftToolBarArea, structure_toolbar)

        # ========== PROPERTIES DOCK ==========
        def create_layout_and_container():
            """
            QSplitter class doesn't allow to add layouts directly, so a workaround is needed.
            This function returns a layout to which widgets can be added and a single widget that
            holds that layout.
            :return: layout and container widget
            """
            layout = QtWidgets.QHBoxLayout()
            container = QtWidgets.QWidget()
            container.setLayout(layout)

            return layout, container

        properties_dock = QtWidgets.QDockWidget("Properties", self)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        properties_dock.setWidget(splitter)

        # Material
        material_layout, mat_container = create_layout_and_container()

        # -- Label material
        label_material = QtWidgets.QLabel("Material")
        material_layout.addWidget(label_material)

        # -- Material comboBox
        self.material_combo_box = QtWidgets.QComboBox()
        self.material_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self._populate_material_combobox()
        self.material_combo_box.currentTextChanged.connect(lambda: self._update_selected_bar_material(
            self.get_currently_selected_material()
        ))

        material_layout.addWidget(self.material_combo_box)

        splitter.addWidget(mat_container)

        # Profile
        profile_layout, profile_container = create_layout_and_container()
        # -- Label profile
        label_profile = QtWidgets.QLabel("Profile")
        profile_layout.addWidget(label_profile)

        # -- ComboBox profile
        self.profile_combo_box = QtWidgets.QComboBox()
        self.profile_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)
        profile_layout.addWidget(self.profile_combo_box)
        self._populate_profile_combobox()
        self.profile_combo_box.currentTextChanged.connect(lambda: self._update_selected_bar_profile(
            self.get_currently_selected_profile()
        ))

        splitter.addWidget(profile_container)

        # Properties
        # -- Node properties
        # ---- Coordinates
        node_coords_layout, node_coords_container = create_layout_and_container()

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
            text_item = PlainTextBox()
            # No wordwrap
            text_item.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            # No tabs inside the textbox
            text_item.setTabChangesFocus(True)
            # Disable scrollbar
            text_item.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            text_item.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

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
                                          self._update_selected_node_position(text_item.toPlainText(),
                                                                              associated_axis))

            # Resize ratio
            node_coords_layout.addWidget(label, 1)
            node_coords_layout.addWidget(text_item, 4)

            return text_item

        # -------- x coordinate
        self.x_coordinate = create_coordinate(self, "x", node_coords_layout)
        # -------- y coordinate
        self.y_coordinate = create_coordinate(self, "y", node_coords_layout)
        # -------- z coordinate
        # self.z_coordinate = create_coordinate("z", node_coords_layout)

        # Initialize textboxes
        self.update_coordinates(0, 0, 0)

        splitter.addWidget(QtWidgets.QLabel("Coordinates:"))
        splitter.addWidget(node_coords_container)

        # ---- Support
        support_layout, support_container = create_layout_and_container()

        self.support_combo_box = QtWidgets.QComboBox()
        self.support_combo_box.addItems([
            "NONE", "ROLLER_X", "ROLLER_Y", "PINNED", "FIXED"
        ])
        self.support_combo_box.currentTextChanged.connect(lambda: self._update_selected_node_support(
            self.support_combo_box.currentText()
        ))

        support_layout.addWidget(QtWidgets.QLabel("Support: "))
        support_layout.addWidget(self.support_combo_box)

        splitter.addWidget(support_container)

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
        self.x_coordinate.setPlainText(str(x))
        self.y_coordinate.setPlainText(str(y))
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
        self.create_support_action = QtWidgets.QAction("&Support", self)
        self.create_charge_action = QtWidgets.QAction("&Charge", self)
        # Add shortcuts
        self.enable_node_mode_action.setShortcut("N")
        self.enable_bar_mode_action.setShortcut("B")
        # Add help tips
        _add_tip(self.enable_node_mode_action, "Create a new node")
        _add_tip(self.enable_bar_mode_action, "Create a new bar")
        _add_tip(self.create_support_action, "Create a new support")
        _add_tip(self.create_charge_action, "Create a new charge")

        # ========== HELP ACTIONS ==========
        # Create actions
        self.help_content_action = QtWidgets.QAction("&Help content", self)
        self.about_action = QtWidgets.QAction("&About", self)
        # Add help tips


class Bar(QtWidgets.QGraphicsLineItem):
    def __init__(self, main_window, node_origin, node_end):
        self.main_window = main_window

        # Origin point
        self.node_origin = node_origin
        self.x1_scene = node_origin.x_scene + NODE_RADIUS / 2
        self.y1_scene = node_origin.y_scene + NODE_RADIUS / 2
        self.node_origin.signals.position_changed.connect(lambda: self.update_point_position(
            "origin",
            self.node_origin.x_scene,
            self.node_origin.y_scene
        ))
        # End Point
        self.node_end = node_end
        self.x2_scene = node_end.x_scene + NODE_RADIUS / 2
        self.y2_scene = node_end.y_scene + NODE_RADIUS / 2
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

        # Bar logic
        bar_name = f"B_{self.x1_scene}_{self.y1_scene}_{self.x2_scene}_{self.y2_scene}"
        origin_node_logic = node_origin.node_logic
        end_node_logic = node_end.node_logic
        material = self.main_window.get_currently_selected_material()
        profile = self.main_window.get_currently_selected_profile()
        self.bar_logic = st.Bar(bar_name, origin_node_logic, end_node_logic,
                                material, profile)

    def change_bar_color(self, color="normal"):
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

        new_pen = QtGui.QPen(new_color, self.drawn_thickness)
        # Actually change the color
        self.setPen(new_pen)
        # Store logically the currently used color
        self.color = new_color

    def update_point_position(self, point_reference, new_x_scene, new_y_scene):
        """
        Updates the position of one of the points of the line
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


class NodeSignals(QtWidgets.QGraphicsObject):
    """
    The class from which inherits Node cannot emit signals. This class is aimed to hold the needed custom signals
    of the Node class and emit them when necessary
    """
    position_changed = QtCore.pyqtSignal()

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

        self.setPos(x_scene, y_scene)

        self.main_window = main_window
        # x and y coordinates in scece reference system
        self.x_scene, self.y_scene = x_scene, y_scene
        # x and y coordinates in centered reference system
        self.x_centered, self.y_centered = self.main_window.centered_coordinates(x_scene, y_scene)
        # Radius of the node
        self.radius = radius

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
        self.update_support(
            self.main_window.support_combo_box.currentText()
        )

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
            draw_pos_x = int(new_x_scene - self.radius / 2)
            draw_pos_y = int(new_y_scene - self.radius / 2)

            try:
                # Pack into a point the draw coordinates
                draw_pos = QtCore.QPoint(draw_pos_x, draw_pos_y)
            except OverflowError:
                # If the coordinates are to big, then cancel the operation
                return

            # Change node position
            self.setPos(draw_pos)
            # This signal communicates with Bar to change its position
            self.signals.position_changed.emit()

    def update_support(self, support_name):
        if support_name == "ROLLER_X":
            support = st.Support.ROLLER_X
        elif support_name == "ROLLER_Y":
            support = st.Support.ROLLER_Y
        elif support_name == "PINNED":
            support = st.Support.PINNED
        elif support_name == "FIXED":
            support = st.Support.FIXED
        else:
            support = st.Support.NONE

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
            x = active_structure_element.node_logic.x()
            y = active_structure_element.node_logic.y()
            z = active_structure_element.node_logic.z()
            # Show coordinates in the textboxes
            self.main_window.update_coordinates(x, y, z)
            # Show support
            support_name = active_structure_element.node_logic.support
            support_name = str(support_name).split(".")
            self.main_window.support_combo_box.setCurrentText(
                support_name[1]
            )
        # BAR item
        elif type(active_structure_element) is Bar:
            # Change bar color
            active_structure_element.change_bar_color("selected")
            material_name = active_structure_element.bar_logic.get_material().name
            profile_name = " ".join([active_structure_element.bar_logic.get_profile().name,
                                     str(active_structure_element.bar_logic.get_profile().name_number)])

            self.main_window.material_combo_box.setCurrentText(material_name)
            self.main_window.profile_combo_box.setCurrentText(profile_name)
            self.main_window.update_coordinates("---", "---", "---")
            self.main_window.support_combo_box.setCurrentText("NONE")
        # No item
        elif active_structure_element is None:
            self.main_window.update_coordinates("---", "---", "---")
            self.main_window.support_combo_box.setCurrentText("NONE")

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


class PlainTextBox(QtWidgets.QPlainTextEdit):
    """
    Extends QPlainTextEdit in order to be able to specify a sizeHint lesser than the default one
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 10)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 30)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    # window.showMaximized()
    sys.exit(app.exec_())
