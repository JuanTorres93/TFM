import enum
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from src.modules import databaseutils as db
from src.modules import structures as st

# Create the database if it does not exist
db.regenerate_initial_database()
meter_to_px = 50
px_to_meter = 1 / meter_to_px

# Currently selected nod or bar
active_structure_element = None


@enum.unique
class ApplicationMode(enum.Enum):
    """
    Enumeration for the different types of application modes
    """
    NODE_MODE = 1
    NORMAL_MODE = 2


# Default mode for application is normal mode
application_mode = ApplicationMode.NORMAL_MODE


class Node(QtWidgets.QGraphicsEllipseItem):
    """
    Class that holds all structure node information, as well as its graphic behavior
    """

    def __init__(self, main_window, x_scene, y_scene, radius, color=QtGui.QColor(0, 0, 0)):
        """

        :param main_window: main_window of the application
        :param x_scene: x position in scene coordinates
        :param y_scene: y position in scene coordinates
        :param radius: radius of the graphical representation
        :param color: color in which the node is drawn
        """
        # 0, 0 are x and y coordinates in ITEM COORDINATES
        super().__init__(0, 0, radius, radius)

        self.setPos(x_scene, y_scene)

        self.main_window = main_window
        self.x_scene = x_scene
        self.y_scene = y_scene
        # x and y coordinates in centered reference system
        self.x_centered, self.y_centered = self.main_window.centered_coordinates(x_scene, y_scene)
        self.radius = radius

        # Normal color of the node
        self._normal_color = color
        # Color of the node when hovering mouse over it
        self._hover_color = QtGui.QColor(235, 204, 55)
        self.setBrush(color)

        # Needed for hover events to take place
        self.setAcceptHoverEvents(True)

        # Node logic
        x_meter = self.x_centered * px_to_meter
        y_meter = self.y_centered * px_to_meter

        node_name = str(x_scene) + "_" + str(y_scene)
        self.node_logic = st.Node(node_name, (x_meter, y_meter, 0))

        # BORRAR
        print(self.boundingRect())
        print(self.sceneBoundingRect())
        self.scene

    def update_position(self, new_x_centered_in_meters=None, new_y_centered_in_meters=None):
        if new_x_centered_in_meters is None:
            new_x_centered_in_meters = self.node_logic.x()
            new_x_centered = self.x_centered
        else:
            new_x_centered = int(new_x_centered_in_meters * meter_to_px)
            self.x_centered = new_x_centered

        if new_y_centered_in_meters is None:
            new_y_centered_in_meters = self.node_logic.y()
            new_y_centered = self.y_centered
        else:
            new_y_centered = int(new_y_centered_in_meters * meter_to_px)
            self.y_centered = new_y_centered

        new_x_scene, new_y_scene = self.main_window.scene_coordinates(new_x_centered, new_y_centered)

        new_x_scene = int(new_x_scene)
        new_y_scene = int(new_y_scene)

        self.x_scene, self.y_scene = new_x_scene, new_y_scene
        self.node_logic.set_position((new_x_centered_in_meters, new_y_centered_in_meters, 0))

        new_pos = QtCore.QPoint(new_x_scene - self.radius / 2, new_y_scene - self.radius / 2)
        self.setPos(new_pos)

    def sceneBoundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.x_scene, self.y_scene, self.radius, self.radius)

    def hoverEnterEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse enters the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE:
            self.setBrush(self._hover_color)

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        """
        Defines node behavior when the mouse exits the node
        :param event:
        :return:
        """
        if application_mode == ApplicationMode.NORMAL_MODE:
            self.setBrush(self._normal_color)

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

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        global application_mode
        global active_structure_element

        # Go back to normal mode
        if event.key() == QtCore.Qt.Key_Escape:
            self.main_window.set_current_mode(ApplicationMode.NORMAL_MODE)
        # Delete active element
        elif event.key() == QtCore.Qt.Key_Delete and application_mode == ApplicationMode.NORMAL_MODE:
            if type(active_structure_element) is Node:
                self.main_window.delete_node(active_structure_element)

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        # NODE MODE functionality
        if application_mode == ApplicationMode.NODE_MODE:
            node_radius = 10
            self.main_window.draw_node(node_radius)
        # NORMAL MODE functionality
        elif application_mode == ApplicationMode.NORMAL_MODE:
            # Get position where the release has happened
            position = event.scenePos()
            # Transform matrix is needed for itemAt methos.
            # It is used the identity matrix in order not to change anything
            transform = QtGui.QTransform(1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1)

            # Get the item at event position
            item = self.itemAt(position, transform)

            # Process the found item, if any
            if item is not None:
                global active_structure_element
                # Select current element as active
                active_structure_element = item

                # NODE item
                if type(item) is Node:
                    x = item.node_logic.x()
                    y = item.node_logic.y()
                    z = item.node_logic.z()
                    self.main_window.update_coordinates(x, y, z)

            else:
                # If no item is found, then deselect the current active elemente, if any
                if active_structure_element is not None:
                    active_structure_element = None


class PlainTextBox(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 10)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, 30)


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

        # Change the mode internally
        global application_mode
        if application_mode != mode:
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
        # TODO rewrite docstring
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
        global application_mode

        if application_mode != ApplicationMode.NODE_MODE:
            self.set_current_mode(ApplicationMode.NODE_MODE)
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

        # Add node to scene
        self.scene.addItem(node)

        global active_structure_element
        active_structure_element = node

    def delete_node(self, node):
        self.scene.removeItem(node)
        global active_structure_element
        active_structure_element = None

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
        x_axis.setZValue(-1)

        # Y Axis
        point_y1 = [self.scene.sceneRect().width() / 2, 0]
        point_y2 = [self.scene.sceneRect().width() / 2, self.scene.sceneRect().height()]
        color = QtGui.QColor(0, 200, 0)
        # Draw axis
        y_axis = self.scene.addLine(point_y1[0], point_y1[1], point_y2[0], point_y2[1], pen=color)
        # Draw it at the bottom in order not to superpose user drawings
        y_axis.setZValue(-1)

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
        edit_menu.addAction(self.create_bar_action)
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
        try:
            new_pos = float(text)
        except ValueError:
            return

        global active_structure_element
        if type(active_structure_element) is Node:
            if axis == "x":
                active_structure_element.update_position(new_pos, None)
            elif axis == "y":
                active_structure_element.update_position(None, new_pos)
            # TODO implement Z if 3D structures

    def _create_toolbars_and_docks(self):
        """
        Creates the toolbars in the main window
        """
        # ========== STRUCTURE TOOLBAR ==========
        structure_toolbar = QtWidgets.QToolBar("Structure", self)

        structure_toolbar.addAction(self.enable_node_mode_action)
        structure_toolbar.addAction(self.create_bar_action)
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
        combo_items = ["S235", "S275"]
        material_combo_box = QtWidgets.QComboBox()
        material_combo_box.addItems(combo_items)
        material_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)

        material_layout.addWidget(material_combo_box)

        splitter.addWidget(mat_container)

        # Profile
        profile_layout, profile_container = create_layout_and_container()
        # -- Label profile
        label_profile = QtWidgets.QLabel("Profile")
        profile_layout.addWidget(label_profile)

        # -- ComboBox profile
        combo_items = ["IPE 300", "IPE 200"]
        profile_combo_box = QtWidgets.QComboBox()
        profile_combo_box.addItems(combo_items)
        profile_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)
        profile_layout.addWidget(profile_combo_box)

        splitter.addWidget(profile_container)

        # Properties
        # -- Node properties
        # ---- Coordinates
        node_coords_layout, node_coords_container = create_layout_and_container()

        def create_coordinate(self, label_text, node_coords_layout):
            label = QtWidgets.QLabel(label_text)
            text_item = PlainTextBox()
            text_item.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

            associated_axis = None
            if label_text.startswith("x"):
                associated_axis = "x"
            elif label_text.startswith("y"):
                associated_axis = "y"
            else:
                raise ValueError(
                    f"Error: label_text must begin with the letter of an axis. Current value is {label_text}")

            text_item.textChanged.connect(lambda:
                                          self._update_selected_node_position(text_item.toPlainText(),
                                                                              associated_axis))

            node_coords_layout.addWidget(label, 1)
            node_coords_layout.addWidget(text_item, 4)

            return text_item

        # -------- x coordinate
        self.x_coordinate = create_coordinate(self, "x", node_coords_layout)
        # -------- y coordinate
        self.y_coordinate = create_coordinate(self, "y", node_coords_layout)
        # -------- z coordinate
        # self.z_coordinate = create_coordinate("z", node_coords_layout)

        self.update_coordinates(0, 0, 0)

        splitter.addWidget(QtWidgets.QLabel("Coordinates:"))
        splitter.addWidget(node_coords_container)

        # TODO borrar selection_properties
        self.selection_properties = QtWidgets.QTextEdit()
        self.selection_properties.setReadOnly(True)
        splitter.addWidget(self.selection_properties)

        # self.addToolBar(QtCore.Qt.RightToolBarArea, properties_toolbar)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, properties_dock)

    def update_coordinates(self, x, y, z):
        self.x_coordinate.setPlainText(str(x))
        self.y_coordinate.setPlainText(str(y))
        # self.z_coordinate.setPlainText(str(z))

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
        self.central_widget.addAction(self.create_bar_action)

    def _connect_actions(self):
        """
        Connects the actions to functions
        """
        self.new_file_action.triggered.connect(self.new_file)
        self.exit_action.triggered.connect(self.close)

        self.enable_node_mode_action.triggered.connect(self.activate_draw_node_mode)

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
        self.enable_node_mode_action = QtWidgets.QAction("New &Node", self)
        self.create_bar_action = QtWidgets.QAction("New &Bar", self)
        self.create_support_action = QtWidgets.QAction("&Support", self)
        self.create_charge_action = QtWidgets.QAction("&Charge", self)
        # Add shortcuts
        self.enable_node_mode_action.setShortcut("N")
        self.create_bar_action.setShortcut("B")
        # Add help tips
        _add_tip(self.enable_node_mode_action, "Create a new node")
        _add_tip(self.create_bar_action, "Create a new bar")
        _add_tip(self.create_support_action, "Create a new support")
        _add_tip(self.create_charge_action, "Create a new charge")

        # ========== HELP ACTIONS ==========
        # Create actions
        self.help_content_action = QtWidgets.QAction("&Help content", self)
        self.about_action = QtWidgets.QAction("&About", self)
        # Add help tips


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    # window.showMaximized()
    sys.exit(app.exec_())
