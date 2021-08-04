import enum
import math
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from src.modules import databaseutils as db

# Create the database if it does not exist
db.regenerate_initial_database()


@enum.unique
class ApplicationMode(enum.Enum):
    """
    Enumeration for the different types of application modes
    """
    NODE_DRAW_MODE = 1
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
        super().__init__(x_scene, y_scene, radius, radius)

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

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        if application_mode == ApplicationMode.NODE_DRAW_MODE:
            node_radius = 10
            self.main_window.draw_node(node_radius)
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
                self.main_window.selection_properties.append(f"Scene -> x: {item.x_scene}, y: {item.y_scene}")
                self.main_window.selection_properties.append(f"Centered -> x: {item.x_centered}, y: {item.y_centered}")
                self.main_window.selection_properties.append("====================")


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
        self._create_toolbars()

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

    def _set_current_mode(self, mode, message):
        """
        Establish the mode in which the user utilizes the application in a given moment
        :param mode: Mode to which the application is going to change
        :param message: Permanent message to show in the status bar
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

        if application_mode != ApplicationMode.NODE_DRAW_MODE:
            self._set_current_mode(ApplicationMode.NODE_DRAW_MODE, "NODE MODE")
        else:
            self._set_current_mode(ApplicationMode.NORMAL_MODE, "NORMAL MODE")

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

    def _create_toolbars(self):
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

        # ========== PROPERTIES TOOLBAR ==========
        properties_toolbar = QtWidgets.QToolBar("Properties", self)

        # Label material
        label_material = QtWidgets.QLabel("Material: ")

        properties_toolbar.addWidget(label_material)

        # Material comboBox
        combo_items = ["S235", "S275"]
        material_combo_box = QtWidgets.QComboBox()
        material_combo_box.addItems(combo_items)
        material_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)

        properties_toolbar.addWidget(material_combo_box)

        # Label profile
        label_profile = QtWidgets.QLabel("Profile: ")

        properties_toolbar.addWidget(label_profile)

        # ComboBox profile
        combo_items = ["IPE 300", "IPE 200"]
        profile_combo_box = QtWidgets.QComboBox()
        profile_combo_box.addItems(combo_items)
        profile_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)

        properties_toolbar.addWidget(profile_combo_box)

        properties_toolbar.addSeparator()
        # Selection properties
        self.selection_properties = QtWidgets.QTextEdit()
        self.selection_properties.setReadOnly(True)
        properties_toolbar.addWidget(self.selection_properties)

        self.addToolBar(QtCore.Qt.RightToolBarArea, properties_toolbar)

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
