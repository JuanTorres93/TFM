import math
import sys
from PyQt5 import QtCore, QtWidgets

from src.modules import databaseutils as db

# Create the database if it does not exist
db.regenerate_initial_database()


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # TODO Borrar cuando se haya puesto la estrucutra de layouts
        self.main_window_width = 1500
        self.main_windows_height = 816

        # Central Widget
        # This widget, in addition, is used to display the drawing scene where the user inputs the structure
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
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 2000, 1000)
        self.scene.addText("Hello world")
        self.scene.addRect(1000, 500, 200, 200)

        self.central_widget.setScene(self.scene)
        self.central_widget.setParent(self)

        # TODO el zoom se implementa con el método scale
        # self.graphics_view.scale(4, 4)
        # TODO interacción con teclado y ratón usando QGraphicsSceneEvent, no sé si lo implementa ya por defecto
        self.central_widget.show()

        # Main window definition
        self.setObjectName("MainWindow")
        self.setWindowTitle("TFM")
        self.resize(self.main_window_width, self.main_windows_height)
        self.setCentralWidget(self.central_widget)

    def new_file(self):
        # TODO escribir funcion
        self.scene.addRect(1000, 500, 100, 100)

    def draw_node(self):
        # TODO escribir funcion
        self.scene.addRect(1500, 500, 100, 100)

    def _create_button(self, button_name, geometry, button_text, parent=None):
        if parent is None:
            parent = self.centralwidget

        button = QtWidgets.QPushButton(parent)
        button.setGeometry(geometry)
        button.setObjectName(button_name)
        button.setText(button_text)

        return button

    def _create_combo_box(self, name, geometry, items, parent=None):
        if parent is None:
            parent = self.centralwidget

        combo_box = QtWidgets.QComboBox(parent)
        combo_box.setGeometry(geometry)
        combo_box.setObjectName(name)
        combo_box.addItems(items)

        return combo_box

    def _create_menu_bar(self):
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
        edit_menu.addAction(self.create_node_action)
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

        test_borrar_menu = help_menu.addMenu("Test Borrar")
        test_borrar_menu.addAction("Test")
        test_borrar_menu.addAction("Borrar")

        menu_bar.addMenu(help_menu)

    def _create_toolbars(self):
        # ========== STRUCTURE TOOLBAR ==========
        structure_toolbar = QtWidgets.QToolBar("Structure", self)

        structure_toolbar.addAction(self.create_node_action)
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

        properties_toolbar.addSeparator()
        # Label profile
        label_profile = QtWidgets.QLabel("Perfil: ")

        properties_toolbar.addWidget(label_profile)

        # ComboBox profile
        combo_items = ["IPE 300", "IPE 200"]
        profile_combo_box = QtWidgets.QComboBox()
        profile_combo_box.addItems(combo_items)
        profile_combo_box.setFocusPolicy(QtCore.Qt.NoFocus)

        properties_toolbar.addWidget(profile_combo_box)

        self.addToolBar(QtCore.Qt.RightToolBarArea, properties_toolbar)

    def _create_status_bar(self):
        self.status_bar = self.statusBar()
        # Temporary message
        self.status_bar.showMessage("Ready", 3000)

        # Permanent message
        f_string_example = f"Permanent {math.pi} message"
        permanent_message = QtWidgets.QLabel(f_string_example)
        self.status_bar.addPermanentWidget(permanent_message)

    def _create_context_menu(self):
        # Creates right-click menus
        # Set contextMenuPolicy
        self.central_widget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        # Widget to act as a separator in context menus, since the method .addSeparator can't be used
        separator = QtWidgets.QAction(self)
        separator.setSeparator(True)

        # Populate widget with action
        self.central_widget.addAction(self.create_node_action)
        # TODO Borrar, este separador, solo esta para motivos de documentacion
        self.central_widget.addAction(separator)
        self.central_widget.addAction(self.create_bar_action)

    def _connect_actions(self):
        self.new_file_action.triggered.connect(self.new_file)
        self.exit_action.triggered.connect(self.close)

        self.create_node_action.triggered.connect(self.draw_node)

    def _create_actions(self):
        def _add_tip(item, tip):
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
        self.create_node_action = QtWidgets.QAction("New &Node", self)
        self.create_bar_action = QtWidgets.QAction("New &Bar", self)
        self.create_support_action = QtWidgets.QAction("&Support", self)
        self.create_charge_action = QtWidgets.QAction("&Charge", self)
        # Add shortcuts
        self.create_node_action.setShortcut("N")
        self.create_bar_action.setShortcut("B")
        # Add help tips
        _add_tip(self.create_node_action, "Create a new node")
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
    sys.exit(app.exec_())
