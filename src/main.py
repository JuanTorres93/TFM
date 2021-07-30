import math

from PyQt5 import QtCore, QtWidgets

from src.modules import databaseutils as db

# Create the database if it does not exist
db.regenerate_initial_database()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.main_window_width = 1500
        self.main_windows_height = 816

        # Main window definition
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.main_window_width, self.main_windows_height)
        MainWindow.setCentralWidget(self.centralwidget)

        # TODO implementar los distintos elementos mediante layouts
        # Left frame
        self.frame_left = self.create_left_frame_main_gui()
        self.frame_left.raise_()

        # Right frame
        self.frame_right = self.create_right_frame_main_gui()

        # Canvas
        self.scene = QtWidgets.QGraphicsScene(MainWindow)
        self.scene.setSceneRect(0, 0, 2000, 1000)
        self.scene.addText("Hello world")
        self.scene.addRect(1000, 500, 200, 200)

        self.graphics_view = QtWidgets.QGraphicsView()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setParent(MainWindow)
        self.graphics_view.setGeometry(QtCore.QRect(self.frame_left.width(), 0,
                                       self.main_window_width - self.frame_left.width() - self.frame_right.width(),
                                                    self.main_windows_height))
        # self.graphics_view.lower()

        # TODO el zoom se implementa con el método scale
        # self.graphics_view.scale(4, 4)
        # TODO interacción con teclado y ratón usando QGraphicsSceneEvent, no sé si lo implementa ya por defecto
        self.graphics_view.show()


        # # Menu bar
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, 1081, 22))
        # self.menubar.setObjectName("menubar")
        #
        # # File menu
        # self.menuFile = QtWidgets.QMenu(self.menubar)
        # self.menuFile.setObjectName("menuFile")
        # MainWindow.setMenuBar(self.menubar)
        #
        # # Status bar
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)
        # self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def create_button(self, button_name, geometry, button_text, parent=None):
        if parent is None:
            parent = self.centralwidget

        button = QtWidgets.QPushButton(parent)
        button.setGeometry(geometry)
        button.setObjectName(button_name)
        button.setText(button_text)

        return button

    def create_combo_box(self, name, geometry, items, parent=None):
        if parent is None:
            parent = self.centralwidget

        combo_box = QtWidgets.QComboBox(parent)
        combo_box.setGeometry(geometry)
        combo_box.setObjectName(name)
        combo_box.addItems(items)

        return combo_box

    def create_left_frame_main_gui(self):
        frame = QtWidgets.QFrame(self.centralwidget)
        frame_width = 120
        frame.setGeometry(QtCore.QRect(0, 140, frame_width, 321))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setObjectName("frame_left")

        width = 80
        height = 51
        y = 20
        # Center horizontally
        x = int(frame_width - frame_width / 2 - width / 2)

        y_offset = height + 20

        # Button Charges
        button_position = 0
        button_y = int(y + y_offset * button_position)
        button_charges = self.create_button(button_name="button_charges", geometry=QtCore.QRect(x, button_y,
                                                                                                width, height),
                                            button_text="Cargas", parent=frame)

        # Button bar
        button_position = 1
        button_y = int(y + y_offset * button_position)
        button_bar = self.create_button(button_name="button_bar", geometry=QtCore.QRect(x, button_y,
                                                                                        width, height),
                                        button_text="Barras", parent=frame)

        # Button support
        button_position = 2
        button_y = int(y + y_offset * button_position)
        button_support = self.create_button(button_name="button_support", geometry=QtCore.QRect(x, button_y,
                                                                                                width, height),
                                            button_text="Apoyos", parent=frame)

        return frame

    def create_right_frame_main_gui(self):
        frame = QtWidgets.QFrame(self.centralwidget)
        frame_width = 120
        frame.setGeometry(QtCore.QRect(self.main_window_width - frame_width, 140, frame_width, 321))
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setObjectName("frame_right")

        width = 91
        height = 25
        y = 20
        # Center horizontally
        x = int(frame_width - frame_width / 2 - width / 2)

        y_offset = height + 10

        # Label material
        item_position = 0
        item_y = int(y + y_offset * item_position)
        label_material = QtWidgets.QLabel(frame)
        label_material.setGeometry(QtCore.QRect(x, item_y, width, height))
        label_material.setText("Material:")
        label_material.setObjectName("label_material")

        # Material comboBox
        item_position = 1
        item_y = int(y + y_offset * item_position)
        combo_items = ["Acero 1", "Acero 2"]
        material_combo_box = self.create_combo_box(name="material_combobox", geometry=QtCore.QRect(x, item_y,
                                                                                                   width, height),
                                                   items=combo_items, parent=frame)

        # Label section
        item_position = 2
        item_y = int(y + y_offset * item_position)
        label_seccion = QtWidgets.QLabel(frame)
        label_seccion.setGeometry(QtCore.QRect(x, item_y, width, height),)
        label_seccion.setText("Sección:")
        label_seccion.setObjectName("label_seccion")

        # Section comboBox
        item_position = 3
        item_y = int(y + y_offset * item_position)
        combo_items = ["IPE 300", "IPE 200"]
        section_combo_box = self.create_combo_box(name="section_combobox", geometry=QtCore.QRect(x, item_y,
                                                                                                   width, height),
                                                   items=combo_items, parent=frame)

        return frame

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # self.menuFile.setTitle(_translate("MainWindow", "File"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
