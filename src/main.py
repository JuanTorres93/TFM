from PyQt5 import QtCore, QtGui, QtWidgets
from src.modules import databaseutils as db

# Create the database if it does not exist
db.regenerate_initial_database()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Main window definition
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1081, 816)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Left Frame
        self.frame_left = QtWidgets.QFrame(self.centralwidget)
        self.frame_left.setGeometry(QtCore.QRect(10, 140, 141, 321))
        self.frame_left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left.setObjectName("frame_left")

        # Button Charges
        self.button_charges = QtWidgets.QPushButton(self.frame_left)
        self.button_charges.setGeometry(QtCore.QRect(10, 210, 80, 51))
        self.button_charges.setObjectName("button_charges")

        # Button bar
        self.button_bar = QtWidgets.QPushButton(self.frame_left)
        self.button_bar.setGeometry(QtCore.QRect(10, 50, 80, 51))
        self.button_bar.setObjectName("button_bar")

        # Button support
        self.button_support = QtWidgets.QPushButton(self.frame_left)
        self.button_support.setGeometry(QtCore.QRect(10, 130, 80, 51))
        self.button_support.setObjectName("button_support")

        # Frame right
        self.frame_right = QtWidgets.QFrame(self.centralwidget)
        self.frame_right.setGeometry(QtCore.QRect(900, 140, 161, 321))
        self.frame_right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_right.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right.setObjectName("frame_right")

        # Section type
        self.comboBox_seccion = QtWidgets.QComboBox(self.frame_right)
        self.comboBox_seccion.setGeometry(QtCore.QRect(30, 160, 91, 25))
        self.comboBox_seccion.setObjectName("comboBox_seccion")
        self.comboBox_seccion.addItem("")
        self.comboBox_seccion.addItem("")

        # Label material
        self.label_material = QtWidgets.QLabel(self.frame_right)
        self.label_material.setGeometry(QtCore.QRect(30, 50, 61, 17))
        self.label_material.setObjectName("label_material")

        # Material selection
        self.comboBox_material = QtWidgets.QComboBox(self.frame_right)
        self.comboBox_material.setGeometry(QtCore.QRect(30, 80, 91, 25))
        self.comboBox_material.setObjectName("comboBox_material")
        self.comboBox_material.addItem("")
        self.comboBox_material.addItem("")

        # Label section
        self.label_seccion = QtWidgets.QLabel(self.frame_right)
        self.label_seccion.setGeometry(QtCore.QRect(30, 130, 61, 17))
        self.label_seccion.setObjectName("label_seccion")
        MainWindow.setCentralWidget(self.centralwidget)

        # Menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1081, 22))
        self.menubar.setObjectName("menubar")

        # File menu
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_charges.setText(_translate("MainWindow", "Cargas"))
        self.button_bar.setText(_translate("MainWindow", "Barras"))
        self.button_bar.setShortcut(_translate("MainWindow", "B"))
        self.button_support.setText(_translate("MainWindow", "Apoyos"))
        self.comboBox_seccion.setItemText(0, _translate("MainWindow", "IPE"))
        self.comboBox_seccion.setItemText(1, _translate("MainWindow", "HBE"))
        self.label_material.setText(_translate("MainWindow", "Material:"))
        self.comboBox_material.setItemText(0, _translate("MainWindow", "Acero 1"))
        self.comboBox_material.setItemText(1, _translate("MainWindow", "Acero 2"))
        self.label_seccion.setText(_translate("MainWindow", "Seccion:"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
