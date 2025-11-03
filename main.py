from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QApplication
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream, Qt, QUrl, pyqtSignal
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QSizePolicy
)
import time

import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLabel, QWidget, QFrame, QScrollArea)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import io
import folium

from design import Ui_window
from utils import *


class MainWindow(QMainWindow):
    estado_conexion_changed = pyqtSignal(bool, float, float)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.conectado = False
        self.coordenadas_iniciales = (20.432939, -99.598862)
        self.ui = Ui_window()
        self.ui.setupUi(self)

        # Crear páginas
        self.tablero_page = page_Tablero()
        self.diagnosticar_page = page_diagnosticar()
        self.estadisticos_page = page_Estadisticas()

        page0 = self.ui.stackedWidget.widget(0)
        if not page0.layout(): page0.setLayout(QVBoxLayout())
        page0.layout().addWidget(self.tablero_page)

        page1 = self.ui.stackedWidget.widget(1)
        if not page1.layout(): page1.setLayout(QVBoxLayout())
        page1.layout().addWidget(self.diagnosticar_page)

        page2 = self.ui.stackedWidget.widget(2)
        if not page2.layout(): page2.setLayout(QVBoxLayout())
        page2.layout().addWidget(self.estadisticos_page)

        self.estado_conexion_changed.connect(self.diagnosticar_page.set_estado_conexion)

        # Conectar botones de sidebarr
        self.ui.btn_tablero.toggled.connect(self.on_btn_tablero_toggled)
        self.ui.btn_tablero_2.toggled.connect(self.on_btn_tablero_toggled)
        self.ui.btn_diagnosticar.toggled.connect(self.on_btn_diagnosticar_toggled)
        self.ui.btn_diagnosticar_2.toggled.connect(self.on_btn_diagnosticar_toggled)
        self.ui.btn_estadisticos.toggled.connect(self.on_btn_estadisticos_toggled)
        self.ui.btn_estadisticos_2.toggled.connect(self.on_btn_estadisticos_toggled)
        self.ui.btn_conectar.toggled.connect(self.on_btn_conectar_toggled)
        self.ui.btn_conectar_2.toggled.connect(self.on_btn_conectar_toggled)

        # Sidebar
        self.ui.short_menu_bar.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.btn_tablero_2.setChecked(True)

        self.setWindowTitle("Sistema Detector de Enfermedades Foliares")

    # Funciones para cambiar de páginas
    def on_btn_tablero_toggled(self):
        if self.ui.btn_tablero.isChecked() or self.ui.btn_tablero_2.isChecked():
            self.ui.stackedWidget.setCurrentIndex(0)
            self.ui.label_11.setText("Tablero de diagnosticos")

    def on_btn_diagnosticar_toggled(self):
        if self.ui.btn_diagnosticar.isChecked() or self.ui.btn_diagnosticar_2.isChecked():
            self.ui.stackedWidget.setCurrentIndex(1)
            self.ui.label_11.setText("Nuevo diagnostico")

    def on_btn_estadisticos_toggled(self):
        if self.ui.btn_estadisticos.isChecked() or self.ui.btn_estadisticos_2.isChecked():
            self.ui.stackedWidget.setCurrentIndex(2)
            self.ui.label_11.setText("Resultados estadisticos")

    # Cuando le picas en conectar
    def on_btn_conectar_toggled(self):
        is_checked = self.ui.btn_conectar.isChecked() or self.ui.btn_conectar_2.isChecked()
        if is_checked:
            ##Aqui se hace la conexion
            conexion = True
            ##Regresa las coordenadas
            lat, long = (19.401036, -99.134668)
            #######
            if conexion:
                self.ui.btn_conectar.setStyleSheet("background-color: rgb(49, 201, 80); color: white")
                self.ui.btn_conectar_2.setStyleSheet("background-color: rgb(49, 201, 80); color: white")
                self.ui.btn_conectar_2.setText("Conectado")
                self.conectado = True
            else:
                self.ui.btn_conectar.setStyleSheet("background-color: rgb(49, 201, 80); color: white")
                self.ui.btn_conectar_2.setStyleSheet("background-color: rgb(49, 201, 80); color: white")
                self.ui.btn_conectar_2.setText("Intentar nuevamente")
                self.conectado = False


        self.estado_conexion_changed.emit(self.conectado, lat, long)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Cargando hoja de estilos e íconos
    with open("./style.qss", "r") as style_file:
        style_str = style_file.read()
    app.setStyleSheet(style_str)

    # ejecutando ventana
    window = MainWindow()
    window.show()

    # Cierre
    sys.exit(app.exec())



