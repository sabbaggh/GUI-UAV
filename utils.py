import json
# --- AÑADIDO ---
import os
import posixpath
import paramiko
# --- FIN AÑADIDO ---
from PyQt5 import QtCore, QtGui
# --- MODIFICADO ---
# Asegúrate de que QThread, QObject, pyqtSignal, QIODevice estén
from PyQt5.QtCore import Qt, pyqtSlot, QPointF, QRectF, QTimer, QFile, QTextStream, QUrl, QObject, pyqtSignal, QIODevice, QThread
# --- FIN MODIFICADO ---
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QPolygonF, QIcon, QBrush, QFont
import numpy as np
import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QVBoxLayout, QStackedWidget, QSizePolicy,
                             QGraphicsScene, QGraphicsEllipseItem, QHBoxLayout, QLabel, QWidget, QFrame, QListWidget,
                             QScrollArea, QMessageBox, QGraphicsView, QGraphicsPixmapItem, QUndoStack, QUndoCommand,
                             QFileDialog) # --- AÑADIDO QFileDialog (por si acaso)

from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import io
import datetime
import folium
# --- MODIFICADO ---
# La importación 'sklearn.externals' está obsoleta y puede fallar. 
# Si no la usas, elimínala. Si la usas, busca un reemplazo.
# from sklearn.externals.array_api_compat.torch import empty 
# --- FIN MODIFICADO ---

from geopy.distance import geodesic
from shapely.geometry import Polygon
from pyproj import Geod

from design import Ui_window
from predictor import *


# --- AÑADIDO --- ###
# Definición de las variables de conexión
# (Cámbialas por tus valores reales)
TAILSCALE_IP = "10.3.141.1"
USERNAME = "pera"
PASSWORD = "2314"
# --- FIN AÑADIDO --- ###


class Bridge(QObject):
    # (Tu clase Bridge no cambia)
    mapClicked = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def onMapClicked(self, lat, lng):
        self.mapClicked.emit(lat, lng)

# ### --- AÑADIDO: WORKER SSH --- ###
class SshWorker(QObject):
    """
    Worker que corre en un hilo separado para manejar la conexión SSH
    y la ejecución de scripts sin bloquear la GUI.
    """
    finished = pyqtSignal(str, str)  # (stdout, stderr)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)       # Para enviar actualizaciones de estado

    @pyqtSlot(str)
    def run_ssh_command(self, json_payload):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.progress.emit("Conectando al UAV...")
            client.connect(TAILSCALE_IP, username=USERNAME, password=PASSWORD, timeout=10)
            
            self.progress.emit("Ejecutando script de monitoreo...")
            # ¡IMPORTANTE! Asegúrate que la ruta a tu script sea correcta
            cmd = f"bash -lc 'source /home/pera/venv_drone/bin/activate && python3 -u /home/pera/xdd3.py'"
            stdin, stdout, stderr = client.exec_command(cmd, get_pty=False)

            stdin.write(json_payload)
            stdin.flush()
            stdin.channel.shutdown_write()

            out_lines = []
            for line in iter(stdout.readline, ""):
                line = line.strip()
                if not line: continue
                self.progress.emit(line) # Enviar cada línea de progreso
                out_lines.append(line)
            
            out = "\n".join(out_lines)
            err = stderr.read().decode('utf-8')
            
            self.progress.emit("Script finalizado.")
            self.finished.emit(out, err)

        except Exception as e:
            self.error.emit(f"Error de conexión o ejecución: {str(e)}")
        finally:
            if 'client' in locals() and client.get_transport() is not None:
                client.close()

# ### --- AÑADIDO: WORKER SFTP (DESCARGA) --- ###
class SftpWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    @pyqtSlot(str, str)
    def download_files(self, remote_dir, local_dir):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.progress.emit("Conectando para transferencia de archivos...")
            client.connect(TAILSCALE_IP, username=USERNAME, password=PASSWORD, timeout=10)
            
            sftp = client.open_sftp()
            self.progress.emit(f"Accediendo a: {remote_dir}")
            
            files = sftp.listdir(remote_dir)
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                self.error.emit("No se encontraron imágenes en el directorio.")
                sftp.close()
                client.close()
                return

            total_images = len(images)
            for i, fname in enumerate(images):
                self.progress.emit(f"Descargando {fname} ({i+1}/{total_images})...")
                remote_path = posixpath.join(remote_dir, fname)
                local_path = os.path.join(local_dir, fname)
                sftp.get(remote_path, local_path)

            sftp.close()
            client.close()
            self.progress.emit("¡Descarga completada!")
            self.finished.emit()

        except Exception as e:
            self.error.emit(f"Error de SFTP: {str(e)}")
        finally:
            if 'client' in locals() and client.get_transport() is not None:
                client.close()

# ### --- AÑADIDO: WORKER DE PREDICCIÓN (IA) --- ###
class PredictionWorker(QObject):
    """
    Worker que corre la predicción de IA en un hilo separado
    para no congelar la GUI.
    """
    # Señal de finalizado emite los 3 diccionarios
    finished = pyqtSignal(dict, dict, dict, dict, list) 
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    @pyqtSlot(str)
    def run_prediction(self, photos_path):
        try:
            self.progress.emit("Iniciando procesamiento de IA...")
            
            # Etiquetas para las clasificaciones
            CLASES_DEL_MODELO = [
                'common_rust', 'gray_leaf_spot', 'healthy', 
                'northern_leaf_blight', 'northern_leaf_spot'
            ]
            LABEL_PROBABILIDAD = [
                'Saludables', 'Leves rasgos', 'Rasgos considerables', 
                'Rasgos altos', 'Enfermas'
            ]

            # Hiperparámetros
            NUM_CLASES = 5
            PATH_MODELO = "./models/densenet_201_fold4.pth"
            
            # Carga y ejecución del modelo
            classifier = ImageClassifier(model_path=PATH_MODELO, num_classes=NUM_CLASES, class_names=CLASES_DEL_MODELO)
            
            # --- ¡IMPORTANTE! ---
            # El worker ahora usa la RUTA DE FOTOS DESCARGADA
            current_results = classifier.predict_folder(photos_path) 

            if not current_results:
                self.error.emit("El modelo no devolvió resultados.")
                return

            # Conteo de resultados
            class_counts = {class_name: 0 for class_name in CLASES_DEL_MODELO}
            class_file_lists = {class_name: [] for class_name in CLASES_DEL_MODELO}
            state_file_lists = {class_name: [] for class_name in LABEL_PROBABILIDAD}
            class_leaf_state = {class_name: 0 for class_name in LABEL_PROBABILIDAD}
            
            for filename, info in current_results.items():
                predicted_class = info['clase']
                if predicted_class in class_counts:
                    class_counts[predicted_class] += 1
                    class_file_lists[predicted_class].append(filename)
                    
                    try:
                        coord_raw = os.path.splitext(filename)[0].split(", ")
                        coord = [float(c) for c in coord_raw]
                    except Exception:
                        coord = [0.0, 0.0] # Coordenada placeholder si el nombre no es válido

                confianza = float(info['confianza healthy'])
                if 85.0 <= confianza <= 100.0:
                    class_leaf_state["Saludables"] += 1
                    state_file_lists["Saludables"].append(coord)
                elif 65.0 <= confianza < 85.0:
                    class_leaf_state["Leves rasgos"] += 1
                    state_file_lists["Leves rasgos"].append(coord)
                elif 30.0 <= confianza < 65.0:
                    class_leaf_state["Rasgos considerables"] += 1
                    state_file_lists["Rasgos considerables"].append(coord)
                elif 15.0 <= confianza < 30.0:
                    class_leaf_state["Rasgos altos"] += 1
                    state_file_lists["Rasgos altos"].append(coord)
                elif 0.0 <= confianza < 15.0:
                    class_leaf_state["Enfermas"] += 1
                    state_file_lists["Enfermas"].append(coord)

            nuevos_conteos = list(class_leaf_state.values())
            
            self.progress.emit("Procesamiento de IA finalizado.")
            # Emitir todos los resultados
            self.finished.emit(class_counts, class_file_lists, class_leaf_state, state_file_lists, nuevos_conteos)

        except Exception as e:
            self.error.emit(f"Error en la predicción: {str(e)}")


# --- PÁGINA TABLERO ---
class page_Tablero(QWidget):
    # (Tu código de page_Tablero initUI está bien)
    def __init__(self, parent=None):
        super(page_Tablero, self).__init__(parent)
        self.class_counts = {}
        self.list_results_per_class = {}
        self.list_leaf_state = {}

        self.dates = ['Esperando...']
        self.data = [[100.0]] 
        self.data_is_placeholder = True
        
        self.label_colores_prob_enfermedad = ['#006A35', '#34A853', '#FBBC04', '#F47C34', '#EA4335']
        self.label_probabilidad_enfermedad = [
            'Saludables', 'Leves rasgos', 'Rasgos considerables', 'Rasgos altos', 'Enfermas'
        ]
        self.initUI()

    def initUI(self):
        # ... (Tu código initUI de page_Tablero está bien) ...
        self.setWindowTitle("Tablero de diagnosticos")
        layout = QVBoxLayout(self)
        title_label = QLabel("Tablero estadístico")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        main_container = QHBoxLayout()
        bar_chart_container = QVBoxLayout()
        bar_title = QLabel("Distribución de salud")
        bar_title.setStyleSheet("font-weight: bold;")
        bar_chart_container.addWidget(bar_title)
        self.fig_bar = Figure(figsize=(8, 5), dpi=100)
        self.ax_bar = self.fig_bar.add_subplot(111)
        bottom = [0] * len(self.dates)
        values = [d[0] for d in self.data]
        self.ax_bar.bar(self.dates, values, bottom=bottom, color=['#E0E0E0'], label='Esperando datos...')
        self.ax_bar.tick_params(axis='x', rotation=20, labelsize=8)
        self.ax_bar.set_ylabel('% del Total de diagnósticos')
        self.ax_bar.set_xlabel('Fecha de diagnóstico')
        self.ax_bar.legend(loc='upper right', fontsize=8)
        self.ax_bar.set_ylim(0, 100)
        self.canvas_bar = FigureCanvas(self.fig_bar)
        bar_chart_container.addWidget(self.canvas_bar)
        legend_layout = QHBoxLayout()
        for i, (color, label) in enumerate(zip(self.label_colores_prob_enfermedad, self.label_probabilidad_enfermedad)):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"background-color: {color}; padding: 5px; border-radius: 5px; font-weight: bold; font-size: 14px")
            if i in [0, 1, 4]:
                lbl.setStyleSheet(f"color: #F5F5F5; background-color: {color}; padding: 5px; border-radius: 5px; font-weight: bold; font-size: 14px")
            legend_layout.addWidget(lbl)
        bar_chart_container.addLayout(legend_layout)
        main_container.addLayout(bar_chart_container)
        pie_chart_container = QVBoxLayout()
        pie_title = QLabel("Último análisis")
        pie_title.setStyleSheet("font-weight: bold;")
        pie_chart_container.addWidget(pie_title)
        pie_data = [1]
        pie_labels = ['Esperando datos...']
        self.fig_pie = Figure(figsize=(5, 5), dpi=100)
        self.ax_pie = self.fig_pie.add_subplot(111)
        wedges, texts = self.ax_pie.pie(pie_data, labels=pie_labels, colors=['#E0E0E0'], startangle=90)
        self.ax_pie.axis('equal')
        self.canvas_pie = FigureCanvas(self.fig_pie)
        pie_chart_container.addWidget(self.canvas_pie)
        analysis_text = """
        • 0% al 40% → Mejoró en un 10% el total de muestras saludables con respecto al último diagnóstico.
        • 41% al 65% → Se mantuvo el tamaño de muestras "con leves rasgos de enfermedad".
        • 66% al 100% → Disminuyó en un 5% el total de muestras "con rasgos considerables de enfermedad".
        """
        analysis_label = QLabel(analysis_text)
        analysis_label.setWordWrap(True)
        pie_chart_container.addWidget(analysis_label)
        main_container.addLayout(pie_chart_container)
        layout.addLayout(main_container)
        layout.addStretch()


    @pyqtSlot(dict, dict, dict)
    def set_result_plots(self, class_counts, list_results_per_class, list_leaf_state):
        # (Tu código set_result_plots está bien, solo añado un print de depuración)
        print(f"DEBUG (page_Tablero): Recibidos {list_leaf_state}")

        self.list_results_per_class = list_results_per_class
        self.list_leaf_state = list_leaf_state

        labels = list(self.list_leaf_state.keys())
        counts = list(self.list_leaf_state.values())
        colors_pie = self.label_colores_prob_enfermedad[:len(labels)]

        self.ax_pie.cla()
        labels_with_counts = [f'{l}\n{c}' for l, c in zip(labels, counts)]
        self.ax_pie.pie(counts, labels=labels_with_counts, colors=colors_pie, startangle=90, textprops={'fontsize': 9},
                        autopct='%1.1f%%')
        self.ax_pie.axis('equal')
        self.ax_pie.set_title("Resultados del Último Análisis")
        self.canvas_pie.draw()

        counts_raw = list(self.list_leaf_state.values())
        total_count = sum(counts_raw)

        if total_count == 0:
            counts_percent = [0.0] * len(counts_raw)
        else:
            counts_percent = [(c / total_count) * 100.0 for c in counts_raw]

        new_date = datetime.datetime.now().strftime("%d/%m/%y-%H:%M:%S")

        if self.data_is_placeholder:
            self.data = [counts_percent.copy()]
            self.dates = [new_date]
            self.data_is_placeholder = False
        else:
            self.data.append(counts_percent.copy())
            self.dates.append(new_date)

        self.ax_bar.cla()
        bottom = [0] * len(self.dates)

        for i in range(len(self.label_probabilidad_enfermedad)):
            # --- MODIFICACIÓN DE SEGURIDAD ---
            # Asegura que no falle si 'd' es más corto que 'i'
            values = [d[i] if i < len(d) else 0 for d in self.data]
            # --- FIN MODIFICACIÓN ---

            self.ax_bar.bar(self.dates, values, bottom=bottom,
                            color=self.label_colores_prob_enfermedad[i],
                            label=self.label_probabilidad_enfermedad[i])
            bottom = [b + v for b, v in zip(bottom, values)]

        self.ax_bar.tick_params(axis='x', rotation=20, labelsize=8)
        self.ax_bar.set_ylabel('% del Total de diagnósticos')
        self.ax_bar.set_xlabel('Fecha de diagnóstico')
        self.ax_bar.legend(loc='upper right', fontsize=8)
        self.ax_bar.set_ylim(0, 100)
        self.canvas_bar.draw()


# --- PÁGINA DIAGNOSTICAR ---
class page_diagnosticar(QWidget):
    # Emisor de señales para comunicar en las otras clases los resultados del diagnostico de IA
    diagnostico_completo = pyqtSignal(dict, dict, dict)

    # ### --- AÑADIDO: SEÑALES PARA WORKERS --- ###
    start_ssh = pyqtSignal(str)
    start_sftp_download = pyqtSignal(str, str)
    start_prediction = pyqtSignal(str)
    # ### --- FIN AÑADIDO --- ###

    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.current_results = []
        self.current_state_filename = {}
        self.conectado = False
        self.counts = []
        
        # ### --- AÑADIDO: RUTAS PREDEFINIDAS --- ###
        # Esto será la raíz de tu proyecto (ej. C:\...GUI-UAV)
        project_root = os.path.dirname(os.path.realpath(__file__))

        # Ahora, crea la ruta de guardado dentro de esa carpeta
        self.PREDEFINED_SAVE_PATH = os.path.join(project_root, "fotos_path_pruebas_2")
        # 2. Ruta remota (en la Raspberry Pi) DE DONDE SE LEERÁN
        # ¡¡ASEGÚRATE DE CAMBIAR ESTA RUTA POR TU RUTA REAL EN LA PI!!
        self.PREDEFINED_REMOTE_PATH = "/home/pera/Downloads/photos/photo_path" 
        # ### --- FIN AÑADIDO --- ###

        self.label_colores_prob_enfermedad = ['#006A35', '#34A853', '#FBBC04', '#F47C34', '#EA4335']
        self.label_probabilidad_enfermedad = [
            'Saludables', 'Leves rasgos', 'Rasgos considerables', 'Rasgos altos', 'Enfermas'
        ]

        self.perimeter_points = []
        self.start_point = None
        
        # ### --- AÑADIDO: INICIALIZACIÓN DE WORKERS --- ###
        # Worker SSH (Página 3)
        self.ssh_thread = QThread(self)
        self.ssh_worker = SshWorker()
        self.ssh_worker.moveToThread(self.ssh_thread)
        self.start_ssh.connect(self.ssh_worker.run_ssh_command)
        self.ssh_worker.progress.connect(self.on_ssh_progress)
        self.ssh_worker.finished.connect(self.on_ssh_finished)
        self.ssh_worker.error.connect(self.on_ssh_error)
        self.ssh_thread.start()

        # Worker SFTP (Página 4)
        self.sftp_thread = QThread(self)
        self.sftp_worker = SftpWorker()
        self.sftp_worker.moveToThread(self.sftp_thread)
        self.start_sftp_download.connect(self.sftp_worker.download_files)
        self.sftp_worker.progress.connect(self.on_download_progress)
        self.sftp_worker.finished.connect(self.on_download_complete)
        self.sftp_worker.error.connect(self.on_download_error)
        self.sftp_thread.start()

        # Worker de Predicción (Página 4)
        self.prediction_thread = QThread(self)
        self.prediction_worker = PredictionWorker()
        self.prediction_worker.moveToThread(self.prediction_thread)
        self.start_prediction.connect(self.prediction_worker.run_prediction)
        self.prediction_worker.progress.connect(self.on_prediction_progress)
        self.prediction_worker.finished.connect(self.on_prediction_finished)
        self.prediction_worker.error.connect(self.on_prediction_error)
        self.prediction_thread.start()
        # ### --- FIN AÑADIDO --- ###

        self.initUI()

    def initUI(self):
        # (Tu initUI está bien, no cambia)
        self.layout = QVBoxLayout(self)
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        self.page0 = self.create_page0()
        self.stacked_widget.addWidget(self.page0)
        self.page1 = self.create_page1()
        self.stacked_widget.addWidget(self.page1)
        self.page2 = self.create_page2()
        self.stacked_widget.addWidget(self.page2)
        self.page3 = self.create_page3()
        self.stacked_widget.addWidget(self.page3)
        self.page4 = self.create_page4()
        self.stacked_widget.addWidget(self.page4)
        self.page5 = self.create_page5()
        self.stacked_widget.addWidget(self.page5)
        self.update_page()

    # --- PÁGINA 0: Conexión ---
    def create_page0(self):
        # (Tu create_page0 está bien)
        page = QWidget()
        self.page1_layout = QVBoxLayout(page)
        title = QLabel("Paso previo del Punto de Despegue")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.page1_layout.addWidget(title)
        self.locked_label = QLabel("Conecta el UAV para comenzar.")
        self.locked_label.setStyleSheet("color: Black; font-size: 20px; font-weight: bold; qproperty-alignment: 'AlignCenter';")
        self.page1_layout.addWidget(self.locked_label)
        return page

    # --- PÁGINA 1: Punto de Despegue ---
    def create_page1(self):
        # (Tu create_page1 está bien)
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Paso 1: Punto de Despegue")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        self.status_label1 = QLabel("Ubicación actual del UAV.")
        self.status_label1.setStyleSheet("color: green; font-size: 19px;")
        layout.addWidget(self.status_label1)
        self.coord_list_widget1 = QListWidget()
        self.coord_list_widget1.setMaximumHeight(20)
        layout.addWidget(self.coord_list_widget1)
        self.web_view1 = QWebEngineView()
        layout.addWidget(self.web_view1, 1)
        self.bridge1 = Bridge()
        self.channel1 = QWebChannel()
        self.channel1.registerObject("bridge", self.bridge1)
        self.web_view1.page().setWebChannel(self.channel1)
        self.web_view1.setHtml(self.get_map_html(), QUrl("qrc:///"))
        btn_actualizar = QPushButton("Actualizar puntos")
        btn_actualizar.setStyleSheet("background-color: #1D8777; color: white;")
        btn_actualizar.clicked.connect(self.up_to_date_map1)
        btn_siguiente = QPushButton("Siguiente")
        btn_siguiente.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_siguiente.clicked.connect(self.go_to_step2)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_actualizar)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_siguiente)
        layout.addLayout(btn_layout)
        return page

    # --- PÁGINA 2: Área de Monitoreo ---
    def create_page2(self):
        # (Tu create_page2 está bien)
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Paso 2: Área de monitoreo")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        self.status_label2 = QLabel("Haz clic en el mapa para seleccionar los 4 puntos del perímetro.")
        self.status_label2.setStyleSheet("color: green; font-size: 19px")
        layout.addWidget(self.status_label2)
        self.coord_list_widget2 = QListWidget()
        self.coord_list_widget2.setMaximumHeight(80)
        layout.addWidget(self.coord_list_widget2)
        self.web_view2 = QWebEngineView()
        layout.addWidget(self.web_view2, 1)
        self.bridge2 = Bridge()
        self.channel2 = QWebChannel()
        self.channel2.registerObject("bridge", self.bridge2)
        self.web_view2.page().setWebChannel(self.channel2)
        self.bridge2.mapClicked.connect(self.handle_perimeter_map_click)
        self.web_view2.setHtml(self.get_map_html(), QUrl("qrc:///"))
        btn_regresar = QPushButton("Regresar")
        btn_regresar.setStyleSheet("background-color: #B2ADA9; color: black;")
        btn_regresar.clicked.connect(self.come_back_to_step1)
        btn_deshacer = QPushButton("Limpiar Selección")
        btn_deshacer.setStyleSheet("background-color: #f44336; color: white;")
        btn_deshacer.clicked.connect(self.clear_perimeter_markers)
        btn_actualizar = QPushButton("Actualizar puntos")
        btn_actualizar.setStyleSheet("background-color: #1D8777; color: white;")
        btn_actualizar.clicked.connect(self.up_to_date_map2)
        btn_siguiente = QPushButton("Siguiente")
        btn_siguiente.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_siguiente.clicked.connect(self.go_to_step3)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_regresar)
        btn_layout.addWidget(btn_deshacer)
        btn_layout.addWidget(btn_actualizar)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_siguiente)
        layout.addLayout(btn_layout)
        return page

    # --- PÁGINA 3: Misión en Progreso ---
    # ### --- MODIFICADO --- ###
    def create_page3(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Paso 3: Realizando diagnostico!")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        layout.addWidget(title)
        
        # Añadimos labels de progreso que podamos actualizar
        self.p3_progress_percent_label = QLabel("0% monitoreado")
        self.p3_progress_percent_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.p3_progress_percent_label)

        self.p3_progress_status_label = QLabel("El UAV se encuentra en movimiento.")
        self.p3_progress_status_label.setStyleSheet(
            "color: Black; font-size: 20px; font-weight: bold; qproperty-alignment: 'AlignCenter';")
        layout.addWidget(self.p3_progress_status_label)

        btn_abortar = QPushButton("Abortar operación")
        btn_abortar.setStyleSheet("background-color: #f44336; color: white;")
        btn_abortar.clicked.connect(self.abort) # (Abortar un hilo es complejo, por ahora solo avanza)

        # Botón "Siguiente" deshabilitado hasta que SshWorker termine
        self.btn_page3_siguiente = QPushButton("Siguiente")
        self.btn_page3_siguiente.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_page3_siguiente.clicked.connect(self.go_to_step4)
        self.btn_page3_siguiente.setEnabled(False) # <--- Deshabilitado al inicio

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_abortar)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_page3_siguiente)
        layout.addLayout(btn_layout)

        return page

    # --- PÁGINA 4: Descarga y Procesamiento ---
    # ### --- MODIFICADO --- ###
    def create_page4(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Paso 4: Obtención de resultados")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        layout.addWidget(title)

        # Labels de estado para descarga y predicción
        self.p4_status_label = QLabel("Listo para descargar y procesar.")
        self.p4_status_label.setStyleSheet(
            "color: Black; font-size: 20px; font-weight: bold; qproperty-alignment: 'AlignCenter';")
        
        self.p4_progress_label = QLabel("Presione el botón para comenzar.")
        self.p4_progress_label.setStyleSheet("font-size: 16px; qproperty-alignment: 'AlignCenter';")
        
        layout.addWidget(self.p4_progress_label)
        layout.addWidget(self.p4_status_label)

        btn_abortar = QPushButton("Abortar procesamiento")
        btn_abortar.setStyleSheet("background-color: #f44336; color: white;")
        btn_abortar.clicked.connect(self.abort)

        self.btn_ejecutar = QPushButton("Ejecutar procesamiento")
        self.btn_ejecutar.setStyleSheet("background-color: #8EC5FF; color: Black;")
        # Conecta al nuevo método que coordina todo
        self.btn_ejecutar.clicked.connect(self.start_download_and_predict) 
        
        self.btn_siguiente = QPushButton("Siguiente")
        self.btn_siguiente.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_siguiente.clicked.connect(self.go_to_step5)
        self.btn_siguiente.hide()  # Oculto hasta que se ejecute el modelo

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_abortar)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ejecutar)
        btn_layout.addWidget(self.btn_siguiente)
        layout.addLayout(btn_layout)

        return page

    # --- PÁGINA 5: Resultados Finales ---
    def create_page5(self):
        # (Tu create_page5 está bien)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        title = QLabel("¡Diagnóstico Finalizado!")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: green;")
        layout.addWidget(title)
        fecha = datetime.datetime.today().strftime("%d/%m/%y-%H:%M")
        subtitle = QLabel("Resultados de diagnóstico - " + fecha + " hrs.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #555;")
        layout.addWidget(subtitle)
        content_card = QFrame()
        content_card.setObjectName("contentCard")
        content_card.setFrameShape(QFrame.StyledPanel)
        content_layout = QHBoxLayout(content_card)
        map_frame = QFrame()
        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.web_view4 = QWebEngineView()
        # --- ERROR CORREGIDO ---
        # self.web_view4 estaba siendo añadido al layout principal,
        # pero también a map_layout. Lo dejamos solo en map_layout.
        # layout.addWidget(self.web_view4, 1) # <--- ELIMINAR ESTA LÍNEA
        self.bridge4 = Bridge()
        self.channel4 = QWebChannel()
        self.channel4.registerObject("bridge", self.bridge4)
        self.web_view4.page().setWebChannel(self.channel4)
        self.web_view4.setHtml(self.get_map_html(), QUrl("qrc:///"))
        map_layout.addWidget(self.web_view4)
        content_layout.addWidget(map_frame, 2)
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(15)
        labels_data = [f'{l}\n{s}' for l, s in zip(self.label_probabilidad_enfermedad, self.counts)]
        fig = Figure(figsize=(5, 4), dpi=100)
        fig.patch.set_alpha(0.0)
        self.ax = fig.add_subplot(111)
        self.ax.set_title('Clasificación del total de fotos tomadas', fontsize=12)
        # Dibujar con datos vacíos al inicio, se actualizará en go_to_step5
        self.ax.pie([1], labels=['Calculando...'], colors=['#E0E0E0'], startangle=90)
        self.ax.axis('equal')
        self.canvas_pie_5 = FigureCanvas(fig)
        self.canvas_pie_5.setStyleSheet("background-color: transparent;")
        right_panel_layout.addWidget(self.canvas_pie_5)
        btn_layout = QVBoxLayout()
        btn_guardar = QPushButton(" Guardar")
        btn_guardar.setIcon(QIcon.fromTheme("document-save"))
        btn_imprimir = QPushButton(" Imprimir")
        btn_imprimir.setIcon(QIcon.fromTheme("document-print"))
        btn_ver_fotos = QPushButton(" Ver fotos")
        btn_ver_fotos.setIcon(QIcon.fromTheme("camera-photo"))
        btn_terminar = QPushButton("Terminar")
        btn_terminar.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_terminar.setObjectName("btnTerminar")
        btn_guardar.clicked.connect(self.guardar_diagnostico)
        btn_imprimir.clicked.connect(lambda: QMessageBox.information(self, "Imprimir", "Imprimiendo..."))
        btn_ver_fotos.clicked.connect(lambda: QMessageBox.information(self, "Ver fotos", "Mostrando fotos..."))
        btn_terminar.clicked.connect(self.reset_diagnostic_ended)
        btn_layout.addWidget(btn_guardar)
        btn_layout.addWidget(btn_imprimir)
        btn_layout.addWidget(btn_ver_fotos)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(btn_terminar)
        right_panel_layout.addLayout(btn_layout)
        right_panel_layout.addStretch()
        content_layout.addLayout(right_panel_layout, 1)
        layout.addWidget(content_card)
        status_bar_frame = QFrame()
        status_bar_frame.setObjectName("statusBar")
        status_bar_frame.setFrameShape(QFrame.StyledPanel)
        status_bar_layout = QHBoxLayout(status_bar_frame)
        status_bar_layout.setContentsMargins(15, 10, 15, 10)
        status_bar_layout.setSpacing(15)
        self.label_ha = QLabel("Área de monitoreo: 1 ha.")
        status_bar_layout.addWidget(QLabel("Sensores: Buen estado"))
        status_bar_layout.addWidget(QLabel("Batería: 41%"))
        status_bar_layout.addStretch()
        status_bar_layout.addWidget(self.label_ha)
        status_bar_layout.addWidget(QLabel("Tiempo de vuelo: 7 min"))
        status_bar_layout.addWidget(QLabel("Tiempo de análisis: 0h 39 min"))
        layout.addWidget(status_bar_frame)
        
        # ... (Tu segunda barra de estado) ...
        # (El código de la segunda barra de estado es cosmético y está bien)
        
        return page

    # --- HTML del Mapa ---
    def get_map_html(self):
        # (Tu get_map_html está bien)
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Map</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
            <style>
                body { margin: 0; padding: 0; }
                #map { height: 100vh; width: 100%; }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([20.432939, -99.598862], 18);
                L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                    attribution: 'mapa interactuable',
                    maxNativeZoom: 18,
                    maxZoom: 22
                }).addTo(map);
                var markerLayer = L.layerGroup().addTo(map);
                var polygonLayer = L.layerGroup().addTo(map);
                var pythonBridge; 
                new QWebChannel(qt.webChannelTransport, function(channel) {
                    pythonBridge = channel.objects.bridge;
                });
                map.on('click', function(e) {
                    if (pythonBridge) {
                        pythonBridge.onMapClicked(e.latlng.lat, e.latlng.lng);
                    }
                });
                function addLocationMarker(lat, lng, color) {
                    L.circleMarker([lat, lng], {
                        radius: 2, color: color, fillColor: color, fillOpacity: 0.8
                    }).bindPopup("Ubicación actual").addTo(markerLayer);
                }
                function addMarker(lat, lng, color) {
                    L.circleMarker([lat, lng], {
                        radius: 2, color: color, fillColor: color, fillOpacity: 0.8
                    }).addTo(markerLayer);
                }
                function addStateMark(lat, lng, color, clase) {
                    var text = `${clase}<br>Lat: ${lat.toFixed(5)}<br>Lng: ${lng.toFixed(5)}`;
                    L.circleMarker([lat, lng], {
                        radius: 30, color: color, fillColor: color, fillOpacity: 0.6
                    }).bindPopup(text).addTo(markerLayer);
                }
                function drawPolygon(points_json) {
                    polygonLayer.clearLayers();
                    var points = JSON.parse(points_json);
                    if (points && points.length >= 3) {
                        L.polygon(points, {
                            color: 'red', weight: 2, fillColor: '#ff0000', fillOpacity: 0.2
                        }).addTo(polygonLayer);
                    }
                }
                function drawLastPolygon(points_json) {
                    polygonLayer.clearLayers();
                    var points = JSON.parse(points_json);
                    L.polygon(points, {
                        color: '#5D6D7E', weight: 2, fillColor: '#D6DBDF', fillOpacity: 0.2
                    }).addTo(polygonLayer);
                }
                function clearMarkers() {
                    markerLayer.clearLayers();
                    polygonLayer.clearLayers();
                }
            </script>
        </body>
        </html>
        """

    # --- Slots de Conexión y Mapa ---
    @pyqtSlot(bool, float, float)
    def set_estado_conexion(self, conectado, lat, long):
        """
        Este es el SLOT que recibe la señal desde MainWindow.
        """
        self.coordenadas_iniciales = (lat, long)  
        self.conectado = conectado
        if conectado:
            self.go_to_step1()

    @pyqtSlot(float, float)
    def handle_start_point_map_click(self, lat, lng):
        # (Tu código está bien)
        if self.start_point is not None:
            item_text = f"Punto de inicio: ({lat:.5f}, {lng:.5f})"
            self.coord_list_widget1.addItem(item_text)
            self.web_view1.page().runJavaScript(f"addLocationMarker({lat}, {lng}, 'lightgreen');")
            self.status_label1 = QLabel("Ubicación actual del UAV.")
            self.status_label1.setStyleSheet("color: green; font-size: 19px;")
        else:
            self.status_label1.setText("Solo se puede seleccionar 1 punto. Limpie para reiniciar.")

    @pyqtSlot(float, float)
    def handle_perimeter_map_click(self, lat, lng):
        # (Tu código está bien)
        if len(self.perimeter_points) < 4:
            self.perimeter_points.append((lat, lng))
            item_text = f"Punto {len(self.perimeter_points)}: ({lat:.5f}, {lng:.5f})"
            self.coord_list_widget2.addItem(item_text)
            self.web_view2.page().runJavaScript(f"addMarker({lat}, {lng}, 'red');")
            if len(self.perimeter_points) >= 3:
                points_json = json.dumps(self.perimeter_points)
                self.web_view2.page().runJavaScript(f"drawPolygon('{points_json}');")
            if len(self.perimeter_points) == 4:
                self.status_label2.setText("Perímetro de 4 puntos seleccionado.")
        else:
            self.status_label2.setText("Máximo de 4 puntos alcanzado. Limpie para reiniciar.")

    # --- Funciones de Geometría ---
    def distance(self, punto1, punto2):
        # (Tu código está bien)
        if not punto1 or not punto2:
            return 0.0
        return geodesic(punto1, punto2).meters

    def calcular_hectarea(self):
        # (Tu código está bien)
        if len(self.perimeter_points) < 3:
            return 0.0
        geod = Geod(ellps='WGS84')
        puntos_poligono = self.perimeter_points + [self.perimeter_points[0]]
        puntos_lon_lat = [(lon, lat) for lat, lon in puntos_poligono]
        polygon = Polygon(puntos_lon_lat)
        area, perimetro = geod.geometry_area_perimeter(polygon)
        hectarea = area * 0.0001 # <-- CORRECCIÓN: 1 m² = 0.0001 ha
        return abs(hectarea)

    # --- Métodos de Limpieza y Actualización de Mapa ---
    def clear_start_point_marker(self):
        # (Tu código está bien)
        self.start_point = (20.432939, -99.598862)
        self.coord_list_widget1.clear()
        self.web_view1.page().runJavaScript("clearMarkers();")
        self.status_label1 = QLabel("Ubicación actual del UAV.")
        self.status_label1.setStyleSheet("color: green; font-size: 19px;")

    def up_to_date_map1(self):
        # (Tu código está bien)
        self.web_view1.page().runJavaScript("clearMarkers();")
        self.web_view1.page().runJavaScript(f"addLocationMarker({self.start_point[0]}, {self.start_point[1]}, 'lightgreen');")

    def up_to_date_map2(self):
        # (Tu código está bien)
        self.web_view2.page().runJavaScript("clearMarkers();")
        self.web_view2.page().runJavaScript(f"addLocationMarker({self.start_point[0]}, {self.start_point[1]}, 'lightgreen');")
        for lat, lng in self.perimeter_points:
            self.web_view2.page().runJavaScript(f"addMarker({lat}, {lng}, 'red');")
        else:
            points_json = json.dumps(self.perimeter_points)
            self.web_view2.page().runJavaScript(f"drawPolygon('{points_json}');")

    def clear_perimeter_markers(self):
        # (Tu código está bien)
        self.perimeter_points = []
        self.coord_list_widget2.clear()
        self.web_view2.page().runJavaScript("clearMarkers();")
        self.web_view2.page().runJavaScript(f"addLocationMarker({self.start_point[0]}, {self.start_point[1]}, 'lightgreen');")
        self.status_label2.setText("Haz clic en el mapa para seleccionar los 4 puntos del perímetro.")

    # --- Navegación entre Páginas (GO_TO_STEP) ---
    def go_to_step1(self):
        if self.conectado:
            self.current_step = 1
            #self.coordenadas_iniciales = (20.432939, -99.598862)  ######### pendiente
            lat, lng = self.coordenadas_iniciales
            self.start_point = (lat, lng)
            zoom = 18
            self.web_view1.page().runJavaScript(f"map.setView([{lat}, {lng}], {zoom});")
            self.web_view1.page().runJavaScript(f"addMarker({lat}, {lng}, 'lightgreen');")
            self.handle_start_point_map_click(lat, lng)
            self.update_page()

    def go_to_step2(self):
        # (Tu código está bien)
        if self.start_point is not None:
            self.current_step = 2
            lat, lng = self.start_point
            zoom = 18
            self.web_view2.page().runJavaScript(f"map.setView([{lat}, {lng}], {zoom});")
            if self.start_point is not None:
                self.web_view2.page().runJavaScript(f"addLocationMarker({lat}, {lng}, 'lightgreen');")
            self.update_page()
        else:
            QMessageBox.warning(self, "Error", "Debes seleccionar un punto de despegue.")

    # ### --- MODIFICADO: go_to_step3 --- ###
    def go_to_step3(self):
        # Validaciones (Tu código)
        if len(self.perimeter_points) != 4:
            QMessageBox.warning(self, "Error", "Debes seleccionar 4 puntos.")
            return
        hectareas = self.calcular_hectarea()
        if hectareas > 1.00:
            QMessageBox.warning(self, "Error", f'{hectareas:.3f} ha seleccionados.\nSolo se permite 1.00 ha como máximo')
            return
        
        # Lógica del Worker (Nuevo)
        self.current_step = 3
        self.update_page()
        
        # Resetea los labels de progreso de la Página 3
        self.p3_progress_status_label.setText("Iniciando conexión con el UAV...")
        self.p3_progress_percent_label.setText("0% monitoreado")
        self.btn_page3_siguiente.setEnabled(False) # Asegurarse de que esté deshabilitado

        # Prepara los datos y emite la señal para iniciar el SshWorker
        json_payload = json.dumps(self.perimeter_points)
        self.start_ssh.emit(json_payload)

    def go_to_step4(self):
        # (Tu código está bien, solo es un switch de página)
        self.current_step = 4
        self.label_ha.setText("Área monitoreado:" + f'{self.calcular_hectarea():.3f}' + "ha.")
        self.update_page()
        
        # Resetea los labels de la Página 4
        self.p4_status_label.setText("Listo para descargar y procesar.")
        self.p4_progress_label.setText(f"Se descargarán fotos de: {self.PREDEFINED_REMOTE_PATH}")
        self.btn_ejecutar.setEnabled(True)
        self.btn_ejecutar.show()
        self.btn_siguiente.hide()


    def go_to_step5(self):
        # (Tu código de go_to_step5 está bien)
        if (len(self.perimeter_points) == 4) and (self.current_results is not None):
            self.current_step = 5
            self.web_view4.page().runJavaScript("clearMarkers();")
            self.web_view4.page().runJavaScript(f"map.setView([{self.start_point[0]}, {self.start_point[1]}], {18});")
            self.web_view4.page().runJavaScript(f"addLocationMarker({self.start_point[0]}, {self.start_point[1]}, 'lightgreen');")
            for p in self.perimeter_points:
                self.web_view4.page().runJavaScript(f"addMarker({p[0]}, {p[1]}, '#5D6D7E');")
            else:
                points_json = json.dumps(self.perimeter_points)
                self.web_view4.page().runJavaScript(f"drawLastPolygon('{points_json}');")
            
            for i, estado in enumerate(self.current_state_filename.keys()):
                color = self.label_colores_prob_enfermedad[i]
                for coord in self.current_state_filename[estado]:
                    print(i, estado, coord)
                    self.web_view4.page().runJavaScript(f"addStateMark({coord[0]}, {coord[1]}, '{color}', '{estado}');")
            
            if len(self.counts) != 0:
                self.ax.cla()
                labels_data = [f'{l}\n{s}' for l, s in zip(self.label_probabilidad_enfermedad, self.counts)]
                print(self.counts)
                print(labels_data)
                self.ax.pie(self.counts, labels=labels_data, colors=self.label_colores_prob_enfermedad, startangle=90,
                            textprops={'fontsize': 9})
                self.ax.axis('equal')
                self.ax.set_title('Clasificación del total de fotos tomadas', fontsize=12)
                self.canvas_pie_5.draw()

            self.update_page()
        else:
            QMessageBox.warning(self, "Error", "No se han procesado resultados o falta el perímetro.")

    # --- Guardar y Resetear ---
    def guardar_diagnostico(self):
        # (Tu código está bien)
        fecha_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostico_mapa_{fecha_hora}.png"
        try:
            pixmap = self.web_view4.grab()
            # --- MODIFICACIÓN DE RUTA ---
            # Asegura que la carpeta exista
            save_dir = "./diagnosticos_guardados/"
            os.makedirs(save_dir, exist_ok=True)
            success = pixmap.save(os.path.join(save_dir, filename), "PNG")
            # --- FIN MODIFICACIÓN ---

            if success: QMessageBox.information(self,"Guardado Exitoso", f"El mapa se ha guardado como:\n{filename}")
            else: QMessageBox.warning(self, "Error al Guardar", "No se pudo guardar la imagen del mapa.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Ocurrió un error al capturar el mapa: {e}")

    def come_back_to_step1(self):
        # (Tu código está bien)
        self.clear_perimeter_markers()
        self.current_step = 1
        self.update_page()

    def abort(self):
        # (Tu código está bien)
        # Nota: Abortar hilos es complejo. Por ahora, solo avanza.
        self.current_step = 4
        self.update_page()

    def reset_diagnostic(self):
        # (Tu código está bien)
        self.current_step = 1 if self.conectado else 0
        self.clear_perimeter_markers()
        self.clear_start_point_marker()
        self.update_page()

    def reset_diagnostic_ended(self):
        # (Tu código está bien)
        self.clear_perimeter_markers()
        self.clear_start_point_marker()
        self.counts = []
        self.current_results = []
        self.current_step = 0
        self.btn_ejecutar.show()
        self.btn_siguiente.hide()
        self.btn_ejecutar.setText("Ejecutar procesamiento")
        self.btn_ejecutar.setEnabled(True)
        QMessageBox.information(self, "Monitoreo finalizado", "Puede ver los resultados en la opción <b>Tablero</b>")
        self.go_to_step1()

    def update_page(self):
        # (Tu código está bien)
        self.stacked_widget.setCurrentIndex(self.current_step)

    # --- Lógica de Procesamiento (Workers) ---

    # ### --- REEMPLAZADO: start_prediction --- ###
    # Esta función ahora solo INICIA el proceso de descarga y predicción
    def start_download_and_predict(self):
        self.btn_ejecutar.setText("Descargando...")
        self.btn_ejecutar.setEnabled(False)
        self.p4_status_label.setText("Iniciando descarga de fotos...")
        QApplication.processEvents()
        
        # 1. Asegurarse de que la carpeta de guardado local exista
        try:
            os.makedirs(self.PREDEFINED_SAVE_PATH, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error de Directorio", f"No se pudo crear la carpeta local: {self.PREDEFINED_SAVE_PATH}\nError: {e}")
            self.btn_ejecutar.setText("Error")
            return
            
        # 2. Iniciar el SftpWorker
        self.start_sftp_download.emit(self.PREDEFINED_REMOTE_PATH, self.PREDEFINED_SAVE_PATH)

    # (La función start_prediction original fue movida al PredictionWorker)
    
    # ### --- AÑADIDOS: SLOTS DE LOS WORKERS --- ###
    
    # --- Slots de SSH (Página 3) ---
    @pyqtSlot(str)
    def on_ssh_progress(self, message):
        if "%" in message:
            self.p3_progress_percent_label.setText(message)
        else:
            self.p3_progress_status_label.setText(message)

    @pyqtSlot(str, str)
    def on_ssh_finished(self, out, err):
        self.p3_progress_status_label.setText("¡Vuelo completado!")
        self.p3_progress_percent_label.setText("100% monitoreado")
        self.btn_page3_siguiente.setEnabled(True) # Habilitar botón
        
        if err:
            print("--- ERRORES DEL SCRIPT SSH ---")
            print(err)

    @pyqtSlot(str)
    def on_ssh_error(self, error_message):
        QMessageBox.critical(self, "Error de Vuelo", error_message)
        self.come_back_to_step1() # Regresar al paso 1

    # --- Slots de SFTP (Página 4) ---
    @pyqtSlot(str)
    def on_download_progress(self, message):
        self.p4_status_label.setText("Descargando...")
        self.p4_progress_label.setText(message)

    @pyqtSlot()
    def on_download_complete(self):
        self.p4_status_label.setText("Descarga completa. Iniciando IA...")
        self.p4_progress_label.setText("Cargando modelo...")
        
        # 3. Iniciar el PredictionWorker
        self.start_prediction.emit(self.PREDEFINED_SAVE_PATH)

    @pyqtSlot(str)
    def on_download_error(self, error_message):
        QMessageBox.critical(self, "Error de Descarga", error_message)
        self.btn_ejecutar.setText("Ejecutar procesamiento")
        self.btn_ejecutar.setEnabled(True)

    # --- Slots de Predicción (Página 4) ---
    @pyqtSlot(str)
    def on_prediction_progress(self, message):
        self.p4_status_label.setText("Procesando con IA...")
        self.p4_progress_label.setText(message)

    @pyqtSlot(dict, dict, dict, dict, list)
    def on_prediction_finished(self, class_counts, class_file_lists, class_leaf_state, state_file_lists, nuevos_conteos):
        self.p4_status_label.setText("¡Procesamiento finalizado!")
        self.p4_progress_label.setText("Resultados listos.")
        
        # Guardar los resultados en la instancia
        self.current_results = class_counts # O el diccionario que más te sirva
        self.current_state_filename = state_file_lists
        self.counts = nuevos_conteos

        # Enviar señal a page_Tablero
        self.diagnostico_completo.emit(class_counts, class_file_lists, class_leaf_state)
        
        # Actualizar UI de la Página 4
        self.btn_ejecutar.hide()
        self.btn_siguiente.show()

    @pyqtSlot(str)
    def on_prediction_error(self, error_message):
        QMessageBox.critical(self, "Error de Predicción", error_message)
        self.btn_ejecutar.setText("Ejecutar procesamiento")
        self.btn_ejecutar.setEnabled(True)

# --- PÁGINA ESTADÍSTICAS ---
class page_Estadisticas(QWidget):
    # (Tu código de page_Estadisticas está bien)
    def __init__(self, parent=None):
        super(page_Estadisticas, self).__init__(parent)
        self.class_counts = {}
        self.initUI()
    
    def initUI(self):
        # ... (Tu código initUI está bien) ...
        self.setWindowTitle("Tablero de estadisticas")
        layout = QVBoxLayout(self)
        title_label = QLabel("Estadísticas Generales")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        main_container = QHBoxLayout()
        line_chart_container = QVBoxLayout()
        line_title = QLabel("Evolución mensual de diagnósticos")
        line_title.setStyleSheet("font-weight: bold;")
        line_chart_container.addWidget(line_title)
        fig_line = Figure(figsize=(8, 5), dpi=100)
        ax_line = fig_line.add_subplot(111)
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
        diagnoses = [120, 150, 180, 200, 220, 250]
        ax_line.plot(months, diagnoses, marker='o', linewidth=2, color='#007BFF')
        ax_line.set_ylabel('Cantidad de diagnósticos')
        ax_line.set_xlabel('Mes')
        ax_line.grid(True, linestyle='--', alpha=0.6)
        canvas_line = FigureCanvas(fig_line)
        line_chart_container.addWidget(canvas_line)
        main_container.addLayout(line_chart_container)
        summary_container = QVBoxLayout()
        summary_title = QLabel("Resumen General")
        summary_title.setStyleSheet("font-weight: bold;")
        summary_container.addWidget(summary_title)
        stats = [
            ("Total diagnósticos", "1,220"),
            ("Promedio mensual", "203"),
            ("Diagnósticos saludables", "85%"),
            ("Diagnósticos con problemas", "15%"),
            ("Mejora promedio", "+8%")
        ]
        for label, value in stats:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            val_label = QLabel(value)
            val_label.setStyleSheet("font-weight: bold; font-size: 16px;")
            row.addWidget(val_label)
            row.addStretch()
            summary_container.addLayout(row)
        main_container.addLayout(summary_container)
        layout.addLayout(main_container)
        export_btn = QPushButton("Exportar reporte")
        export_btn.setStyleSheet("background-color: #007BFF; color: white; padding: 10px;")
        layout.addWidget(export_btn)
        layout.addStretch()

    @pyqtSlot(dict, dict, dict)
    def set_result_plots(self, class_counts, list_results_per_class, list_leaf_state):
        self.class_counts = class_counts
        self.list_results_per_class = list_results_per_class
        print("Recibidos en", self.__class__.__name__)


class InteractiveMapView(QGraphicsView):
    # (Tu clase InteractiveMapView no cambia)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)
        self.background_pixmap = QPixmap("./icons/field_map.png")
        if self.background_pixmap.isNull():
            self.background_pixmap = QPixmap(600, 400)
            self.background_pixmap.fill(Qt.lightGray)
        self.bg_item = QGraphicsPixmapItem(self.background_pixmap)
        self.scene.addItem(self.bg_item)
        self.selected_point = None
        self.perimeter_points = []
        self.polygon_item = None
        self.parent = parent
    
    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            if isinstance(self.parent, page_diagnosticar):
                if self.parent.current_step == 0:
                    self.clear_selection()
                    self.selected_point = pos
                    self.draw_point(pos)
                    if hasattr(self.parent, 'status_label'):
                        self.parent.status_label.setText(f"Ubicación seleccionada: ({pos.x():.6f}, {pos.y():.6f})")
                elif self.parent.current_step == 1:
                    self.perimeter_points.append(pos)
                    self.draw_point(pos, color=Qt.red)
                    if len(self.perimeter_points) >= 3:
                        self.draw_polygon()
    
    def draw_point(self, pos, color=Qt.green):
        ellipse = self.scene.addEllipse(pos.x() - 5, pos.y() - 5, 10, 10, QPen(color), QBrush(color))
        ellipse.setZValue(10)
    
    def draw_polygon(self):
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        if len(self.perimeter_points) >= 3:
            polygon = QPolygonF([QPointF(p.x(), p.y()) for p in self.perimeter_points])
            self.polygon_item = self.scene.addPolygon(polygon, QPen(Qt.red, 2), QBrush(Qt.transparent))
            self.polygon_item.setZValue(5)
    
    def clear_selection(self):
        self.selected_point = None
        self.perimeter_points = []
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon_item = None
        for item in self.scene.items():
            if isinstance(item, QGraphicsEllipseItem):
                self.scene.removeItem(item)
    
    def reset(self):
        self.clear_selection()
        self.scene.clear()
        self.bg_item = QGraphicsPixmapItem(self.background_pixmap)
        self.scene.addItem(self.bg_item)