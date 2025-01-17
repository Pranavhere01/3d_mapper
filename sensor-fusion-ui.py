from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (QFileDialog, QMessageBox, QProgressBar, 
                           QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                           QWidget, QLabel, QPushButton, QRadioButton, 
                           QComboBox)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
from datetime import datetime 
import os
import math
import traceback

class Ui_sensor_fusion(object):
    def load_csv1(self):
        if self.radioButton_1.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Temperature Data File",
                "",
                "CSV Files (*.csv)"
            )
            if file_path:
                try:
                    self.temp_data = pd.read_csv(file_path)
                    self.temp_data['date'] = pd.to_datetime(self.temp_data['date'], format='%d-%m-%Y %H:%M')
                    print(f"Loaded Temperature Data: {self.temp_data.head()}")
                    self.label_file1_status.setText(f"Loaded: {os.path.basename(file_path)}")
                    self.show_success_message("Temperature data loaded successfully!")
                except Exception as e:
                    self.show_error_message(f"Error loading temperature data: {str(e)}")
            else:
                self.show_error_message("No file selected!")
        else:
            self.show_error_message("Please select the Temperature data radio button.")

    def load_csv2(self):
        if self.radioButton_2.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select CO2 Data File",
                "",
                "CSV Files (*.csv)"
            )
            if file_path:
                try:
                    self.co2_data = pd.read_csv(file_path)
                    self.co2_data['date'] = pd.to_datetime(self.co2_data['date'], format='%d-%m-%Y %H:%M')
                    self.label_file2_status.setText(f"Loaded: {os.path.basename(file_path)}")
                    self.show_success_message("CO2 data loaded successfully!")
                except Exception as e:
                    self.show_error_message(f"Error loading CO2 data: {str(e)}")
            else:
                self.show_error_message("No file selected!")
        else:
            self.show_error_message("Please select the CO2 data radio button.")

    def load_csv3(self):
        if self.radioButton_3.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Humidity Data File",
                "",
                "CSV Files (*.csv)"
            )
            if file_path:
                try:
                    self.humidity_data = pd.read_csv(file_path)
                    self.humidity_data['date'] = pd.to_datetime(self.humidity_data['date'], format='%d-%m-%Y %H:%M')
                    self.label_file3_status.setText(f"Loaded: {os.path.basename(file_path)}")
                    self.show_success_message("Humidity data loaded successfully!")
                except Exception as e:
                    self.show_error_message(f"Error loading humidity data: {str(e)}")
            else:
                self.show_error_message("No file selected!")
        else:
            self.show_error_message("Please select the Humidity data radio button.")

    def calc_sad(self, X):
        sad = 0
        X_filled = X.fillna(0)
        valid_indices = X_filled.index.tolist()
        for i in range(len(valid_indices) - 1):
            sad += abs(X_filled[valid_indices[i]] - X_filled[valid_indices[i + 1]])
        return sad
    
    def detect_onoff(self, X):
        timecount = 0
        thA = 30
        X_filled = X.fillna(0)
        valid_indices = X_filled.index.tolist()
        for i in range(len(valid_indices) - 1):
            if abs(X_filled[valid_indices[i]] - X_filled[valid_indices[i + 1]]) >= thA:
                timecount += 1
        return timecount

    def target(self):
        if self.radioButton_raw_data1.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "Select Occupancy Data Files",
                "",
                "CSV Files (*.csv)"
            )
            if file_paths:
                try:
                    data_frames = []
                    for file_path in file_paths:
                        print(f"Processing file: {file_path}")
                        df = pd.read_csv(file_path, header=0, index_col=0)
                        df.index = pd.to_datetime(df.index, format='%d-%b-%y', errors='coerce')
                        if df.index.isnull().any():
                            print(f"Warning: some dates could not be parsed in {file_path}")
                        df = df.apply(pd.to_numeric, errors='coerce')
                        data_frames.append(df)

                    combined_df = pd.concat(data_frames, axis=0, ignore_index=True)
                    print(f"Combined data shape: {combined_df.shape}")

                    df_numeric = combined_df.select_dtypes(include=[np.number])

                    n = 900
                    cuml_list = []
                    for index, row in df_numeric.iterrows():
                        occ_summer = []
                        list_oc_summer = [row[i:i+n] for i in range(0, len(row), n)]
                        for chunk in list_oc_summer:
                            chunk = np.array(chunk, dtype=int)
                            occ_summer.append(np.bincount(chunk).argmax())
                        occ_summer = occ_summer[24:89]
                        cuml_list.extend(occ_summer)
                    
                    occ_summer_data = pd.DataFrame(cuml_list, columns=['occupancy'])
                    output_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'New folder', 'occupancy_data1.csv')
                    occ_summer_data.to_csv(output_file, index=False)
                    self.occupancy_data = occ_summer_data
                    self.label_raw_data_status1.setText(f"Processed {len(file_paths)} files")
                    self.show_success_message("Occupancy data processed successfully!")

                except Exception as e:
                    self.show_error_message(f"Error processing occupancy data: {str(e)}")
                    traceback.print_exc()
            else:
                self.show_error_message("No files selected")
        else:
            self.show_error_message("Please select the occupancy radio button.")

    def load_raw_data(self):
        if self.radioButton_raw_data.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "Select Raw Data Files",
                "",
                "CSV Files (*.csv)"
            )
            if not file_paths:
                self.show_error_message("No files selected!")
                return

            try:
                df_summer = {}
                for file_path in file_paths:
                    raw_data = pd.read_csv(file_path, header=None)
                    
                    if raw_data.shape[1] != 16:
                        print(f"Warning: File {file_path} does not have expected columns. Found {raw_data.shape[1]} columns.")
                        continue

                    raw_data.columns = ['powerallphases', 'powerl1', 'powerl2', 'powerl3',
                                      'currentneutral', 'currentl1', 'currentl2', 'currentl3',
                                      'voltagel1', 'voltagel2', 'voltagel3',
                                      'phaseanglevoltagel2l1', 'phaseanglevoltagel3l1',
                                      'phaseanglecurrentvoltagel1', 'phaseanglecurrentvoltagel2',
                                      'phaseanglecurrentvoltagel3']
                    
                    day = os.path.basename(file_path)
                    df_summer[day] = raw_data

                # Process the data
                features = self.process_raw_data(df_summer)
                
                # Save processed data
                self.custom_data = features
                output_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'New folder', 'statistics2.csv')
                write_header = not os.path.exists(output_path)
                features.to_csv(output_path, mode='a', header=write_header, index=False)
                
                self.label_raw_data_status.setText(f"Processed {len(file_paths)} files")
                self.show_success_message("Raw data processed successfully!")
                
            except Exception as e:
                print(f"Detailed error loading raw data: {str(e)}")
                traceback.print_exc()
                self.show_error_message(f"Error processing raw data: {str(e)}")
        else:
            self.show_error_message("Please select the raw data radio button.")

    def process_raw_data(self, df_summer):
        n = 900
        features = pd.DataFrame()
        
        for day, data in df_summer.items():
            list_df_summer = [data[i:i+n] for i in range(0, len(data), n)]
            
            # Initialize feature lists
            feature_lists = self.initialize_feature_lists(list_df_summer)
            
            # Calculate features for each chunk
            for chunk in list_df_summer:
                self.calculate_chunk_features(chunk, feature_lists)
            
            # Create DataFrame from features
            data1 = self.create_features_dataframe(feature_lists)
            features = pd.concat([features, data1], ignore_index=True)
        
        # Post-process features
        features.loc[features.ptime < 25, 'ptime'] = 0
        features.loc[features.ptime > 89, 'ptime'] = 0
        features = features.loc[(features['ptime'] > 0)]
        features['ptime'] = features['ptime'] - 24
        
        return features

    def run_model(self):
        if not hasattr(self, 'custom_data') or not hasattr(self, 'occupancy_data'):
            self.show_error_message("Please load all required data files before running the model.")
            return
            
        try:
            # Prepare data
            self.custom_data.reset_index(drop=True, inplace=True)
            self.occupancy_data.reset_index(drop=True, inplace=True)
            featuredata = pd.concat([self.custom_data, self.occupancy_data], axis=1)
            
            # Save merged data
            output_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'New folder', 'mergedfile3.csv')
            featuredata.to_csv(output_file, index=False)
            
            # Prepare features and target
            X = featuredata.drop(columns=['occupancy', 'date'], errors='ignore')
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            y = pd.to_numeric(featuredata['occupancy'], errors='coerce').fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Get selected algorithm
            algorithm = self.comboBox_1.currentText().strip()
            
            # Initialize model
            if algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)
            elif algorithm == "SVM":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                model = SVC(kernel='rbf', C=1.0, gamma='scale')
            elif algorithm == "Naive Bayes":
                model = GaussianNB()
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                self.show_error_message("Invalid algorithm selected")
                return
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metric = self.comboBox_2.currentText()
            if metric == "Accuracy":
                result = f"{accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%"
            elif metric == "Confusion matrix":
                result = str(confusion_matrix(y_test, model.predict(X_test)))
            elif metric == "F1 Score":
                result = f"{f1_score(y_test, model.predict(X_test)):.4f}"
            else:
                self.show_error_message("Invalid metric selected")
                return
            
            self.final_result.setText(result)
            
        except Exception as e:
            self.show_error_message(f"Error running model: {str(e)}")
            traceback.print_exc()

    def setupUi(self, sensor_fusion):
        # Main window setup
        sensor_fusion.setObjectName("sensor_fusion")
        sensor_fusion.resize(600, 800)
        sensor_fusion.setStyleSheet("""
            QWidget {
                background-color: lavender;
                color: black;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border-radius: 4px;
                padding: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                background-color: lavender;
            }
            QLabel {
                color: black;
                padding: 2px;
            }
            QRadioButton {
                color: black;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
        """)

        # Main layout
        self.main_layout = QVBoxLayout(sensor_fusion)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        self.title_label = QLabel("Sensor Fusion")
        self.title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            color: #2a2a2a;
        """)
        self.main_layout.addWidget(self.title_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Data Input Section
        self.input_group = QGroupBox("Sensor Data Input")
        self.input_layout = QGridLayout()
        self.input_layout.setSpacing(10)

        # Temperature Row
        self.setup_input_row(0, "Temperature", "radioButton_1", "pushButton_4_csv1", "label_file1_status", 
                           "Load temperature sensor data")
        
        # CO2 Row
        self.setup_input_row(1, "CO2", "radioButton_2", "pushButton_5_csv2", "label_file2_status",
                           "Load CO2 sensor data")
        
        # Humidity Row
        self.setup_input_row(2, "Humidity", "radioButton_3", "pushButton_6_csv3", "label_file3_status",
                           "Load humidity sensor data")
        
        # Custom Data Row
        self.setup_input_row(3, "Custom", "radioButton_raw_data", "pushButton_raw_data", "label_raw_data_status",
                           "Load custom raw data files")
        
        # Occupancy Row
        self.setup_input_row(4, "Occupancy", "radioButton_raw_data1", "pushButton_raw_data1", "label_raw_data_status1",
                           "Load occupancy data files")

        self.input_group.setLayout(self.input_layout)
        self.main_layout.addWidget(self.input_group)

        # Model Configuration Section
        self.model_group = QGroupBox("Model Configuration")
        self.model_layout = QGridLayout()
        self.model_layout.setSpacing(10)

        # Algorithm Selection
        self.label_algorithm = QLabel("Algorithm:")
        self.comboBox_1 = QComboBox()
        self.comboBox_1.addItems(["Random Forest", "SVM", "Naive Bayes", "Decision Tree"])
        self.comboBox_1.setToolTip("Select the machine learning algorithm to use")
        self.model_layout.addWidget(self.label_algorithm, 0, 0)
        self.model_layout.addWidget(self.comboBox_1, 0, 1)

        # Metric Selection
        self.label_metric = QLabel("Evaluation Metric:")
        self.comboBox_2 = QComboBox()
        self.comboBox_2.addItems(["Accuracy", "Confusion matrix", "F1 Score"])
        self.comboBox_2.setToolTip("Select the metric to evaluate model performance")
        self.model_layout.addWidget(self.label_metric, 1, 0)
        self.model_layout.addWidget(self.comboBox_2, 1, 1)

        self.model_group.setLayout(self.model_layout)
        self.main_layout.addWidget(self.model_group)

        # Results Section
        self.results_group = QGroupBox("Results")
        self.results_layout = QVBoxLayout()
        
        # Run button and progress container
        self.run_container = QHBoxLayout()
        
        self.pushButton_run = QPushButton("Run Model")
        self.pushButton_run.setToolTip("Run the selected model with current configuration")
        self.pushButton_run.setMinimumWidth(100)
        self.run_container.addWidget(self.pushButton_run)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.run_container.addWidget(self.progress_bar)
        
        self.results_layout.addLayout(self.run_container)
        
        # Results display
        self.final_result = QLabel("Results will appear here")
        self.final_result.setStyleSheet("""
            font-weight: bold;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
        """)
        self.final_result.setWordWrap(True)
        self.results_layout.addWidget(self.final_result)
        
        self.results_group.setLayout(self.results_layout)
        self.main_layout.addWidget(self.results_group)

        # Connect signals
        self.setup_connections()

        # Add stretch at the bottom for better layout
        self.main_layout.addStretch()

    def setup_input_row(self, row, label_text, radio_name, button_name, status_name, tooltip):
        """Helper function to set up a row in the input section"""
        label = QLabel(label_text)
        radio = QRadioButton()
        button = QPushButton("Load File")
        status = QLabel("No file loaded")
        
        # Store references
        setattr(self, radio_name, radio)
        setattr(self, button_name, button)
        setattr(self, status_name, status)
        
        # Style status label
        status.setStyleSheet("""
            background-color: white;
            border-radius: 4px;
            padding: 4px;
        """)
        
        # Add tooltips
        button.setToolTip(tooltip)
        radio.setToolTip(f"Select for {label_text} data input")
        
        # Add to layout
        self.input_layout.addWidget(label, row, 0)
        self.input_layout.addWidget(radio, row, 1)
        self.input_layout.addWidget(button, row, 2)
        self.input_layout.addWidget(status, row, 3)

    def setup_connections(self):
        """Set up all signal connections"""
        self.pushButton_4_csv1.clicked.connect(self.load_csv1)
        self.pushButton_5_csv2.clicked.connect(self.load_csv2)
        self.pushButton_6_csv3.clicked.connect(self.load_csv3)
        self.pushButton_raw_data.clicked.connect(self.load_raw_data)
        self.pushButton_raw_data1.clicked.connect(self.target)
        self.pushButton_run.clicked.connect(self.run_model)