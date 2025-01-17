from PyQt6.QtWidgets import (QFileDialog, QMessageBox, QProgressBar,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                            QWidget, QLabel, QPushButton, QRadioButton, QComboBox)

from PyQt6.QtCore import Qt
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
                "CSV Files (*.csv)")
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

            df_summer = {}
            try:   
                for file_path in file_paths:
                    raw_data = pd.read_csv(file_path, header=None)
                    if raw_data.shape[1] != 16:
                        print(f"Warning: File {file_path} does not have 16 columns. It has {raw_data.shape[1]} columns.")
                        continue
                    print(raw_data.columns)
                    
                    if raw_data.empty:
                        print(f"Warning: {file_path} has no valid data after removing NaN values!")
                        continue

                 
                    raw_data.columns = ['powerallphases', 'powerl1', 'powerl2', 'powerl3',
                                        'currentneutral', 'currentl1', 'currentl2', 'currentl3',
                                        'voltagel1', 'voltagel2', 'voltagel3',
                                        'phaseanglevoltagel2l1', 'phaseanglevoltagel3l1',
                                        'phaseanglecurrentvoltagel1', 'phaseanglecurrentvoltagel2', 'phaseanglecurrentvoltagel3']
                    print(raw_data.head())
                    day = os.path.basename(file_path)
                    df_summer[day] = raw_data

                n = 900    
                features = pd.DataFrame()
                for day, data in df_summer.items():
                    list_df_summer = [data[i:i+n] for i in range(0, len(data), n)]
                    print(f"Number of chunks in local: {len(list_df_summer)}")

                    # Initialize all the lists for various features
                    meanp123, meanp1, meanp2, meanp3 = [], [], [], []
                    meancp123, meancp1, meancp2, meancp3 = [], [], [], []
                    meanvp123, meanvp1, meanvp2, meanvp3 = [], [], [], []
                    meancvp123, meancvp1, meancvp2, meancvp3 = [], [], [], []

                    stdp123, stdp1, stdp2, stdp3 = [], [], [], []
                    stdcp123, stdcp1, stdcp2, stdcp3 = [], [], [], []
                    stdvp123, stdvp1, stdvp2, stdvp3 = [], [], [], []
                    stdcvp123, stdcvp1, stdcvp2, stdcvp3 = [], [], [], []

                    sadp1, sadp2, sadp3, sadp123 = [], [], [], []
                    sadcp1, sadcp2, sadcp3, sadcp123 = [], [], [], []
                    sadvp1, sadvp2, sadvp3, sadvp123 = [], [], [], []
                    sadcvp1, sadcvp2, sadcvp3, sadcvp123 = [], [], [], []

                    rangep1, rangep2, rangep3, rangep123 = [], [], [], []
                    rangecp1, rangecp2, rangecp3, rangecp123 = [], [], [], []
                    rangevp1, rangevp2, rangevp3, rangevp123 = [], [], [], []
                    rangecvp1, rangecvp2, rangecvp3, rangecvp123 = [], [], [], []

                    maxp123, maxp1, maxp2, maxp3 = [], [], [], []
                    maxcp123, maxcp1, maxcp2, maxcp3 = [], [], [], []
                    maxvp123, maxvp1, maxvp2, maxvp3 = [], [], [], []
                    maxcvp123, maxcvp1, maxcvp2, maxcvp3 = [], [], [], []

                    minp123, minp1, minp2, minp3 = [], [], [], []
                    mincp123, mincp1, mincp2, mincp3 = [], [], [], []
                    minvp123, minvp1, minvp2, minvp3 = [], [], [], []
                    mincvp123, mincvp1, mincvp2, mincvp3 = [], [], [], []

                    corrp123, corrp1, corrp2, corrp3 = [], [], [], []
                    corrcp123, corrcp1, corrcp2, corrcp3 = [], [], [], []
                    corrvp123, corrvp1, corrvp2, corrvp3 = [], [], [], []
                    corrcvp123, corrcvp1, corrcvp2, corrcvp3 = [], [], [], []

                    onoffp123, onoffp1, onoffp2, onoffp3 = [], [], [], []
                    onoffcp123, onoffcp1, onoffcp2, onoffcp3 = [], [], [], []
                    onoffvp123, onoffvp1, onoffvp2, onoffvp3 = [], [], [], []
                    onoffcvp123, onoffcvp1, onoffcvp2, onoffcvp3 = [], [], [], []

                    ptime = []

                    for chunk in list_df_summer:
                        print(f"Chunk size: {len(chunk)}")
                        print(chunk.head())

                        # Calculate means
                        meanp123.append(np.mean(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        meanp1.append(np.mean(chunk['powerl1']))
                        meanp2.append(np.mean(chunk['powerl2']))
                        meanp3.append(np.mean(chunk['powerl3']))

                        meancp123.append(np.mean(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        meancp1.append(np.mean(chunk['currentl1']))
                        meancp2.append(np.mean(chunk['currentl2']))
                        meancp3.append(np.mean(chunk['currentl3']))

                        meanvp123.append(np.mean(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        meanvp1.append(np.mean(chunk['voltagel1']))
                        meanvp2.append(np.mean(chunk['voltagel2']))
                        meanvp3.append(np.mean(chunk['voltagel3']))

                        meancvp123.append(np.mean(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        meancvp1.append(np.mean(chunk['phaseanglecurrentvoltagel1']))
                        meancvp2.append(np.mean(chunk['phaseanglecurrentvoltagel2']))
                        meancvp3.append(np.mean(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate standard deviations
                        stdp123.append(np.std(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        stdp1.append(np.std(chunk['powerl1']))
                        stdp2.append(np.std(chunk['powerl2']))
                        stdp3.append(np.std(chunk['powerl3']))

                        stdcp123.append(np.std(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        stdcp1.append(np.std(chunk['currentl1']))
                        stdcp2.append(np.std(chunk['currentl2']))
                        stdcp3.append(np.std(chunk['currentl3']))

                        stdvp123.append(np.std(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        stdvp1.append(np.std(chunk['voltagel1']))
                        stdvp2.append(np.std(chunk['voltagel2']))
                        stdvp3.append(np.std(chunk['voltagel3']))

                        stdcvp123.append(np.std(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        stdcvp1.append(np.std(chunk['phaseanglecurrentvoltagel1']))
                        stdcvp2.append(np.std(chunk['phaseanglecurrentvoltagel2']))
                        stdcvp3.append(np.std(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate SADs
                        sadp123.append(self.calc_sad(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        sadp1.append(self.calc_sad(chunk['powerl1']))
                        sadp2.append(self.calc_sad(chunk['powerl2']))
                        sadp3.append(self.calc_sad(chunk['powerl3']))

                        sadcp123.append(self.calc_sad(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        sadcp1.append(self.calc_sad(chunk['currentl1']))
                        sadcp2.append(self.calc_sad(chunk['currentl2']))
                        sadcp3.append(self.calc_sad(chunk['currentl3']))

                        sadvp123.append(self.calc_sad(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        sadvp1.append(self.calc_sad(chunk['voltagel1']))
                        sadvp2.append(self.calc_sad(chunk['voltagel2']))
                        sadvp3.append(self.calc_sad(chunk['voltagel3']))

                        sadcvp123.append(self.calc_sad(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        sadcvp1.append(self.calc_sad(chunk['phaseanglecurrentvoltagel1']))
                        sadcvp2.append(self.calc_sad(chunk['phaseanglecurrentvoltagel2']))
                        sadcvp3.append(self.calc_sad(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate max values
                        maxp123.append(np.amax(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        maxp1.append(np.amax(chunk['powerl1']))
                        maxp2.append(np.amax(chunk['powerl2']))
                        maxp3.append(np.amax(chunk['powerl3']))

                        maxcp123.append(np.amax(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        maxcp1.append(np.amax(chunk['currentl1']))
                        maxcp2.append(np.amax(chunk['currentl2']))
                        maxcp3.append(np.amax(chunk['currentl3']))

                        maxvp123.append(np.amax(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        maxvp1.append(np.amax(chunk['voltagel1']))
                        maxvp2.append(np.amax(chunk['voltagel2']))
                        maxvp3.append(np.amax(chunk['voltagel3']))

                        maxcvp123.append(np.amax(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        maxcvp1.append(np.amax(chunk['phaseanglecurrentvoltagel1']))
                        maxcvp2.append(np.amax(chunk['phaseanglecurrentvoltagel2']))
                        maxcvp3.append(np.amax(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate min values
                        minp123.append(np.amin(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        minp1.append(np.amin(chunk['power']))

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

    def run_model(self):
        if not hasattr(self, 'custom_data') or not hasattr(self, 'occupancy_data'):
            self.show_error_message("Please load all required data files before running the model.")
            return
            
        try:
            except Exception as e:  print(f'Error during data concatenation: {e}')
               
            self.custom_data.reset_index(drop=True, inplace=True)
            self.occupancy_data.reset_index(drop=True, inplace=True)
            featuredata = pd.concat([self.custom_data, self.occupancy_data], axis=1)
            
            output_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'New folder', 'mergedfile3.csv')
            featuredata.to_csv(output_file, index=False)
            
            X = featuredata.drop(columns=['occupancy', 'date'], errors='ignore')
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            y = pd.to_numeric(featuredata['occupancy'], errors='coerce').fillna(0)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            algorithm = self.comboBox_1.currentText().strip()
            
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
            
            model.fit(X_train, y_train)
            
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

    def calculate_chunk_features(self, chunk, feature_lists):
        """Calculate all features for a given chunk of data"""
        # Means
        feature_lists['meanp123'].append(np.mean(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['meanp1'].append(np.mean(chunk['powerl1']))
        feature_lists['meanp2'].append(np.mean(chunk['powerl2']))
        feature_lists['meanp3'].append(np.mean(chunk['powerl3']))

        feature_lists['meancp123'].append(np.mean(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
        feature_lists['meancp1'].append(np.mean(chunk['currentl1']))
        feature_lists['meancp2'].append(np.mean(chunk['currentl2']))
        feature_lists['meancp3'].append(np.mean(chunk['currentl3']))

        feature_lists['meanvp123'].append(np.mean(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
        feature_lists['meanvp1'].append(np.mean(chunk['voltagel1']))
        feature_lists['meanvp2'].append(np.mean(chunk['voltagel2']))
        feature_lists['meanvp3'].append(np.mean(chunk['voltagel3']))

        feature_lists['meancvp123'].append(np.mean(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
        feature_lists['meancvp1'].append(np.mean(chunk['phaseanglecurrentvoltagel1']))
        feature_lists['meancvp2'].append(np.mean(chunk['phaseanglecurrentvoltagel2']))
        feature_lists['meancvp3'].append(np.mean(chunk['phaseanglecurrentvoltagel3']))

        # Standard deviations
        feature_lists['stdp123'].append(np.std(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['stdp1'].append(np.std(chunk['powerl1']))
        feature_lists['stdp2'].append(np.std(chunk['powerl2']))
        feature_lists['stdp3'].append(np.std(chunk['powerl3']))

        feature_lists['stdcp123'].append(np.std(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
        feature_lists['stdcp1'].append(np.std(chunk['currentl1']))
        feature_lists['stdcp2'].append(np.std(chunk['currentl2']))
        feature_lists['stdcp3'].append(np.std(chunk['currentl3']))

        feature_lists['stdvp123'].append(np.std(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
        feature_lists['stdvp1'].append(np.std(chunk['voltagel1']))
        feature_lists['stdvp2'].append(np.std(chunk['voltagel2']))
        feature_lists['stdvp3'].append(np.std(chunk['voltagel3']))

        feature_lists['stdcvp123'].append(np.std(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
        feature_lists['stdcvp1'].append(np.std(chunk['phaseanglecurrentvoltagel1']))
        feature_lists['stdcvp2'].append(np.std(chunk['phaseanglecurrentvoltagel2']))
        feature_lists['stdcvp3'].append(np.std(chunk['phaseanglecurrentvoltagel3']))

        # SAD calculations
        feature_lists['sadp123'].append(self.calc_sad(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['sadp1'].append(self.calc_sad(chunk['powerl1']))
        feature_lists['sadp2'].append(self.calc_sad(chunk['powerl2']))
        feature_lists['sadp3'].append(self.calc_sad(chunk['powerl3']))

        feature_lists['sadcp123'].append(self.calc_sad(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
        feature_lists['sadcp1'].append(self.calc_sad(chunk['currentl1']))
        feature_lists['sadcp2'].append(self.calc_sad(chunk['currentl2']))
        feature_lists['sadcp3'].append(self.calc_sad(chunk['currentl3']))

        feature_lists['sadvp123'].append(self.calc_sad(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
        feature_lists['sadvp1'].append(self.calc_sad(chunk['voltagel1']))
        feature_lists['sadvp2'].append(self.calc_sad(chunk['voltagel2']))
        feature_lists['sadvp3'].append(self.calc_sad(chunk['voltagel3']))

        feature_lists['sadcvp123'].append(self.calc_sad(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
        feature_lists['sadcvp1'].append(self.calc_sad(chunk['phaseanglecurrentvoltagel1']))
        feature_lists['sadcvp2'].append(self.calc_sad(chunk['phaseanglecurrentvoltagel2']))
        feature_lists['sadcvp3'].append(self.calc_sad(chunk['phaseanglecurrentvoltagel3']))

        # Max values
        feature_lists['maxp123'].append(np.max(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['maxp1'].append(np.max(chunk['powerl1']))
        feature_lists['maxp2'].append(np.max(chunk['powerl2']))
        feature_lists['maxp3'].append(np.max(chunk['powerl3']))

        feature_lists['maxcp123'].append(np.max(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
        feature_lists['maxcp1'].append(np.max(chunk['currentl1']))
        feature_lists['maxcp2'].append(np.max(chunk['currentl2']))
        feature_lists['maxcp3'].append(np.max(chunk['currentl3']))

        feature_lists['maxvp123'].append(np.max(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
        feature_lists['maxvp1'].append(np.max(chunk['voltagel1']))
        feature_lists['maxvp2'].append(np.max(chunk['voltagel2']))
        feature_lists['maxvp3'].append(np.max(chunk['voltagel3']))

        feature_lists['maxcvp123'].append(np.max(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
        feature_lists['maxcvp1'].append(np.max(chunk['phaseanglecurrentvoltagel1']))
        feature_lists['maxcvp2'].append(np.max(chunk['phaseanglecurrentvoltagel2']))
        feature_lists['maxcvp3'].append(np.max(chunk['phaseanglecurrentvoltagel3']))

        # Min values
        feature_lists['minp123'].append(np.min(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['minp1'].append(np.min(chunk['powerl1']))
        feature_lists['minp2'].append(np.min(chunk['powerl2']))
        feature_lists['minp3'].append(np.min(chunk['powerl3']))

        feature_lists['mincp123'].append(np.min(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
        feature_lists['mincp1'].append(np.min(chunk['currentl1']))
        feature_lists['mincp2'].append(np.min(chunk['currentl2']))
        feature_lists['mincp3'].append(np.min(chunk['currentl3']))

        feature_lists['minvp123'].append(np.min(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
        feature_lists['minvp1'].append(np.min(chunk['voltagel1']))
        feature_lists['minvp2'].append(np.min(chunk['voltagel2']))
        feature_lists['minvp3'].append(np.min(chunk['voltagel3']))

        feature_lists['mincvp123'].append(np.min(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
        feature_lists['mincvp1'].append(np.min(chunk['phaseanglecurrentvoltagel1']))
        feature_lists['mincvp2'].append(np.min(chunk['phaseanglecurrentvoltagel2']))
        feature_lists['mincvp3'].append(np.min(chunk['phaseanglecurrentvoltagel3']))

        # Ranges
        feature_lists['rangep123'].append(max([np.max(chunk['powerl1']), np.max(chunk['powerl2']), np.max(chunk['powerl3'])])
                                        - min([np.min(chunk['powerl1']), np.min(chunk['powerl2']), np.min(chunk['powerl3'])]))
        feature_lists['rangep1'].append(np.max(chunk['powerl1']) - np.min(chunk['powerl1']))
        feature_lists['rangep2'].append(np.max(chunk['powerl2']) - np.min(chunk['powerl2']))
        feature_lists['rangep3'].append(np.max(chunk['powerl3']) - np.min(chunk['powerl3']))

        # Add similar calculations for other range features...

        # Correlations
        feature_lists['corrp123'].append((chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']).corr((chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']).shift(1)))
        feature_lists['corrp1'].append(chunk['powerl1'].corr(chunk['powerl1'].shift(1)))
        feature_lists['corrp2'].append(chunk['powerl2'].corr(chunk['powerl2'].shift(1)))
        feature_lists['corrp3'].append(chunk['powerl3'].corr(chunk['powerl3'].shift(1)))

        # Add similar calculations for other correlation features...

        # On/Off detection
        feature_lists['onoffp123'].append(self.detect_onoff(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
        feature_lists['onoffp1'].append(self.detect_onoff(chunk['powerl1']))
        feature_lists['onoffp2'].append(self.detect_onoff(chunk['powerl2']))
        feature_lists['onoffp3'].append(self.detect_onoff(chunk['powerl3']))

        # Add similar calculations for other on/off features...

        # Time period
        feature_lists['ptime'].append(list(range(1, 97)))

    def create_features_dataframe(self, feature_lists):
        """Create DataFrame from calculated features"""
        columns = [
            'minp1', 'minp2', 'minp3', 'minp123',
            'mincp1', 'mincp2', 'mincp3', 'mincp123',
            'minvp1', 'minvp2', 'minvp3', 'minvp123',
            'mincvp1', 'mincvp2', 'mincvp3', 'mincvp123',
            'maxp1', 'maxp2', 'maxp3', 'maxp123',
            'maxcp1', 'maxcp2', 'maxcp3', 'maxcp123',
            'maxvp1', 'maxvp2', 'maxvp3', 'maxvp123',
            'maxcvp1', 'maxcvp2', 'maxcvp3', 'maxcvp123',
            'meanp1', 'meanp2', 'meanp3', 'meanp123',
            'meancp1', 'meancp2', 'meancp3', 'meancp123',
            'meanvp1', 'meanvp2', 'meanvp3', 'meanvp123',
            'meancvp1', 'meancvp2', 'meancvp3', 'meancvp123',
            'stdp1', 'stdp2', 'stdp3', 'stdp123',
            'stdcp1', 'stdcp2', 'stdcp3', 'stdcp123',
            'stdvp1', 'stdvp2', 'stdvp3', 'stdvp123',
            'stdcvp1', 'stdcvp2', 'stdcvp3', 'stdcvp123',
            'sadp1', 'sadp2', 'sadp3', 'sadp123',
            'sadcp1', 'sadcp2', 'sadcp3', 'sadcp123',
            'sadvp1', 'sadvp2', 'sadvp3', 'sadvp123',
            'sadcvp1', 'sadcvp2', 'sadcvp3', 'sadcvp123',
            'corrp1', 'corrp2', 'corrp3', 'corrp123',
            'corrcp1', 'corrcp2', 'corrcp3', 'corrcp123',
            'corrvp1', 'corrvp2', 'corrvp3', 'corrvp123',
            'corrcvp1', 'corrcvp2', 'corrcvp3', 'corrcvp123',
            'onoffp1', 'onoffp2', 'onoffp3', 'onoffp123',
            'onoffcp1', 'onoffcp2', 'onoffcp3', 'onoffcp123',
            'onoffvp1', 'onoffvp2', 'onoffvp3', 'onoffvp123',
            'onoffcvp1', 'onoffcvp2', 'onoffcvp3', 'onoffcvp123',
            'rangep1', 'rangep2', 'rangep3', 'rangep123',
            'rangecp1', 'rangecp2', 'rangecp3', 'rangecp123',
            'rangevp1', 'rangevp2', 'rangevp3', 'rangevp123',
            'rangecvp1', 'rangecvp2', 'rangecvp3', 'rangecvp123',
            'ptime'
        ]
        
        # Create DataFrame with pre-defined columns
        df = pd.DataFrame(columns=columns)
        
        # Add data from feature lists
        for col in columns:
            if col in feature_lists:
                df[col] = feature_lists[col]
                
        return df

    def setupUi(self, sensor_fusion):
        # Main window setup
        sensor_fusion.setObjectName("sensor_fusion")
        sensor_fusion.resize(800, 600)  # Increased width for better layout
        
        # Set the window title
        sensor_fusion.setWindowTitle("Sensor Fusion Analysis")
        
        # Add stylesheet
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
            QLabel {
                color: black;
                padding: 2px;
            }
            QRadioButton {
                color: black;
                spacing: 5px;
            }
        """)

        # Main layout
        self.main_layout = QVBoxLayout(sensor_fusion)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        self.title_label = QLabel("Sensor Fusion Analysis")
        self.title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            color: #2a2a2a;
        """)
        self.main_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Input Section
        self.input_group = QGroupBox("Data Input")
        self.input_layout = QGridLayout()
        self.input_layout.setSpacing(10)

        # Setup input rows
        self.setup_input_row(0, "Temperature", "radioButton_1", "pushButton_4_csv1", "label_file1_status")
        self.setup_input_row(1, "CO2", "radioButton_2", "pushButton_5_csv2", "label_file2_status")
        self.setup_input_row(2, "Humidity", "radioButton_3", "pushButton_6_csv3", "label_file3_status")
        self.setup_input_row(3, "Custom Data", "radioButton_raw_data", "pushButton_raw_data", "label_raw_data_status")
        self.setup_input_row(4, "Occupancy", "radioButton_raw_data1", "pushButton_raw_data1", "label_raw_data_status1")

        self.input_group.setLayout(self.input_layout)
        self.main_layout.addWidget(self.input_group)

        # Model Configuration Section
        self.model_group = QGroupBox("Model Configuration")
        self.model_layout = QGridLayout()

        # Algorithm Selection
        self.label_algorithm = QLabel("Algorithm:")
        self.comboBox_1 = QComboBox()
        self.comboBox_1.addItems(["Random Forest", "SVM", "Naive Bayes", "Decision Tree"])
        self.model_layout.addWidget(self.label_algorithm, 0, 0)
        self.model_layout.addWidget(self.comboBox_1, 0, 1)

        # Metric Selection
        self.label_metric = QLabel("Metric:")
        self.comboBox_2 = QComboBox()
        self.comboBox_2.addItems(["Accuracy", "Confusion matrix", "F1 Score"])
        self.model_layout.addWidget(self.label_metric, 1, 0)
        self.model_layout.addWidget(self.comboBox_2, 1, 1)

        self.model_group.setLayout(self.model_layout)
        self.main_layout.addWidget(self.model_group)

        # Results Section
        self.results_group = QGroupBox("Results")
        self.results_layout = QVBoxLayout()

        # Run button and progress bar
        self.run_container = QHBoxLayout()
        self.pushButton_run = QPushButton("Run Model")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.run_container.addWidget(self.pushButton_run)
        self.run_container.addWidget(self.progress_bar)
        self.results_layout.addLayout(self.run_container)

        # Results display
        self.final_result = QLabel("Results will appear here")
        self.final_result.setStyleSheet("""
            background-color: white;
            padding: 10px;
            border-radius: 4px;
            font-weight: bold;
        """)
        self.results_layout.addWidget(self.final_result)

        self.results_group.setLayout(self.results_layout)
        self.main_layout.addWidget(self.results_group)

        # Connect signals
        self.setup_connections()

    def setup_input_row(self, row, label_text, radio_name, button_name, status_name):
        label = QLabel(label_text)
        radio = QRadioButton()
        button = QPushButton(f"Load {label_text}")
        status = QLabel("No file loaded")
        
        setattr(self, radio_name, radio)
        setattr(self, button_name, button)
        setattr(self, status_name, status)
        
        self.input_layout.addWidget(label, row, 0)
        self.input_layout.addWidget(radio, row, 1)
        self.input_layout.addWidget(button, row, 2)
        self.input_layout.addWidget(status, row, 3)

    def setup_connections(self):
        self.pushButton_4_csv1.clicked.connect(self.load_csv1)
        self.pushButton_5_csv2.clicked.connect(self.load_csv2)
        self.pushButton_6_csv3.clicked.connect(self.load_csv3)
        self.pushButton_raw_data.clicked.connect(self.load_raw_data)
        self.pushButton_raw_data1.clicked.connect(self.target)
        self.pushButton_run.clicked.connect(self.run_model)

    def show_success_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(message)
        msg.setWindowTitle("Success")
        msg.exec()

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec()

class SensorFusionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_sensor_fusion()
        self.ui.setupUi(self)

if __name__ == "__main__":
    import sys

                        minp1.append(np.amin(chunk['powerl1']))
                        minp2.append(np.amin(chunk['powerl2']))
                        minp3.append(np.amin(chunk['powerl3']))

                        mincp123.append(np.amin(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        mincp1.append(np.amin(chunk['currentl1']))
                        mincp2.append(np.amin(chunk['currentl2']))
                        mincp3.append(np.amin(chunk['currentl3']))

                        minvp123.append(np.amin(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        minvp1.append(np.amin(chunk['voltagel1']))
                        minvp2.append(np.amin(chunk['voltagel2']))
                        minvp3.append(np.amin(chunk['voltagel3']))

                        mincvp123.append(np.amin(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        mincvp1.append(np.amin(chunk['phaseanglecurrentvoltagel1']))
                        mincvp2.append(np.amin(chunk['phaseanglecurrentvoltagel2']))
                        mincvp3.append(np.amin(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate ranges
                        rangep123.append(np.amax([np.amax(chunk['powerl1']), np.amax(chunk['powerl2']), np.amax(chunk['powerl3'])]) - 
                                    np.amin([np.amin(chunk['powerl1']), np.amin(chunk['powerl2']), np.amin(chunk['powerl3'])]))
                        rangep1.append(np.amax(chunk['powerl1']) - np.amin(chunk['powerl1']))
                        rangep2.append(np.amax(chunk['powerl2']) - np.amin(chunk['powerl2']))
                        rangep3.append(np.amax(chunk['powerl3']) - np.amin(chunk['powerl3']))

                        rangecp123.append(np.amax([np.amax(chunk['currentl1']), np.amax(chunk['currentl2']), np.amax(chunk['currentl3'])]) -
                                    np.amin([np.amin(chunk['currentl1']), np.amin(chunk['currentl2']), np.amin(chunk['currentl3'])]))
                        rangecp1.append(np.amax(chunk['currentl1']) - np.amin(chunk['currentl1']))
                        rangecp2.append(np.amax(chunk['currentl2']) - np.amin(chunk['currentl2']))
                        rangecp3.append(np.amax(chunk['currentl3']) - np.amin(chunk['currentl3']))

                        rangevp123.append(np.amax([np.amax(chunk['voltagel1']), np.amax(chunk['voltagel2']), np.amax(chunk['voltagel3'])]) -
                                    np.amin([np.amin(chunk['voltagel1']), np.amin(chunk['voltagel2']), np.amin(chunk['voltagel3'])]))
                        rangevp1.append(np.amax(chunk['voltagel1']) - np.amin(chunk['voltagel1']))
                        rangevp2.append(np.amax(chunk['voltagel2']) - np.amin(chunk['voltagel2']))
                        rangevp3.append(np.amax(chunk['voltagel3']) - np.amin(chunk['voltagel3']))

                        rangecvp123.append(np.amax([np.amax(chunk['phaseanglecurrentvoltagel1']), 
                                                np.amax(chunk['phaseanglecurrentvoltagel2']), 
                                                np.amax(chunk['phaseanglecurrentvoltagel3'])]) -
                                        np.amin([np.amin(chunk['phaseanglecurrentvoltagel1']), 
                                            np.amin(chunk['phaseanglecurrentvoltagel2']), 
                                            np.amin(chunk['phaseanglecurrentvoltagel3'])]))
                        rangecvp1.append(np.amax(chunk['phaseanglecurrentvoltagel1']) - np.amin(chunk['phaseanglecurrentvoltagel1']))
                        rangecvp2.append(np.amax(chunk['phaseanglecurrentvoltagel2']) - np.amin(chunk['phaseanglecurrentvoltagel2']))
                        rangecvp3.append(np.amax(chunk['phaseanglecurrentvoltagel3']) - np.amin(chunk['phaseanglecurrentvoltagel3']))

                        # Calculate correlations
                        corrp123.append((chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']).corr((chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']).shift(1)))
                        corrp1.append(chunk['powerl1'].corr(chunk['powerl1'].shift(1)))
                        corrp2.append(chunk['powerl2'].corr(chunk['powerl2'].shift(1)))
                        corrp3.append(chunk['powerl3'].corr(chunk['powerl3'].shift(1)))

                        corrcp123.append((chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']).corr((chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']).shift(1)))
                        corrcp1.append(chunk['currentl1'].corr(chunk['currentl1'].shift(1)))
                        corrcp2.append(chunk['currentl2'].corr(chunk['currentl2'].shift(1)))
                        corrcp3.append(chunk['currentl3'].corr(chunk['currentl3'].shift(1)))

                        corrvp123.append((chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']).corr((chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']).shift(1)))
                        corrvp1.append(chunk['voltagel1'].corr(chunk['voltagel1'].shift(1)))
                        corrvp2.append(chunk['voltagel2'].corr(chunk['voltagel2'].shift(1)))
                        corrvp3.append(chunk['voltagel3'].corr(chunk['voltagel3'].shift(1)))

                        corrcvp123.append((chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']).corr(
                            (chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']).shift(1)))
                        corrcvp1.append(chunk['phaseanglecurrentvoltagel1'].corr(chunk['phaseanglecurrentvoltagel1'].shift(1)))
                        corrcvp2.append(chunk['phaseanglecurrentvoltagel2'].corr(chunk['phaseanglecurrentvoltagel2'].shift(1)))
                        corrcvp3.append(chunk['phaseanglecurrentvoltagel3'].corr(chunk['phaseanglecurrentvoltagel3'].shift(1)))

                        # Calculate on/off events
                        onoffp123.append(self.detect_onoff(chunk['powerl1'] + chunk['powerl2'] + chunk['powerl3']))
                        onoffp1.append(self.detect_onoff(chunk['powerl1']))
                        onoffp2.append(self.detect_onoff(chunk['powerl2']))
                        onoffp3.append(self.detect_onoff(chunk['powerl3']))

                        onoffcp123.append(self.detect_onoff(chunk['currentl1'] + chunk['currentl2'] + chunk['currentl3']))
                        onoffcp1.append(self.detect_onoff(chunk['currentl1']))
                        onoffcp2.append(self.detect_onoff(chunk['currentl2']))
                        onoffcp3.append(self.detect_onoff(chunk['currentl3']))

                        onoffvp123.append(self.detect_onoff(chunk['voltagel1'] + chunk['voltagel2'] + chunk['voltagel3']))
                        onoffvp1.append(self.detect_onoff(chunk['voltagel1']))
                        onoffvp2.append(self.detect_onoff(chunk['voltagel2']))
                        onoffvp3.append(self.detect_onoff(chunk['voltagel3']))

                        onoffcvp123.append(self.detect_onoff(chunk['phaseanglecurrentvoltagel1'] + chunk['phaseanglecurrentvoltagel2'] + chunk['phaseanglecurrentvoltagel3']))
                        onoffcvp1.append(self.detect_onoff(chunk['phaseanglecurrentvoltagel1']))
                        onoffcvp2.append(self.detect_onoff(chunk['phaseanglecurrentvoltagel2']))
                        onoffcvp3.append(self.detect_onoff(chunk['phaseanglecurrentvoltagel3']))

                    ptime = list(range(1, 97))
                    
                    # Create DataFrame with all features
                    data1 = pd.DataFrame(np.column_stack((
                        minp1, minp2, minp3, minp123, mincp1, mincp2, mincp3, mincp123,
                        minvp1, minvp2, minvp3, minvp123, mincvp1, mincvp2, mincvp3, mincvp123,
                        maxp1, maxp2, maxp3, maxp123, maxcp1, maxcp2, maxcp3, maxcp123,
                        maxvp1, maxvp2, maxvp3, maxvp123, maxcvp1, maxcvp2, maxcvp3, maxcvp123,
                        meanp1, meanp2, meanp3, meanp123, meancp1, meancp2, meancp3, meancp123,
                        meanvp1, meanvp2, meanvp3, meanvp123, meancvp1, meancvp2, meancvp3, meancvp123,
                        stdp1, stdp2, stdp3, stdp123, stdcp1, stdcp2, stdcp3, stdcp123,
                        stdvp1, stdvp2, stdvp3, stdvp123, stdcvp1, stdcvp2, stdcvp3, stdcvp123,
                        sadp1, sadp2, sadp3, sadp123, sadcp1, sadcp2, sadcp3, sadcp123,
                        sadvp1, sadvp2, sadvp3, sadvp123, sadcvp1, sadcvp2, sadcvp3, sadcvp123,
                        corrp1, corrp2, corrp3, corrp123, corrcp1, corrcp2, corrcp3, corrcp123,
                        corrvp1, corrvp2, corrvp3, corrvp123, corrcvp1, corrcvp2, corrcvp3, corrcvp123,
                        onoffp1, onoffp2, onoffp3, onoffp123, onoffcp1, onoffcp2, onoffcp3, onoffcp123,
                        onoffvp1, onoffvp2, onoffvp3, onoffvp123, onoffcvp1, onoffcvp2, onoffcvp3, onoffcvp123,
                        rangep1, rangep2, rangep3, rangep123, rangecp1, rangecp2, rangecp3, rangecp123,
                        rangevp1, rangevp2, rangevp3, rangevp123, rangecvp1, rangecvp2, rangecvp3, rangecvp123,
                        ptime)),
                        columns=['minp1', 'minp2', 'minp3', 'minp123', 'mincp1', 'mincp2', 'mincp3', 'mincp123',
                                'minvp1', 'minvp2', 'minvp3', 'minvp123', 'mincvp1', 'mincvp2', 'mincvp3', 'mincvp123',
                                'maxp1', 'maxp2', 'maxp3', 'maxp123', 'maxcp1', 'maxcp2', 'maxcp3', 'maxcp123',
                                'maxvp1', 'maxvp2', 'maxvp3', 'maxpv123', 'maxcvp1', 'maxcvp2', 'maxcvp3', 'maxcvp123',
                                'meanp1', 'meanp2', 'meanp3', 'meanp123', 'meancp1', 'meancp2', 'meancp3', 'meancp123',
                                'meanvp1', 'meanvp2', 'meanvp3', 'meanvp123', 'meancvp1', 'meancvp2', 'meancvp3', 'meancvp123',
                                'stdp1', 'stdp2', 'stdp3', 'stdp123', 'stdcp1', 'stdcp2', 'stdcp3', 'stdcp123',
                                'stdvp1', 'stdvp2', 'stdvp3', 'stdvp123', 'stdcvp1', 'stdcvp2', 'stdcvp3', 'stdcvp123',
                                'sadp1', 'sadp2', 'sadp3', 'sadp123', 'sadcp1', 'sadcp2', 'sadcp3', 'sadcp123',
                                'sadvp1', 'sadvp2', 'sadvp3', 'sadvp123', 'sadcvp1', 'sadcvp2', 'sadcvp3', 'sadcvp123',
                                'corrp1', 'corrp2', 'corrp3', 'corrp123', 'corrcp1', 'corrcp2', 'corrcp3', 'corrcp123',
                                'corrvp1', 'corrvp2', 'corrvp3', 'corrvp123', 'corrcvp1', 'corrcvp2', 'corrcvp3', 'corrcvp123',
                                'onoffp1', 'onoffp2', 'onoffp3', 'onoffp123', 'onoffcp1', 'onoffcp2', 'onoffcp3', 'onoffcp123',
                                'onoffvp1', 'onoffvp2', 'onoffvp3', 'onoffcp123', 'onoffvp1', 'onoffvp2', 'onoffvp3']
    app = QtWidgets.QApplication(sys.argv)
    window = SensorFusionApp()
    window.show()
    sys.exit(app.exec())