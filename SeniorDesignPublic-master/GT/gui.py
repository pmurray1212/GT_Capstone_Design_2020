import os
from PyQt5.QtCore import QDateTime, Qt, QTimer, QFileInfo, QModelIndex, QAbstractTableModel, QVariant
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QMenuBar, QFileDialog, QListView, QTreeView, 
        QAbstractItemView, QFrame, QTableView, QTableWidgetItem, QHeaderView)
from PyQt5.QtGui import QPixmap, QStandardItemModel, QStandardItem
from pathlib import Path

import csv

import time

import support.dev.optimizer as optimizer

import pandas as pd
from pprint import pprint


import support.dev.datastore as ds
import support.dev.regstore as rs
import support.dev.restaurant as restaurant
import pickle
from math import *
import random


class LoginPage(QDialog):
    def __init__(self):
        super(LoginPage, self).__init__()
        self.setWindowTitle("Buffalo Wild Wings Beer Selection Tool")
        self.setFixedSize(900,450)

        #GT DISCLAIMERS
        disclaimer1 = QLabel("This project has been created as part of a student design project at Georgia Institute of Technology")
        disclaimer2 = QLabel("This project is subject to a Non-Disclosure Agreement between Inspire Brands, Inc. and Georgia Tech gtNDA-5346.")
        disclaimer1.setAlignment(Qt.AlignCenter)
        disclaimer2.setAlignment(Qt.AlignCenter)

        #BWW LOGO
        img = QLabel()
        dir_path = Path.cwd()
        imgs = None
        for child in dir_path.parent.iterdir():
            if child.name == "GT":
                imgs = dir_path.parent.joinpath(child.name).joinpath("support/other/BWW.svg")
        img.setPixmap(QPixmap(QPixmap(str(imgs))))
        img.setFixedSize(800,320)
        img.setAlignment(Qt.AlignCenter)

        #SPACER
        spacer = QLabel(" ")
        spacer.setFixedSize(800,700)

        #HORIZONTAL BUTTONS
        self.continueB = QPushButton("Continue")
        self.continueB.setFixedWidth(200)
        self.continueB.clicked.connect(self.continueClicked)  
        self.helpB = QPushButton("Help")
        self.helpB.clicked.connect(self.helpClicked)
        self.helpB.setFixedWidth(200)
        hbox_layout = QHBoxLayout()
        hbox_layout.addStretch(1)
        hbox_layout.addWidget(self.continueB)
        hbox_layout.addStretch(1)
        hbox_layout.addWidget(self.helpB)
        hbox_layout.addStretch(1)

        #VBOX LAYOUT
        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(img)
        vbox_layout.addWidget(spacer)
        vbox_layout.addLayout(hbox_layout)
        vbox_layout.addWidget(disclaimer1)
        vbox_layout.addWidget(disclaimer2)
        
        self.setLayout(vbox_layout)
        self.help = WidgetPage(self)
        self.nav = MainWindow(self)
    
            
    def helpClicked(self):
        self.help.show()

    def continueClicked(self):
        self.close()
        if self.help.isHidden == False:
            self.help.show()
        self.nav.show()
    
#HELP PAGE
class HelpPage(QDialog):
    def __init__(self, parent = None):
        super(HelpPage, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("User Guide")

#TEMP PAGE ACTING AS HELP PAGE
class WidgetPage(QDialog):
    def __init__(self, parent = None):
        super(WidgetPage, self).__init__(parent)

        self.parent = parent

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

        disableWidgetsCheckBox = QCheckBox("&Disable widgets")

        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()
        self.createProgressBar()

        styleComboBox.activated[str].connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        disableWidgetsCheckBox.toggled.connect(self.topLeftGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.topRightGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomLeftTabWidget.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomRightGroupBox.setDisabled)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)
        topLayout.addWidget(disableWidgetsCheckBox)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Styles")
        #self.changeStyle('Windows')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Group 1")

        radioButton1 = QRadioButton("Radio button 1")
        radioButton2 = QRadioButton("Radio button 2")
        radioButton3 = QRadioButton("Radio button 3")
        radioButton1.setChecked(True)

        checkBox = QCheckBox("Tri-state check box")
        checkBox.setTristate(True)
        checkBox.setCheckState(Qt.PartiallyChecked)

        layout = QVBoxLayout()
        layout.addWidget(radioButton1)
        layout.addWidget(radioButton2)
        layout.addWidget(radioButton3)
        layout.addWidget(checkBox)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)    

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Group 2")

        defaultPushButton = QPushButton("Default Push Button")
        defaultPushButton.setDefault(True)

        togglePushButton = QPushButton("Toggle Push Button")
        togglePushButton.setCheckable(True)
        togglePushButton.setChecked(True)

        flatPushButton = QPushButton("Flat Push Button")
        flatPushButton.setFlat(True)

        layout = QVBoxLayout()
        layout.addWidget(defaultPushButton)
        layout.addWidget(togglePushButton)
        layout.addWidget(flatPushButton)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Ignored)

        tab1 = QWidget()
        tableWidget = QTableWidget(10, 10)

        tab1hbox = QHBoxLayout()
        tab1hbox.setContentsMargins(5, 5, 5, 5)
        tab1hbox.addWidget(tableWidget)
        tab1.setLayout(tab1hbox)

        tab2 = QWidget()
        textEdit = QTextEdit()

        textEdit.setPlainText("Twinkle, twinkle, little star,\n"
                              "How I wonder what you are.\n" 
                              "Up above the world so high,\n"
                              "Like a diamond in the sky.\n"
                              "Twinkle, twinkle, little star,\n" 
                              "How I wonder what you are!\n")

        tab2hbox = QHBoxLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)
        tab2hbox.addWidget(textEdit)
        tab2.setLayout(tab2hbox)

        self.bottomLeftTabWidget.addTab(tab1, "&Table")
        self.bottomLeftTabWidget.addTab(tab2, "Text &Edit")

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Group 3")
        self.bottomRightGroupBox.setCheckable(True)
        self.bottomRightGroupBox.setChecked(True)

        lineEdit = QLineEdit('s3cRe7')
        lineEdit.setEchoMode(QLineEdit.Password)

        spinBox = QSpinBox(self.bottomRightGroupBox)
        spinBox.setValue(50)

        dateTimeEdit = QDateTimeEdit(self.bottomRightGroupBox)
        dateTimeEdit.setDateTime(QDateTime.currentDateTime())

        slider = QSlider(Qt.Horizontal, self.bottomRightGroupBox)
        slider.setValue(40)

        scrollBar = QScrollBar(Qt.Horizontal, self.bottomRightGroupBox)
        scrollBar.setValue(60)

        dial = QDial(self.bottomRightGroupBox)
        dial.setValue(30)
        dial.setNotchesVisible(True)

        layout = QGridLayout()
        layout.addWidget(lineEdit, 0, 0, 1, 2)
        layout.addWidget(spinBox, 1, 0, 1, 2)
        layout.addWidget(dateTimeEdit, 2, 0, 1, 2)
        layout.addWidget(slider, 3, 0)
        layout.addWidget(scrollBar, 4, 0)
        layout.addWidget(dial, 3, 1, 2, 1)
        layout.setRowStretch(5, 1)
        self.bottomRightGroupBox.setLayout(layout)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(1000)


# QABSTRACTTABLEMODEL
# https://doc-snapshots.qt.io/qtforpython-dev/tutorials/datavisualize/add_tableview.html
class SimpleTableModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self, None)
        self.data = data
        #self.headers = [str(k) for k, v in data[0].items()]
        self.headers = [str(k) for k in data[0]]
        #self.rows = [[str(v) for k, v in record.items()] for record in data]
        self.rows = [[str(v) for v in row]for row in data[1:]]
    def rowCount(self, parent):
        return len(self.rows)
    def columnCount(self, parent):
        return len(self.headers)
    def data(self, index, role):
        if (not index.isValid()) or (role != Qt.DisplayRole):
            return QVariant()
        else:
            return QVariant(self.rows[index.row()][index.column()])
    def row(self, index):
        return self.data[index]
    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return QVariant()
        elif orientation == Qt.Vertical:
            return section + 1
        else:
            return self.headers[section]







#MAIN WINDOW AFTER START WINDOW
class MainWindow(QDialog):

    FILENAME, SIZE, PATH = range(3)

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("Beer Selection Tool Window")
        self.setMinimumWidth(1200)
        self.setMaximumWidth(1200)
        self.setMinimumHeight(700)
        self.setMaximumHeight(700)

        # Main Layout
        self.main_layout = QVBoxLayout()

        # Extra Info
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        self.months = {months[n]:n + 1 for n in range(12)}

        self.state_dict = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
        'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 
        'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 
        'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 
        'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 
        'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 
        'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}

        self.all_optimal_plans = {}

        """This stuff doesn't have to run when the app opens"""
        # Clean Restaurant Data
        #self.stores = self.gen_stores()
        self.stores = None

         # Beer Mapping
        #self.beer_map = self.gen_beer_map()
        self.beer_map = None

        # Clean Promo Data
        #self.promo = self.gen_promo()
        self.promo = None
        """This stuff doesn't have to run when the app open"""

        """
        DATA STORE
        """
        self.datastore = None

        """
        REG STORE
        """
        self.regstore = None

        """
        RESTAURANTS TO OPTIMIZE
        """
        self.restaurant_nums = None
        self.restaurants = [] 

        """
        RESULTS
        """
        self.results_dict = {}


        #List of all unique states from uploaded POS
        self.states = [" ","GA", "SC", "NC"]


        """
        **********
        Status Bar
        **********
        """
        # Status Bar Layout
        self.status_label = QLabel("Welcome to the Beer Selection Tool! Please click the '?' buttons for help in any of the steps")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Adds Status Bar
        self.gen_status_bar()
        """
        **************
        END STATUS BAR
        **************
        """


        # All File, Training, Optimization Manager Layouts
        self.manager_layout = QHBoxLayout()


        """
        *********
        FILE VIEW
        *********
        """
        # Help Button
        self.file_reload_button = QPushButton("Reload")
        self.file_reload_button.setMaximumWidth(70)
        self.file_reload_button.clicked.connect(self.reload_files)
        self.file_help_button = QPushButton("?")
        self.file_help_button.setMaximumWidth(40)
        self.file_help_button.clicked.connect(self.file_help)

        # Tables of Files
        self.promo = QTableView()
        self.pos = QTableView()
        self.cost = QTableView()

        self.gen_file_layout()
        """
        *************
        END FILE VIEW
        *************
        """


        # Training & Optimizaation Manager Group
        self.train_n_op_layout = QVBoxLayout()


        """
        ****************
        TRAINING MANAGER
        ****************
        """
        # Help Button
        self.train_help_button = QPushButton("?")
        self.train_help_button.setMaximumWidth(40)
        self.train_help_button.clicked.connect(self.training_help)

        # Status Label
        self.train_status = QLabel("Once you are satisfied with the files in File View press 'Train Model' to begin")
        self.train_status.setAlignment(Qt.AlignLeft)

        # Train Button
        self.train_button = QPushButton("Train Model")
        self.train_button.setFixedWidth(200)
        self.train_button.clicked.connect(self.train)

        self.gen_train_layout()
        """
        ********************
        END TRAINING MANAGER
        ********************
        """


        """
        ********************
        OPTIMIZATION MANAGER
        ********************
        """
        # Help Button
        self.opti_help_button = QPushButton("?")
        self.opti_help_button.setMaximumWidth(40)
        self.opti_help_button.clicked.connect(self.opti_help)

        # Status Label
        self.opti_status = QLabel("Select the month(s) to generate beer lists for:")
        self.opti_status.setAlignment(Qt.AlignCenter)

        # Percent label
        self.opti_perc = QLabel("            Store List")
        self.opti_perc.setAlignment(Qt.AlignCenter)

        # Progress Bar
        self.opti_progress = QProgressBar()
        self.opti_progress.setRange(0, 10000)
        self.opti_progress.setValue(0)

        self.opti_checkbox_jan = QCheckBox("Jan")
        self.opti_checkbox_feb = QCheckBox("Feb")
        self.opti_checkbox_mar = QCheckBox("Mar")
        self.opti_checkbox_apr = QCheckBox("Apr")
        self.opti_checkbox_may = QCheckBox("May")
        self.opti_checkbox_jun = QCheckBox("Jun")
        self.opti_checkbox_jul = QCheckBox("Jul")
        self.opti_checkbox_aug = QCheckBox("Aug")
        self.opti_checkbox_sep = QCheckBox("Sep")
        self.opti_checkbox_oct = QCheckBox("Oct")
        self.opti_checkbox_nov = QCheckBox("Nov")
        self.opti_checkbox_dec = QCheckBox("Dec")

        self.opti_checkbox_jan.setEnabled(False)
        self.opti_checkbox_feb.setEnabled(False)
        self.opti_checkbox_mar.setEnabled(False)
        self.opti_checkbox_apr.setEnabled(False)
        self.opti_checkbox_may.setEnabled(False)
        self.opti_checkbox_jun.setEnabled(False)
        self.opti_checkbox_jul.setEnabled(False)
        self.opti_checkbox_aug.setEnabled(False)
        self.opti_checkbox_sep.setEnabled(False)
        self.opti_checkbox_oct.setEnabled(False)
        self.opti_checkbox_nov.setEnabled(False)
        self.opti_checkbox_dec.setEnabled(False)

        self.opti_checkboxes = [self.opti_checkbox_jan, self.opti_checkbox_feb, self.opti_checkbox_mar, self.opti_checkbox_apr, self.opti_checkbox_may, self.opti_checkbox_jun, self.opti_checkbox_jul, self.opti_checkbox_aug, self.opti_checkbox_sep, self.opti_checkbox_oct, self.opti_checkbox_nov, self.opti_checkbox_dec]


        # Optimize Button
        self.opti_button = QPushButton("Optimize Model")
        self.opti_button.setFixedWidth(200)
        self.opti_button.setDisabled(True)
        self.opti_button.clicked.connect(self.run_opti)

        # Download Button
        self.opti_download_button = QPushButton("Download CSV")
        self.opti_download_button.setFixedWidth(200)
        self.opti_download_button.setDisabled(True)
        self.opti_download_button.clicked.connect(self.download_mandates)

        # State Combo Box
        """
        self.opti_state_combo = QComboBox()
        self.opti_state_combo.addItems(self.states)
        self.opti_state_combo.currentIndexChanged.connect(self.update_state)
        """

        # Table of Stores & Max Taps
        self.opti_state_list = QTableView()
        self.opti_state_list.setMinimumWidth(500)

        self.gen_opti_layout()
        """
        ************************
        END OPTIMIZATION MANAGER
        ************************
        """


        """
        #############################################
        #########  OPTIMIZATION ATTRIBUTES  #########
        #############################################
        """
        self.opti_object_dict = {}

        """
        #############################################
        #########  OPTIMIZATION ATTRIBUTES  #########
        #############################################
        """

        self.populate_files()

        # Adds all widgets to main layout
        self.gen_main_layout()

        # Shows GUI
        self.setLayout(self.main_layout)

    def gen_stores(self):
        """
        Cleans the store_list.xlsx file from the /GT/data/store directory

        returns:
            dict - {state (str): pd.DF}
        """
        d = Path.cwd()
        stores = pd.read_excel([f for f in d.parent.joinpath("GT/data/store").iterdir() if "xlsx" in f.name][0])
        stores = stores[stores["Ownership Type"] == "Corporate"]
        unique_states = stores["State"].unique()
        store_dict = {}
        for state in unique_states:
            store_dict[state] = stores[stores["State"] == state]
        return store_dict

    def gen_promo(self):
        """
        Cleans the promo.xlsx filee from the /GT/data/promo directory

        returns:
            dict - {state (str): {month {int}: beer type}}
        """
        d = Path.cwd()
        promos = pd.read_excel([f for f in d.parent.joinpath("GT/data/promo").iterdir() if "xlsx" in f.name][0])
        unique_states = promos["State"].unique()

        promo_dict = {}
        for state in unique_states:
            if state not in ["California - Franchise", "California - NorCal", "Washington D.C."]:
                promo_dict[state] = {}
                for month in range(1,13):
                    promo_dict[state][month] = []
        for index, row in promos.iterrows():
            if row["State"] not in ["California - Franchise", "California - NorCal", "Washington D.C."]:
                promo_dict[row["State"]][self.months[row["Month"]]].append(row["Name"])
        #print(promo_dict)
        return promo_dict

    def gen_beer_map(self):
        """
        Creates Beer Mapping Dictionary from Beer_Style_Mapping.csv
        """
        #print("\n\n\n\n BEER MAP")
        d = Path.cwd()
        beer_map = csv.reader(open([f for f in d.parent.joinpath("data").iterdir() if "Beer_Style_Mapping.csv" in f.name][0]))
        beer_map_dict = {}
        for beer, style in beer_map:
            beer_map_dict[beer] = style
        return beer_map_dict

    def reload_files(self):
        self.populate_files()

    def gen_status_bar(self):
        status_groupbox = QGroupBox()
        status_layout = QHBoxLayout()
        status_groupbox.setLayout(status_layout)
        status_layout.addWidget(QLabel("Status |"))
        status_layout.addWidget(self.status_label, stretch = 10)
        self.main_layout.addWidget(status_groupbox)

    def gen_file_layout(self):
        file_groupbox = QGroupBox()
        file_layout = QVBoxLayout()
        file_groupbox.setLayout(file_layout)

        # Create File Header
        file_header = QHBoxLayout()
        file_header_label = QLabel("File View      ")
        file_header_label.setAlignment(Qt.AlignCenter)
        file_header_label.setStyleSheet("font-style: bold")
        file_header.addWidget(self.file_reload_button)
        file_header.addWidget(file_header_label)
        file_header.addWidget(self.file_help_button)
        file_layout.addLayout(file_header)

        # Create File Layout Divider
        file_hsplit = QFrame()
        file_hsplit.setFrameShape(QFrame.HLine)
        file_hsplit.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Fixed)
        file_hsplit.setLineWidth(1)
        file_hsplit.setStyleSheet("color: gray")
        file_layout.addWidget(file_hsplit)
        
        # Create File Table Widgets
        file_view_lists = QVBoxLayout()
        promo_label = QLabel("Promotion Data")
        pos_label = QLabel("POS Data")
        cost_label = QLabel("Cost Data")
        file_view_lists.addWidget(promo_label)
        file_view_lists.addWidget(self.promo)
        file_view_lists.addSpacing(6)
        file_view_lists.addWidget(pos_label)
        file_view_lists.addWidget(self.pos)
        file_view_lists.addSpacing(6)
        file_view_lists.addWidget(cost_label)
        file_view_lists.addWidget(self.cost)
        
        file_layout.addLayout(file_view_lists)
        self.manager_layout.addWidget(file_groupbox, stretch = 1)

    def gen_train_layout(self):
        train_groupbox = QGroupBox()
        train_layout = QVBoxLayout()
        train_groupbox.setLayout(train_layout)

        # Create Training Header
        train_header = QHBoxLayout()
        train_header_label = QLabel("           Training Manager")
        train_header_label.setAlignment(Qt.AlignCenter)
        train_header_label.setStyleSheet("font-style: bold")
        train_header.addWidget(train_header_label)
        train_header.addWidget(self.train_help_button)
        train_layout.addLayout(train_header)

        # Create Training Layout Divider
        train_hsplit = QFrame()
        train_hsplit.setFrameShape(QFrame.HLine)
        train_hsplit.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Fixed)
        train_hsplit.setLineWidth(1)
        train_hsplit.setStyleSheet("color: gray")
        train_layout.addWidget(train_hsplit)

        # Create Training Widgets
        train_labels = QGridLayout()
        train_labels.addWidget(self.train_status, 0, 0, 1, 2, alignment=Qt.AlignCenter)
        train_labels.addWidget(self.train_button, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        train_layout.addSpacing(10)
        train_layout.addLayout(train_labels)
        self.train_n_op_layout.addWidget(train_groupbox, stretch = 1)

    def gen_opti_layout(self):
        opti_groupbox = QGroupBox()
        opti_layout = QVBoxLayout()
        opti_groupbox.setLayout(opti_layout)

        # Create Optimization Header
        opti_header = QHBoxLayout()
        opti_header_label = QLabel("           Optimization Manager")
        opti_header_label.setAlignment(Qt.AlignCenter)
        opti_header_label.setStyleSheet("font-style: bold")
        opti_header.addWidget(opti_header_label)
        opti_header.addWidget(self.opti_help_button)
        opti_layout.addLayout(opti_header)

        # Create Optimization Layout Divider
        opti_hsplit = QFrame()
        opti_hsplit.setFrameShape(QFrame.HLine)
        opti_hsplit.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Fixed)
        opti_hsplit.setLineWidth(1)
        opti_hsplit.setStyleSheet("color: gray")
        opti_layout.addWidget(opti_hsplit)

        # Create Optimization Month Check Box
        opti_months = QHBoxLayout()
        opti_months.addWidget(self.opti_checkbox_jan)
        opti_months.addWidget(self.opti_checkbox_feb)
        opti_months.addWidget(self.opti_checkbox_mar)
        opti_months.addWidget(self.opti_checkbox_apr)
        opti_months.addWidget(self.opti_checkbox_may)
        opti_months.addWidget(self.opti_checkbox_jun)
        opti_months.addWidget(self.opti_checkbox_jul)
        opti_months.addWidget(self.opti_checkbox_aug)
        opti_months.addWidget(self.opti_checkbox_sep)
        opti_months.addWidget(self.opti_checkbox_oct)
        opti_months.addWidget(self.opti_checkbox_nov)
        opti_months.addWidget(self.opti_checkbox_dec)

        # Create Optimization Widgets
        opti_widgets = QGridLayout()
        #opti_state_label = QLabel("State: ")
        #opti_state_label.setBuddy(self.opti_state_combo)
        opti_widgets.addWidget(self.opti_status, 0, 0, 1, 2, alignment=Qt.AlignCenter)
        opti_widgets.addLayout(opti_months, 1, 0, 1, 2)
        #opti_widgets.addWidget(opti_state_label, 2, 0, 1, 1, alignment=Qt.AlignRight)
        #opti_widgets.addWidget(self.opti_state_combo, 2, 1, 1, 1, alignment=Qt.AlignLeft)
        opti_widgets.addWidget(self.opti_perc, 2, 0, 1, 2, alignment=Qt.AlignLeft)
        opti_widgets.addWidget(self.opti_state_list, 3, 0, 1, 2, alignment=Qt.AlignCenter)
        opti_widgets.addWidget(self.opti_button, 4, 0, 1, 1, alignment=Qt.AlignCenter)
        opti_widgets.addWidget(self.opti_download_button, 4, 1, 1, 1, alignment=Qt.AlignCenter)

        opti_layout.addSpacing(10)
        opti_layout.addLayout(opti_widgets)
        self.train_n_op_layout.addWidget(opti_groupbox, stretch = 2)

    def gen_main_layout(self):
        """
        Adds all previously made layouts to the main layout
        """
        self.manager_layout.addLayout(self.train_n_op_layout, stretch = 1)
        self.main_layout.addLayout(self.manager_layout, stretch = 10)

    def file_help(self):
        return None
    
    def training_help(self):
        return None
    
    def opti_help(self):
        return None

    def update_main_status(self, status):
        """
        Params:
            status (str) - status message
        """
        self.status_label.setText(status)
    
    #Functions to update progress bar that's no longer being used
    """
    def update_train_progress(self, status, perc, value):
        # DocStrings
        Params:
            status (str) - status message
            pecr (str) - perc left message
            value (int) - value to set prograss bar to
        
        self.train_status.setText(status)
        self.train_perc.setText(perc)
        self.train_progress.setValue(value)
    
    def update_opti_progress(self, status, perc, value):
        # DocStrings
        Params:
            status (str) - status message
            perc (str) - perc left message
            value (int) - value to set prograss bar to
        
        self.opti_status.setText(status)
        self.opti_perc.setText(perc)
        self.opti_progress.setValue(value)
    """

    def sizeof_fmt(self, num, suffix='B'):
        """
        Converts the pathlib file size unit to interpretable size
        """
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix) 

    def populate_files(self):
        """
        Populates Promotion, POS, and Cost tables with the files in their respective folders
        """
        d = Path.cwd()
        pos = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/pos").iterdir() if "DS_Store" not in f.name]
        cost = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/cost").iterdir() if "DS_Store" not in f.name]
        promo = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/promo").iterdir() if "DS_Store" not in f.name]
        header = [["File Name", "Size"]]

        self.promo_model= SimpleTableModel(header + [f[1:] for f in promo])
        self.promo.setModel(self.promo_model)
        self.pos_model= SimpleTableModel(header + [f[1:] for f in pos])
        self.pos.setModel(self.pos_model)
        self.cost_model= SimpleTableModel(header + [f[1:] for f in cost])
        self.cost.setModel(self.cost_model)

        self.promo.setSelectionMode(QAbstractItemView.NoSelection)
        self.pos.setSelectionMode(QAbstractItemView.NoSelection)
        self.cost.setSelectionMode(QAbstractItemView.NoSelection)

        self.promo.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.promo.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.pos.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.pos.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.cost.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cost.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    
    def train(self):
        """MASSIVE HUGE GIGANTIC FUNCTION HERE"""
        self.train_button.setEnabled(False)
        self.file_reload_button.setEnabled(False)
        self.status_label.setText("The model is currently undergoing the training process where regression, model fitting, and value generation is being done. This will take some time")
        self.train_status.setText("Training . . .    Please be patient and reference command line for progress")
        """
        # Clean Restaurant Data
        self.stores = self.gen_stores()

        # Beer Mapping
        self.beer_map = self.gen_beer_map()
        """

        d = Path.cwd()
        pos = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/pos").iterdir() if "DS_Store" not in f.name]
        cost = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/cost").iterdir() if "DS_Store" not in f.name]
        promo = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/promo").iterdir() if "DS_Store" not in f.name]
        store = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/store").iterdir() if "DS_Store" not in f.name]
        sports = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/sports").iterdir() if "DS_Store" not in f.name]
        beermap = [(f, f.name, self.sizeof_fmt(f.stat().st_size)) for f in d.joinpath("data/beer").iterdir() if "DS_Store" not in f.name]

        start = time.time()
        print(start)

        # Clean Promo Data
        self.promo = self.gen_promo()

        self.datastore = ds.DataStore() # exog parameter
        
        for f in pos:
            # ADD PROGRESS PRINT STATEMENTS HERE
            # ADD PROGRESS PRINT STATEMENTS HERE
            # ADD PROGRESS PRINT STATEMENTS HERE
            # ADD PROGRESS PRINT STATEMENTS HERE
            # ADD PROGRESS PRINT STATEMENTS HERE
            print("loading " + str(f[0]))
            self.datastore.load_sales(str(f[0]))
            print("done")
        
        # Load Sports Data
        print("loading " + str(sports[0][0]))
        self.datastore.load_sports(str(sports[0][0]))
        print("done")
        
        # Load Beer Style Mapping
        print("loading " + str(beermap[0][0]))
        self.datastore.load_beer_styles(str(beermap[0][0]))
        print("done")

        # Load Cost Data
        print("loading " + str(cost[0][0]))
        self.datastore.load_costs(str(cost[0][0]))
        print("done")

        print("loading " + str(store[0][0]))
        self.datastore.load_store_info(str(store[0][0]))
        print("done")
        
        self.datastore.raw_sales.drop(self.datastore.raw_sales[self.datastore.raw_sales["RestaurantNumber"] == 3672].index)
        self.datastore.raw_sales.drop(self.datastore.raw_sales[self.datastore.raw_sales["Segment"] == 'Marble Cucumber Key Lime Hard Seltzer'].index)

        self.datastore.group_data("Month")
        self.datastore.load_exog()
        self.datastore.exog.to_pickle("month_exog_final.zip")

        self.datastore.load_exog('month_exog_final.zip')
        self.datastore.get_base_plan()
        self.datastore.separate_exog(subcategory=True)
        
        # create Segment column in DS.style_exog_avg
        self.datastore.style_exog_avg['Segment'] = self.datastore.style_exog_avg['Style']
        # rename DS.exog to be concatenated DS.base_exog and DS.style_exog_avg
        self.datastore.exog = pd.concat([self.datastore.base_exog,self.datastore.style_exog_avg],sort=False)


        # import RS object from regstore.py
        self.regstore = rs.RegStore()
        print("building model list in regstore")
        self.regstore.build_model_list(self.datastore)
        print("built model list in regstore")

        # initialize number of additional beers added to base plan
        num_beers = 8

        # create Restaurant objects for each restaurant in POS data
        self.restaurant_nums = [85,86]#self.datastore.exog.RestaurantNumber.unique()

        for restnum in self.restaurant_nums:
            row = self.datastore.store_info.loc[self.datastore.store_info["RestaurantNumber"] == restnum]
            tap_cap = row.iloc[0]["# of Handles"]
            name = row.iloc[0]["Location Name"]
            state = row.iloc[0]["State"]
            R = restaurant.Restaurant(self.datastore,self.regstore,restnum, tap_cap, name, state)
            R.gen_promo()
            R.build_proxys(num_beers,read_cache=True)
            R.get_profits(read_cache=True)
            self.restaurants.append(R)
            
        end = time.time()
        print(end)

        self.train_status.setText("Training Completed Successfully!")
        self.status_label.setText("The training model is complete! The table in the Optimization Manager has a list of stores that beer lists can be generated for. Click 'Optimize Model' when ready")

        # Enables all Optimization Month Check Boxes
        for checkbox in self.opti_checkboxes:
            checkbox.setEnabled(True)
            checkbox.setChecked(True)
        
        if len(self.restaurants) == 1:
            self.opti_perc.setText("            Store List (" + str(len(self.restaurants)) + " store)")
        else:
            self.opti_perc.setText("            Store List (" + str(len(self.restaurants)) + " stores)")
        self.update_restaurant_table()

        self.opti_button.setEnabled(True)
        print("done")
        return None

    def update_restaurant_table(self):
        """
        Will update TableView in Optimization Manager with restaurant lists depending on which restaurants there is data for
        """
        header = [["Restaurant #", "Location Name", "# of Handles"]]
        store_list =  []
        for restaurant in self.restaurants:
            store_list.append((restaurant.restnum, restaurant.name, int(restaurant.tap_capacity)-len(self.datastore.base_plan)))
        
        self.opti_state_list_model = SimpleTableModel(header + store_list)
        self.opti_state_list.setModel(self.opti_state_list_model)

        self.opti_state_list.setSelectionMode(QAbstractItemView.NoSelection)

        self.opti_state_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def update_state(self):
        """
        Will update TableView in Optimization Manager with restaurant lists depending on which state is chosen
        """
        state = self.opti_state_combo.currentText()

        if state == " ":
            self.opti_button.setDisabled(True)
            #self.opti_status.setText("Status: Select a State")

            self.opti_state_list.setModel(None)
        else:
            self.opti_button.setDisabled(False)
            #self.opti_status.setText("Status: Press 'Optimize Model' to build schedule for {} restaruants".format(state))
            header = [["Restaurant #", "Location Name", "# of Handles"]]
            store_list = [(store["Restaurant #"], store["Location Name"], store["# of Handles"]) for index, store in self.stores[state].iterrows()]
            #
            # TO DO
            #
            # Reference DS Object to filter based on restaurants that are loaded into DS
            #
            #
            self.opti_state_list_model = SimpleTableModel(header + store_list)
            self.opti_state_list.setModel(self.opti_state_list_model)

            self.opti_state_list.setSelectionMode(QAbstractItemView.NoSelection)

            self.opti_state_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def run_opti(self):
        """
        Function for "Optimize Model Button" which will run the optimizer on all of the restaurants
        """
        
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        months_dict = {months[i-1]:i for i in range(1,13)}

        months_checked = []
        for checkbox in self.opti_checkboxes:
                if checkbox.isChecked():
                    months_checked.append(months_dict[checkbox.text()])

        key = []
        try:
            f = open("results/results.csv", "r")
            reader = csv.reader(f)
            next(reader,None)
            for row in reader:
                key.append([int(row[0]),int(row[1])])
        except:
            pass

        if len(months_checked) != 0:
        # Creates Month List based on check boxes
            for checkbox in self.opti_checkboxes:
                checkbox.setDisabled(True)

            self.opti_button.setDisabled(True)
            self.status_label.setText("Optimal Beer Lists are being generated for {} stores. Please be patient as this may take some time".format(len(self.restaurants)))
            

            
            self.all_optimal_plans = {}
            # Loops through all the restaurants and runs the optimziation model on each one
            for r in self.restaurants:
                restaurant = r.restnum
            
                optimal_plans_dict = {}

                
                # try to read cached data. If not available, run all optimizations
                try:
                    with open('support/dev/Pickles/optimalPlans/'+str(restaurant)+'_optimal_plan.pickle','rb') as handle:
                        optimal_plans_dict = pickle.load(handle)
                        print("Successfully opened cached optimal_plan.pickle")
                except:
                    pass
                

                filename = 'support/dev/Pickles/profitDictionaries/'+str(restaurant)+'_profit_dict.pickle'
                with open(filename, 'rb') as handle:
                    profit_dict = pickle.load(handle)
                
                # only perform optimization for months not in cached dictionary
                for month in months_checked:
                    if (month not in optimal_plans_dict.keys()):
                        beer_map = self.datastore.beer_styles.set_index("Segment").to_dict()["Style"]
                        
                        bur_promo_input = self.promo[self.state_dict[r.state]][month]
                        print(bur_promo_input)
                        # run optimization
                        print("running optimizer for month " + str(month))
                        opt = optimizer.Optimizer(restaurant, int(r.tap_capacity), profit_dict[month], beer_map, self.datastore.base_plan, bur_promo_input)
                        opt.gen_results_list()

                        # get organized results list
                        test = list(opt.mandate_list.items())
                        test.sort(key=lambda x: (x[0][0], x[0][1]))
                        print("******** MANDATE_LIST ********")
                        print("restaurant " + str(restaurant))
                        print("month " + str(month))
                        print(test)
                        
                        optimal_plans_dict[month] = test
                        
                print("***************** OPTIMAL_PLANS_DICT ****************")
                self.all_optimal_plans[r.restnum] = optimal_plans_dict
                optimal_plan_filename = 'support/dev/Pickles/optimalPlans/'+str(restaurant)+'_optimal_plan.pickle'
                with open(optimal_plan_filename,'wb') as handle:
                    pickle.dump(optimal_plans_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)

            print(self.all_optimal_plans)
            self.status_label.setText("The optimal beer lists have successfully been generated! Please download the mandates and reference them in the 'results' folder")
            self.opti_download_button.setDisabled(False)
        else:
            self.status_label.setText("Please check at least one month to generate beer lists for")

    def download_mandates(self):
        """
        Downloads a CSV file of beer lists
        """
        d = Path.cwd()
        
        # make "results" folder if does not already exist
        try:
            os.mkdir("results")        
        except:
            pass


        with open("results/results.csv","a+") as f:
            
            writer = csv.writer(f)
            reader = csv.reader(f)
            num = 20

            if os.stat("results/results.csv").st_size == 0:
                # Generates Header
                header = ["Restaurant Number", "Month"]
                style = ["Beer Style " + str(n) for n in range(1,num+1)]
                volume = ["Beer Volume " + str(n) for n in range(1,num+1)]
                for n in range(num):
                    header.append(style[n])
                    header.append(volume[n])
                writer.writerow(header)

                # Generates Rows
                for restaurant in self.restaurants:
                    for month in self.all_optimal_plans[restaurant.restnum]:
                        row = [restaurant.restnum, month]
                        style = [beer[0][0] + " " + str(beer[0][1]) for beer in self.all_optimal_plans[restaurant.restnum][month]]
                        value = [round(value[1][1],4) for value in self.all_optimal_plans[restaurant.restnum][month]]
                        for n in range(len(style)):
                            row.append(style[n])
                            row.append(value[n])
                        writer.writerow(row)
            else:
                key = []
                print("RESULTSCSV ALREADY EXISTS")
                f.seek(0)
                next(reader,None)
                for row in reader:
                    key.append([int(row[0]),int(row[1])])
                print("PRINTING KEY")
                print(key)

                # Generates Rows
                for restaurant in self.restaurants:
                    for month in self.all_optimal_plans[restaurant.restnum]:
                        row = [int(restaurant.restnum), int(month)]
                        if row not in key:
                            style = [beer[0][0] + " " + str(beer[0][1]) for beer in self.all_optimal_plans[restaurant.restnum][month]]
                            value = [round(value[1][1],4) for value in self.all_optimal_plans[restaurant.restnum][month]]
                            for n in range(len(style)):
                                row.append(style[n])
                                row.append(value[n])
                            writer.writerow(row)
                        else:
                            print(key)
                            print(row)
        
        # Widget Updates
        self.status_label.setText("Beer Lists have successfully been downloaded please reference them in the 'results' folder. You may now select another set of months or exit the app")
        self.opti_button.setEnabled(True)
        self.opti_download_button.setEnabled(False)
        for checkbox in self.opti_checkboxes:
            checkbox.setDisabled(False)

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    login = LoginPage()
    login.show()

    sys.exit(app.exec_()) 