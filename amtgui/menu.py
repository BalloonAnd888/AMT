from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow

class MenuBar:
    def __init__(self, main_window: QMainWindow):
        menu = main_window.menuBar()

        # File menu
        create_session_button_action = QAction("Create Session", main_window)
        create_session_button_action.triggered.connect(self.create_session_button_clicked)
        create_session_button_action.setShortcut(QKeySequence("Ctrl+p"))

        open_session_button_action = QAction("Open Session", main_window)
        open_session_button_action.triggered.connect(self.open_session_button_clicked)
        open_session_button_action.setShortcut(QKeySequence("Ctrl+o"))

        save_session_button_action = QAction("Save Session", main_window)
        save_session_button_action.triggered.connect(self.save_session_button_clicked)
        save_session_button_action.setShortcut(QKeySequence("Ctrl+s"))

        file_menu = menu.addMenu("&File")
        file_menu.addAction(create_session_button_action)
        file_menu.addSeparator() 
        file_menu.addAction(open_session_button_action)
        file_menu.addSeparator() 
        file_menu.addAction(save_session_button_action)

        # Help menu
        keyboard_shortcut_button_action = QAction("Keyboard Shortcuts", main_window)
        keyboard_shortcut_button_action.triggered.connect(self.keyboard_shortcut_button_clicked)
        keyboard_shortcut_button_action.setShortcut(QKeySequence("Ctrl+h"))
 
        keyboard_shortcut_menu = menu.addMenu("&Help")
        keyboard_shortcut_menu.addAction(keyboard_shortcut_button_action)
    
    def create_session_button_clicked(self):
        print("Create Session button clicked")

    def open_session_button_clicked(self):
        print("Open Session button clicked")
    
    def save_session_button_clicked(self):
        print("Save Session button clicked")
    
    def keyboard_shortcut_button_clicked(self):
        print("Keyboard Shortcut button clicked")
        