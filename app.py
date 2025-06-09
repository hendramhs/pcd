import sys
from PyQt5 import QtWidgets
from ui_handler import TrashClassificationUI

# Main application entry point
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TrashClassificationUI()
    window.show()
    sys.exit(app.exec_())