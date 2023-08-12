import logging
import os
from datetime import datetime

class CustomLogger:
    """
    A custom logger class that provides both file and console logging capabilities.
    """

    def __init__(self):
        """
        Initialize the CustomLogger instance.

        This constructor sets up the logger with a file handler and a console handler.
        It configures the log format and directory.
        """
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"{current_date}.log")
        
        # Create a logger instance
        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.DEBUG)
        
        # Define the log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger instance
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def __getattr__(self, name):
        """
        Get an attribute of the logger instance.

        This method allows access to attributes and methods of the underlying logger instance.
        """
        return getattr(self.logger, name)

logger  = CustomLogger()