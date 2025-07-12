# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Claude
# @Desc    : Simple colored logger with file output

import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional, TextIO, Union

class Colors:
    """Terminal color codes for different log levels"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class LogLevel(Enum):
    """Log levels with corresponding colors"""
    DEBUG = (10, Colors.BLUE)
    INFO = (20, Colors.GREEN)
    WARNING = (30, Colors.YELLOW)
    ERROR = (40, Colors.RED)
    CRITICAL = (50, Colors.MAGENTA)

class SimpleLogger:
    """Simple logger class that supports both colored terminal output and file logging"""
    
    def __init__(
        self, 
        name: str = "MermaidFlow",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        console_output: bool = True
    ):
        """
        Initialize the Logger
        
        Args:
            name: Logger name
            log_level: Minimum log level to display
            log_file: Log file name (if None, will use name_YYYY-MM-DD.log)
            log_dir: Directory to store log files
            console_output: Whether to output logs to console
        """
        self.name = name
        
        # Convert LogLevel enum to int if needed
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value[0]
        else:
            self.log_level = log_level
        
        self.console_output = console_output
        self.file_output = None
        
        # Set up file logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate default log filename if not provided
            if log_file is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = f"{name}_{current_date}.log"
            
            file_path = os.path.join(log_dir, log_file)
            self.file_output = open(file_path, 'a', encoding='utf-8')
    
    def _log(self, level: LogLevel, message: str) -> None:
        """Internal method to log messages at specified level"""
        if level.value[0] < self.log_level:
            return
            
        # Format the log message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = level.name
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        
        # Write to console if enabled
        if self.console_output:
            color = level.value[1]
            # Add bold to critical messages
            if level == LogLevel.CRITICAL:
                colored_msg = f"{Colors.BOLD}{color}{formatted_msg}{Colors.RESET}"
            else:
                colored_msg = f"{color}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        
        # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    
    def debug(self, message: str) -> None:
        """Log a debug message"""
        self._log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log an info message"""
        self._log(LogLevel.INFO, message)
    
    def warning(self, message: str) -> None:
        """Log a warning message"""
        self._log(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log an error message"""
        self._log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """Log a critical message"""
        self._log(LogLevel.CRITICAL, message)
    
    def __del__(self):
        """Close file handle when logger is destroyed"""
        if self.file_output:
            self.file_output.close()

# Create a singleton instance for easy import
logger = SimpleLogger()


def test_logger():
    """Test function to verify the SimpleLogger functionality"""
    
    # Create a test log directory
    test_log_dir = "test_logs"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
    
    # Initialize the logger with a specific test file
    test_logger = SimpleLogger(
        name="test_logger",
        log_level=LogLevel.DEBUG,  # Set to lowest level to show all logs
        log_file="test_logger.log",
        log_dir=test_log_dir
    )
    
    print("\n===== Testing SimpleLogger =====\n")
    
    # Test all log levels
    test_logger.debug("This is a DEBUG message - Should appear in BLUE")
    test_logger.info("This is an INFO message - Should appear in GREEN")
    test_logger.warning("This is a WARNING message - Should appear in YELLOW")
    test_logger.error("This is an ERROR message - Should appear in RED")
    test_logger.critical("This is a CRITICAL message - Should appear in BOLD MAGENTA")
    
    # Test log filtering by creating a new logger with higher minimum level
    print("\n===== Testing Log Level Filtering =====\n")
    filtered_logger = SimpleLogger(
        name="filtered_logger",
        log_level=LogLevel.WARNING,  # Only WARNING and above will be shown
        log_file="filtered_logger.log",
        log_dir=test_log_dir
    )
    
    filtered_logger.debug("This DEBUG message should NOT appear")
    filtered_logger.info("This INFO message should NOT appear")
    filtered_logger.warning("This WARNING message should appear in YELLOW")
    filtered_logger.error("This ERROR message should appear in RED")
    filtered_logger.critical("This CRITICAL message should appear in BOLD MAGENTA")
    
    # Verify file output
    print("\n===== Verifying File Output =====\n")
    
    # Get path to the log file
    log_file_path = os.path.join(test_log_dir, "test_logger.log")
    
    # Check if the log file exists
    if os.path.exists(log_file_path):
        # Read and print the last 5 lines from the log file
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print(f"Last 5 lines from log file ({log_file_path}):")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
        print(f"\nLog file successfully created at: {log_file_path}")
    else:
        print(f"ERROR: Log file was not created at: {log_file_path}")
    
    print("\n===== Testing Complete =====\n")


def test_in_app_scenario():
    """Test function to demonstrate how the logger would be used in an application"""
    
    logger = SimpleLogger(name="app_logger")
    
    print("\n===== Simulating Application Logs =====\n")
    
    # Simulate application startup
    logger.info("Application starting up...")
    
    # Simulate configuration loading
    logger.debug("Loading configuration from config.json")
    time.sleep(0.5)
    logger.info("Configuration loaded successfully")
    
    # Simulate some application processing
    logger.info("Processing data files...")
    time.sleep(0.5)
    
    # Simulate a warning condition
    logger.warning("Memory usage is high (85%)")
    
    # Simulate an error condition
    try:
        # Simulate a division by zero error
        result = 100 / 0
    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}")
    
    # Simulate a critical error
    logger.critical("Database connection lost! System cannot continue.")
    
    # Simulate application shutdown
    logger.info("Application shutting down")
    
    print("\n===== Simulation Complete =====\n")


if __name__ == "__main__":
    # Run tests
    test_logger()
    test_in_app_scenario()