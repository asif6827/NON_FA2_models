import logging
import os
from datetime import datetime

class PuzzleLogger:
    """
    Logger class for puzzle solving system.
    
    This class provides a unified logging interface with support for different logging levels
    and both file and console output.
    """
    def __init__(self, name: str = "puzzle_solver", log_file: str = None, level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional, if not provided, only console output is used)
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log_file is provided
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger.critical(message)
    
    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

# Create a default logger instance
DEFAULT_LOGGER = PuzzleLogger()

# Export convenience functions for easy logging
def get_logger(name: str = "puzzle_solver", log_file: str = None, level: int = logging.INFO) -> PuzzleLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        PuzzleLogger instance
    """
    return PuzzleLogger(name, log_file, level)

def debug(message: str) -> None:
    """Convenience function for debug logging."""
    DEFAULT_LOGGER.debug(message)

def info(message: str) -> None:
    """Convenience function for info logging."""
    DEFAULT_LOGGER.info(message)

def warning(message: str) -> None:
    """Convenience function for warning logging."""
    DEFAULT_LOGGER.warning(message)

def error(message: str) -> None:
    """Convenience function for error logging."""
    DEFAULT_LOGGER.error(message)

def critical(message: str) -> None:
    """Convenience function for critical logging."""
    DEFAULT_LOGGER.critical(message)

def create_timestamped_log_file(output_dir: str, prefix: str = "puzzle_log") -> str:
    """
    Create a timestamped log file path.
    
    Args:
        output_dir: Output directory for the log file
        prefix: Log file prefix
        
    Returns:
        Path to the timestamped log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")
    return log_file
