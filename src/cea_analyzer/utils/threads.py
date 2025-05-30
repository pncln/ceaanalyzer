"""
Background Thread Implementation Module
------------------------------------

This module provides thread implementations for background processing tasks
in the CEA Analyzer application. These threads are used to avoid blocking the
main UI thread during potentially long-running operations.
"""

import logging
from typing import Optional, Any
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from ..analysis.cea_parser import parse_cea_output


class ParserThread(QThread):
    """
    Background thread for parsing CEA output files.
    
    This thread performs the potentially time-consuming operation of parsing
    a CEA output file without blocking the UI thread.
    
    Signals
    -------
    progress : int
        Signal emitted with progress percentage (0-100)
    finished : pd.DataFrame
        Signal emitted with the parsed DataFrame when complete
    error : str
        Signal emitted with error message on failure
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, filepath: str):
        """
        Initialize the parser thread.
        
        Parameters
        ----------
        filepath : str
            Path to the CEA output file to parse
        """
        super().__init__()
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """
        Execute the parsing operation in the background thread.
        
        This method is automatically called when the thread is started.
        It parses the CEA output file and emits signals for progress updates,
        completion, or errors.
        """
        try:
            self.logger.info(f"Starting to parse file: {self.filepath}")
            df = parse_cea_output(self.filepath, self.progress.emit)
            self.logger.info(f"Successfully parsed file with {len(df)} records")
            self.finished.emit(df)
        except Exception as e:
            self.logger.exception(f"Error parsing CEA output: {e}")
            self.error.emit(str(e))
