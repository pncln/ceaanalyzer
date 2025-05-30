"""
Data Models Module
-----------------

This module provides data models for the CEA Analyzer application.
Primarily, it contains the PandasModel class which makes pandas DataFrames
compatible with Qt's model/view architecture.
"""

from typing import Any, Optional, Union
from PyQt6.QtCore import QAbstractTableModel, Qt, QModelIndex
import pandas as pd
import numpy as np


class PandasModel(QAbstractTableModel):
    """
    A Qt model to display a pandas DataFrame in Qt views.
    
    This class implements the necessary methods to convert a pandas DataFrame
    into a Qt TableModel that can be displayed in various Qt views like QTableView.
    """
    
    def __init__(self, df: pd.DataFrame = pd.DataFrame(), parent: Any = None):
        """
        Initialize the model with a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            The DataFrame to display, defaults to empty DataFrame
        parent : Any, optional
            Parent widget, defaults to None
        """
        super().__init__(parent)
        self._df = df.copy()
        
        # Cache column data types for formatting
        self._dtypes = self._df.dtypes

    def rowCount(self, parent: QModelIndex = None) -> int:
        """
        Return the number of rows in the model.
        
        Parameters
        ----------
        parent : QModelIndex, optional
            Parent index, defaults to None
            
        Returns
        -------
        int
            Number of rows in the DataFrame
        """
        if parent is None or not parent.isValid():
            return len(self._df)
        return 0

    def columnCount(self, parent: QModelIndex = None) -> int:
        """
        Return the number of columns in the model.
        
        Parameters
        ----------
        parent : QModelIndex, optional
            Parent index, defaults to None
            
        Returns
        -------
        int
            Number of columns in the DataFrame
        """
        if parent is None or not parent.isValid():
            return len(self._df.columns)
        return 0

    def headerData(self, section: int, orientation: Qt.Orientation, 
                  role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        """
        Return header data for the specified role and section.
        
        Parameters
        ----------
        section : int
            Row or column number
        orientation : Qt.Orientation
            Horizontal or vertical orientation
        role : int, optional
            Data role, defaults to DisplayRole
            
        Returns
        -------
        Optional[str]
            Header data for the specified role and section
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal and section < len(self._df.columns):
                return str(self._df.columns[section])
            elif orientation == Qt.Orientation.Vertical and section < len(self._df):
                return str(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        """
        Return data for the specified index and role.
        
        Parameters
        ----------
        index : QModelIndex
            Index to retrieve data for
        role : int, optional
            Data role, defaults to DisplayRole
            
        Returns
        -------
        Optional[str]
            Data for the specified index and role
        """
        if not index.isValid():
            return None
            
        row, col = index.row(), index.column()
        
        # Check if within bounds
        if row < 0 or row >= len(self._df) or col < 0 or col >= len(self._df.columns):
            return None
            
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            value = self._df.iat[row, col]
            
            # Format based on data type
            if pd.isna(value):
                return ""
            elif isinstance(value, float):
                return f"{value:.6g}"  # Use general format with 6 significant digits
            else:
                return str(value)
                
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            # Right-align numeric columns
            col_type = self._dtypes.iloc[col]
            if np.issubdtype(col_type, np.number):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            
        return None
        
    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        """
        Set data for the specified index and role.
        
        Parameters
        ----------
        index : QModelIndex
            Index to set data for
        value : Any
            Value to set
        role : int, optional
            Data role, defaults to EditRole
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
            
        row, col = index.row(), index.column()
        
        # Check if within bounds
        if row < 0 or row >= len(self._df) or col < 0 or col >= len(self._df.columns):
            return False
            
        # Try to convert value to the correct type
        try:
            col_type = self._dtypes.iloc[col]
            if np.issubdtype(col_type, np.number):
                value = float(value)
            self._df.iat[row, col] = value
            self.dataChanged.emit(index, index)
            return True
        except (ValueError, TypeError):
            return False
            
    def flags(self, index: QModelIndex) -> int:
        """Return the item flags for the given index."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
            
        # By default, items are enabled and selectable but not editable
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
