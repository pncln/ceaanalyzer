"""
Data Export Module
----------------

This module provides functionality for exporting CEA analysis results
to various file formats including CSV, Excel, and PDF.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# Configure module logger
logger = logging.getLogger(__name__)


def export_csv(df: pd.DataFrame, filename: str) -> bool:
    """
    Export a DataFrame to CSV format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    filename : str
        Target filename for the CSV file
        
    Returns
    -------
    bool
        True if export was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(filename, index=False, float_format='%.6g')
        logger.info(f"Successfully exported data to CSV: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        return False


def export_excel(df: pd.DataFrame, summary: Optional[pd.DataFrame] = None, 
                filename: str = None) -> bool:
    """
    Export data to Excel format with multiple sheets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame to export as the 'Data' sheet
    summary : pd.DataFrame, optional
        Summary DataFrame to export as the 'Summary' sheet
    filename : str
        Target filename for the Excel file
        
    Returns
    -------
    bool
        True if export was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            
            # Add summary sheet if provided
            if summary is not None:
                summary.to_excel(writer, sheet_name="Summary", index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Export Date', 'Records', 'Columns', 'Library Versions'],
                'Value': [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    len(df),
                    len(df.columns),
                    f"pandas {pd.__version__}"
                ]
            })
            metadata.to_excel(writer, sheet_name="Metadata", index=False)
            
        logger.info(f"Successfully exported data to Excel: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export Excel file: {e}")
        return False


def export_pdf(figures: Dict[str, Figure], title: str, filename: str) -> bool:
    """
    Save a sequence of matplotlib Figure objects into a single PDF file.
    
    Parameters
    ----------
    figures : Dict[str, Figure]
        Dictionary of named matplotlib Figure objects to include in the PDF
    title : str
        Title to display on the cover page
    filename : str
        Target filename for the PDF file
        
    Returns
    -------
    bool
        True if export was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(filename) as pdf:
            # Create cover page if not provided
            if "Cover" not in figures:
                cover_fig = plt.figure(figsize=(8.5, 11))
                cover_ax = cover_fig.add_subplot(111)
                cover_ax.text(0.5, 0.6, title, fontsize=24, ha='center')
                cover_ax.text(0.5, 0.5, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                          fontsize=14, ha='center')
                cover_ax.text(0.5, 0.4, "CEA Analyzer", fontsize=18, ha='center')
                cover_ax.axis('off')
                pdf.savefig(cover_fig)
                plt.close(cover_fig)
            else:
                # Use provided cover page
                pdf.savefig(figures["Cover"])
            
            # Add all other figures
            for name, fig in figures.items():
                if name != "Cover":
                    # Add a title to the figure if it doesn't have one
                    if not fig._suptitle:
                        fig.suptitle(name)
                    pdf.savefig(fig)
            
            # Set document metadata
            d = pdf.infodict()
            d['Title'] = title
            d['Author'] = 'CEA Analyzer'
            d['Subject'] = 'CEA Analysis Results'
            d['Keywords'] = 'CEA, rocket, propulsion, analysis'
            d['CreationDate'] = datetime.datetime.now()
            d['ModDate'] = datetime.datetime.now()
        
        logger.info(f"Successfully exported figures to PDF: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export PDF: {e}")
        return False


def export_report(df: pd.DataFrame, figures: Dict[str, Figure], 
                 filename: str, title: str = "CEA Analysis Report") -> bool:
    """
    Create a comprehensive PDF report with data tables and figures.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the analysis results
    figures : Dict[str, Figure]
        Dictionary of matplotlib Figure objects to include in the report
    filename : str
        Target filename for the PDF report
    title : str, optional
        Title for the report (default: "CEA Analysis Report")
        
    Returns
    -------
    bool
        True if export was successful, False otherwise
    """
    try:
        # Create table figure
        table_fig = plt.figure(figsize=(11, 8.5))
        ax = table_fig.add_subplot(111)
        ax.axis('off')
        
        # Create a smaller dataframe for visualization if too large
        display_df = df
        if len(df) > 20:
            # Show first 10 and last 10 rows
            display_df = pd.concat([df.head(10), df.tail(10)])
        
        # Create the table
        table = ax.table(cellText=display_df.values, 
                         colLabels=display_df.columns, 
                         cellLoc='center', 
                         loc='center',
                         colWidths=[0.1] * len(display_df.columns))
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        table_fig.suptitle("Data Table Preview", fontsize=16)
        
        # Add table figure to the dictionary
        report_figures = {"DataTable": table_fig, **figures}
        
        # Export the report
        result = export_pdf(report_figures, title, filename)
        plt.close(table_fig)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create report: {e}")
        return False
