import sys
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib
# Set the backend to qtagg which works with PyQt6
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableView, QTabWidget, QWidget, \
    QVBoxLayout, QTextEdit, QDockWidget, QFormLayout, QLineEdit, QPushButton, \
    QStatusBar, QProgressBar, QFileDialog, QSizePolicy, QComboBox, \
    QHBoxLayout, QLabel, QGroupBox, QRadioButton, QButtonGroup, QCheckBox, QGridLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QAction

from .analysis.cea_parser import parse_cea_output
from .core.models import PandasModel
from .utils.threads import ParserThread
from .utils.plots import create_graphs
from .analysis import compute_system
from .utils.export import export_csv, export_excel, export_pdf
from .core.config import CONFIG, CONFIG_PATH
from .propulsion import nozzle
from .ui.widgets.grain_visualization_widget import GrainVisualizationWidget
# MOC import removed

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CEA Analyzer")
        self.resize(1000, 700)

        # Menu
        men = self.menuBar().addMenu("File")
        act_open = QAction("Open", self)
        # ignore the 'checked' bool and always call open_file() with no args
        act_open.triggered.connect(lambda checked=False: self.open_file())
        act_open.triggered.connect(self.open_file)
        men.addAction(act_open)

        # Tabs
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        # Data table
        self.tbl = QTableView(); self.tabs.addTab(self.tbl, "Data")
        # Graphs
        # Graphs (start with empty canvases; real plots come after loading data)
        self.graphTabs = QTabWidget(); self.figures, self.canvases = {}, {}
        for name in ["Isp","Temp","PressureRatio","Enthalpy"]:
            fig = Figure(figsize=(5,3))
            can = FigureCanvas(fig)
            w = QWidget(); l = QVBoxLayout(w); l.addWidget(can)
            self.graphTabs.addTab(w, name)
            self.figures[name] = fig
            self.canvases[name] = can
        self.tabs.addTab(self.graphTabs, "Graphs")

        # Summary & Optimization & Nozzle/System & Recommendations
        self.sum_text = QTextEdit(); self.sum_text.setReadOnly(True); self.tabs.addTab(self.sum_text, "Summary")
        self.opt_canvas = FigureCanvas(Figure(figsize=(8,4)))
        self.opt_text = QTextEdit(); self.opt_text.setReadOnly(True)
        wopt=QWidget(); lopt=QVBoxLayout(wopt); lopt.addWidget(self.opt_canvas); lopt.addWidget(self.opt_text)
        self.tabs.addTab(wopt, "Optimization")
        self.sys_canvas = FigureCanvas(self.figures["Isp"])  # placeholder
        self.sys_text = QTextEdit(); self.sys_text.setReadOnly(True)
        wsys=QWidget(); lsys=QVBoxLayout(wsys); lsys.addWidget(self.sys_canvas); lsys.addWidget(self.sys_text)
        self.tabs.addTab(wsys, "Nozzle/System")
        self.reco = QTextEdit(); self.reco.setReadOnly(True); self.tabs.addTab(self.reco, "Recommendations")
        
        # ─── Nozzle Design Tab ───
        self.nozzle_widget = QWidget()
        self.nozzle_layout = QVBoxLayout(self.nozzle_widget)
        
        # Control panel for nozzle design
        control_panel = QGroupBox("Nozzle Design Controls")
        control_layout = QGridLayout()
        
        # Nozzle type selection
        self.nozzle_type_label = QLabel("Nozzle Type:")
        self.nozzle_type_combo = QComboBox()
        self.nozzle_type_combo.addItems(["Conical", "Rao Optimum", "80% Bell", "Method of Characteristics (MOC)", "Truncated Ideal Contour (TIC)"])
        self.nozzle_type_combo.currentIndexChanged.connect(self.update_nozzle_design)
        control_layout.addWidget(self.nozzle_type_label, 0, 0)
        control_layout.addWidget(self.nozzle_type_combo, 0, 1)
        
        # Throat radius control
        self.throat_radius_label = QLabel("Throat Radius (m):")
        self.throat_radius_edit = QLineEdit()
        self.throat_radius_edit.setText("0.05")
        self.throat_radius_edit.textChanged.connect(self.update_nozzle_design)
        control_layout.addWidget(self.throat_radius_label, 1, 0)
        control_layout.addWidget(self.throat_radius_edit, 1, 1)
        
        # Include inlet section checkbox
        self.include_inlet_checkbox = QCheckBox("Include Inlet Section")
        self.include_inlet_checkbox.setChecked(True)
        self.include_inlet_checkbox.stateChanged.connect(self.update_nozzle_design)
        control_layout.addWidget(self.include_inlet_checkbox, 2, 0, 1, 2)
        
        # Export nozzle coordinates button
        self.export_nozzle_button = QPushButton("Export Nozzle Coordinates")
        self.export_nozzle_button.clicked.connect(self.export_nozzle_coordinates)
        control_layout.addWidget(self.export_nozzle_button, 3, 0, 1, 2)
        
        control_panel.setLayout(control_layout)
        self.nozzle_layout.addWidget(control_panel)
        
        # Nozzle visualization
        self.nozzle_canvas = FigureCanvas(Figure(figsize=(10, 6), tight_layout=True))
        self.nozzle_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.nozzle_layout.addWidget(self.nozzle_canvas)
        
        # Nozzle performance text
        self.nozzle_text = QTextEdit()
        self.nozzle_text.setReadOnly(True)
        self.nozzle_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.nozzle_text.setMaximumHeight(150)
        self.nozzle_layout.addWidget(self.nozzle_text)
        
        self.tabs.addTab(self.nozzle_widget, "Nozzle Design")
        
        # ─── Grain Design Tab ───
        self.grain_design_widget = GrainVisualizationWidget()
        grain_tab_index = self.tabs.addTab(self.grain_design_widget, "Grain Design")
        
        # Ensure the grain design tab is visible and properly initialized
        self.grain_design_widget.setVisible(True)
        
        # Switch to the grain design tab to ensure it's initialized properly
        # Comment out the next line if you want to start on a different tab
        self.tabs.setCurrentIndex(grain_tab_index)

        # MOC tab has been removed

        # Filters dock
        dock = QDockWidget("Filters", self)
        fw = QWidget(); fl = QFormLayout(fw)
        self.filters = {}
        for col in ["O/F","Pc (bar)","Isp (s)"]:
            mn, mx = QLineEdit(), QLineEdit()
            fl.addRow(f"{col} min:", mn); fl.addRow(f"{col} max:", mx)
            self.filters[col] = (mn, mx)
        btnA = QPushButton("Apply"); btnR = QPushButton("Reset")
        btnA.clicked.connect(self.apply_filters); btnR.clicked.connect(self.reset_filters)
        fl.addRow(btnA, btnR)
        dock.setWidget(fw); self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # Status bar
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.pbar = QProgressBar(); self.status.addPermanentWidget(self.pbar)

        # Export actions
        exp = men.addMenu("Export")
        act_csv = QAction("CSV", self); act_csv.triggered.connect(self.export_csv)
        act_xlsx = QAction("Excel", self); act_xlsx.triggered.connect(self.export_excel)
        act_pdf = QAction("PDF", self); act_pdf.triggered.connect(self.export_pdf)
        exp.addAction(act_csv); exp.addAction(act_xlsx); exp.addAction(act_pdf)

        # Data holders
        self.df_full = self.df = None

    def open_file(self, path=None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open CEA Output", "", "Text Files (*.txt *.out);;All Files (*)")
        if not path:
            return
        self.thread = ParserThread(path)
        self.thread.progress.connect(self.pbar.setValue)
        self.thread.finished.connect(self._on_parsed)
        self.thread.error.connect(lambda e: self.status.showMessage(f"Error: {e}", 5000))
        self.status.showMessage("Parsing...", 2000)
        self.thread.start()

    def _on_parsed(self, df):
        self.df_full = df.copy(); self.df = df
        self.update_all()
        self.status.showMessage("Done", 2000)

    def apply_filters(self):
        df = self.df_full.copy()
        for col, (mn, mx) in self.filters.items():
            try:
                lo = float(mn.text()) if mn.text() else None
                hi = float(mx.text()) if mx.text() else None
                if lo is not None: df = df[df[col] >= lo]
                if hi is not None: df = df[df[col] <= hi]
            except ValueError:
                pass
        self.df = df; self.update_all()

    def reset_filters(self):
        for mn, mx in self.filters.values():
            mn.clear(); mx.clear()
        self.df = self.df_full.copy(); self.update_all()

    def update_all(self):
        if self.df is None or self.df.empty:
            return
        self.update_table()
        self.update_graphs()
        self.update_summary()
        # self.update_optimization()
        self.update_system()
        # MOC functionality removed
        self.update_recommendations()
        self.update_nozzle_design()
        # Grain design tab is self-contained and doesn't need explicit updating

    def update_table(self):
        self.tbl.setModel(PandasModel(self.df))

    def update_graphs(self):
        # build brand‐new figures
        new_figs = create_graphs(self.df)
        for name, new_fig in new_figs.items():
            # Only update canvases that exist
            if name in self.canvases:
                canvas = self.canvases[name]
                # swap out the old Figure for the new one
                canvas.figure = new_fig
                canvas.draw()
                self.figures[name] = new_fig

    def update_summary(self):
        best = self.df.loc[self.df["Isp (s)"].idxmax()]
        html = (
            f"<h2>Summary</h2>"
            f"<p>Max Isp: <b>{best['Isp (s)']:.2f} s</b><br>"
            f"at O/F = <b>{best['O/F']:.2f}</b>, Pc = <b>{best['Pc (bar)']} bar</b></p>"
        )
        self.sum_text.setHtml(html)

    def update_optimization(self):
        # Display a message that optimization feature has been removed
        self.opt_text.setHtml("<h2>Optimization</h2><p>Optimization heatmaps feature has been removed.</p>")

    # update_moc method has been removed

    def update_system(self):
        """
        Compute & display nozzle sketch, thrust vs. altitude,
        prompting once if Expansion Ratio is missing.
        """
        # 1) Find the index of the best‐Isp row
        best_idx = self.df["Isp (s)"].idxmax()

        # 2) Pull that row
        best = self.df.loc[best_idx]

        # 3) Get (or prompt for) Expansion Ratio
        ar = best["Expansion Ratio"]
        if ar is None:
            ar, ok = QInputDialog.getDouble(
                self,
                "Missing Expansion Ratio",
                "Enter nozzle expansion ratio Aₑ/A*:",
                10.0,   # default
                1.0,    # min
                1e4,    # max
                2       # decimals
            )
            if not ok:
                return
            # Write it back into the one row
            self.df.at[best_idx, "Expansion Ratio"] = ar
            best = self.df.loc[best_idx]  # re‐fetch with updated ar

        # 4) Now call compute_system (which will use that ar)
        res = compute_system(self.df)
        At = res["At"]
        Ae = res["Ae"]

        # 5) Plot your nozzle sketch & thrust vs altitude (unchanged)
        fig = self.sys_canvas.figure
        fig.clear()

        ax1 = fig.add_subplot(121)
        x = [0.0, 0.2, 0.5, 1.0, 1.2]
        y = [0.0, 0.6, 1.0, 0.8, 0.8]
        ax1.plot(x, y, lw=2)
        ax1.plot(x, [-yy for yy in y], lw=2)
        ax1.set(aspect='equal', title='Nozzle Sketch', xlabel='Axial', ylabel='Radius')

        ax2 = fig.add_subplot(122)
        alts = res["alts"]
        Fs   = res["Fs"]
        ax2.plot(alts, Fs)
        ax2.set(title='Thrust vs Altitude', xlabel='Altitude (m)', ylabel='Thrust (N)')

        self.sys_canvas.draw()

        # 6) Show key numbers
        html = (
            f"<h2>Nozzle & System</h2>"
            f"<p>At = {At:.6f} m²<br>"
            f"Ae = {Ae:.6f} m²<br>"
            f"Expansion ratio = {ar:.2f}</p>"
        )
        self.sys_text.setHtml(html)


    def update_recommendations(self):
        b = self.df.loc[self.df["Isp (s)"].idxmax()]
        rec = (
            f"<h2>Recommendation</h2>"
            f"<p>Use O/F = {b['O/F']:.2f} at Pc = {b['Pc (bar)']} bar for max Isp.</p>"
        )
        self.reco.setHtml(rec)

    def export_csv(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if fn:
            export_csv(self.df, fn)
            
    def export_excel(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
        if fn:
            # summary as small DataFrame
            summary = pd.DataFrame([self.df.loc[self.df["Isp (s)"].idxmax()]])
            export_excel(self.df, summary, fn)
            
    def export_pdf(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if fn:
            figs = {"Cover": self.figures["Isp"]}
            figs.update(create_graphs(self.df))
            export_pdf(figs, CONFIG["pdf_report_title"], fn)
            
    def update_nozzle_design(self):
        """Update the nozzle design based on current settings"""
        if self.df is None or len(self.df) == 0:
            return
            
        # Get the best case from the dataframe
        best_case = self.df.iloc[self.df['Isp (s)'].idxmax()]
        
        # Get the throat radius from the input field
        try:
            R_throat = float(self.throat_radius_edit.text())
        except ValueError:
            R_throat = 0.05  # Default to 5cm if invalid input
            self.throat_radius_edit.setText("0.05")
        
        # Get the nozzle type
        nozzle_type = self.nozzle_type_combo.currentText()
        
        # Store the nozzle type in the CEA data for performance calculations
        cea_data = best_case.copy()
        cea_data['nozzle_type'] = nozzle_type
        
        # Generate the nozzle contour based on the selected type
        if nozzle_type == "Conical":
            x, r = nozzle.conical_nozzle(cea_data, R_throat=R_throat)
        elif nozzle_type == "Rao Optimum":
            x, r = nozzle.rao_optimum_nozzle(cea_data, R_throat=R_throat)
        elif nozzle_type == "80% Bell":
            x, r = nozzle.bell_nozzle(cea_data, R_throat=R_throat, percent_bell=80)
        elif nozzle_type == "Method of Characteristics (MOC)":
            x, r = nozzle.moc_nozzle(cea_data, R_throat=R_throat)
        elif nozzle_type == "Truncated Ideal Contour (TIC)":
            x, r = nozzle.truncated_ideal_contour(cea_data, R_throat=R_throat, truncation_factor=0.8)
        else:
            # Default to conical if something goes wrong
            x, r = nozzle.conical_nozzle(cea_data, R_throat=R_throat)
        
        # Add inlet section if requested
        if self.include_inlet_checkbox.isChecked():
            x, r = nozzle.add_inlet_section(x, r, R_throat)
        
        # Store the current coordinates for export
        self.current_nozzle_coords = (x, r)
        
        # Calculate performance metrics
        performance = nozzle.calculate_performance(cea_data, (x, r))
        
        # Plot the nozzle with professional engineering styling
        fig = self.nozzle_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Find the actual throat position
        if self.include_inlet_checkbox.isChecked():
            # If inlet is included, throat is at x=0
            throat_idx = np.argmin(np.abs(x))
        else:
            # Otherwise find the minimum radius
            throat_idx = np.argmin(r)
            
        throat_x = x[throat_idx]
        throat_r = r[throat_idx]
        exit_r = r[-1]
        
        # Plot with engineering-standard styling
        # Outer contour (thick blue line)
        ax.plot(x, r, 'b-', lw=2.5)
        ax.plot(x, -r, 'b-', lw=2.5)
        
        # Fill the nozzle with a subtle gradient
        # Create a gradient fill from dark in chamber to light at exit
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('nozzle_gradient', ['#d0d0d0', '#f8f8f8'])
        for i in range(len(x)-1):
            ax.fill_between(x[i:i+2], r[i:i+2], -r[i:i+2], 
                           color=cmap(i/len(x)), alpha=0.7, linewidth=0)
            
        # Add centerline
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, lw=0.5)
        
        # Calculate aspect ratio to make the nozzle look proportional
        length = x[-1] - x[0]
        max_radius = max(r)
        aspect_ratio = length / (max_radius * 2.2)  # Adjust for proper proportions
        ax.set_aspect(aspect_ratio)
        
        # Add key dimension lines and annotations using improved techniques
        # Use a common offset for all dimension lines to maintain consistency
        dimension_gap = length * 0.03  # Gap between contour and dimension line start
        
        # Instead of dividing the throat with a line, use a marker and offset dimension
        # Add throat marker
        ax.plot([throat_x], [0], 'ro', markersize=4)
        
        # Throat radius - use an offset horizontal leader line that doesn't cross the nozzle
        # Draw a small tick at the throat wall
        ax.plot([throat_x-dimension_gap*0.2, throat_x+dimension_gap*0.2], [throat_r, throat_r], 'r-', lw=1)
        # Draw the leader line to the side
        ax.plot([throat_x-dimension_gap*0.2, throat_x-dimension_gap], [throat_r, throat_r], 'r-', lw=1)
        # Add a small vertical tick at the end of the leader line
        ax.plot([throat_x-dimension_gap, throat_x-dimension_gap], [throat_r-dimension_gap*0.2, throat_r+dimension_gap*0.2], 'r-', lw=1)
        # Add the dimension text
        ax.text(throat_x-dimension_gap*1.2, throat_r, f"R$_t$ = {throat_r:.3f}m", 
                verticalalignment='center', horizontalalignment='right',
                fontsize=9, color='darkred', fontweight='bold')
        
        # Exit radius - use the same approach for consistency
        # Draw a small tick at the exit wall
        ax.plot([x[-1]-dimension_gap*0.2, x[-1]+dimension_gap*0.2], [exit_r, exit_r], 'r-', lw=1)
        # Draw the leader line to the side
        ax.plot([x[-1]+dimension_gap*0.2, x[-1]+dimension_gap], [exit_r, exit_r], 'r-', lw=1)
        # Add a small vertical tick at the end of the leader line
        ax.plot([x[-1]+dimension_gap, x[-1]+dimension_gap], [exit_r-dimension_gap*0.2, exit_r+dimension_gap*0.2], 'r-', lw=1)
        # Add the dimension text
        ax.text(x[-1]+dimension_gap*1.2, exit_r, f"R$_e$ = {exit_r:.3f}m", 
                verticalalignment='center', horizontalalignment='left',
                fontsize=9, color='darkred', fontweight='bold')
        
        # Total length with engineering dimension line
        dim_offset = -max_radius * 1.3  # Offset for dimension line
        ax.plot([x[0], x[-1]], [dim_offset, dim_offset], 'r-', lw=1)
        # Add arrow markers
        ax.plot(x[0], dim_offset, 'r<', markersize=5)
        ax.plot(x[-1], dim_offset, 'r>', markersize=5)
        # Add dimension text
        ax.text((x[0]+x[-1])/2, dim_offset * 1.1, f"L = {x[-1]-x[0]:.3f}m", 
                verticalalignment='top', horizontalalignment='center',
                fontsize=9, color='darkred', fontweight='bold')
        
        # Add area ratio annotation with correct calculation
        # Double check area ratio calculation to ensure accuracy
        # Use actual throat and exit area instead of just radius ratio squared
        A_throat = np.pi * throat_r**2
        A_exit = np.pi * exit_r**2
        if A_throat > 0 and not np.isclose(A_throat, 0):
            area_ratio = A_exit / A_throat
        else:
            # Fallback to CEA data area ratio if throat area is zero
            area_ratio = cea_data.get('area_ratio', cea_data.get('Ae/At', 8.0))
        
        # Format area ratio to avoid extremely large values (limit decimal places)
        area_ratio_text = f"{area_ratio:.2f}" if area_ratio < 1000 else f"{area_ratio:.1f}"
        
        # Add the area ratio annotation in a clean box
        ax.text(throat_x + length*0.4, max_radius*0.7, 
                f"Area Ratio (Ae/At) = {area_ratio_text}",
                fontsize=9, color='navy', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', pad=3))
        
        # Plot formatting
        ax.set_title(f"{nozzle_type} Nozzle Design")
        ax.set_xlabel("Axial Distance (m)")
        ax.set_ylabel("Radial Distance (m)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ensure the plot is centered on the axis
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Add a tight layout
        fig.tight_layout()
        
        # Render the figure
        self.nozzle_canvas.draw()
        
        # Update the performance text with more comprehensive metrics
        performance_text = f"""Nozzle Performance Metrics:
        
        Area Ratio (Ae/At): {performance['area_ratio']:.2f}
        Thrust Coefficient (Cf): {performance['thrust_coefficient']:.3f}
        Ideal Thrust Coefficient: {performance['ideal_thrust_coefficient']:.3f}
        Divergence Loss Factor: {performance['divergence_loss_factor']:.3f}
        Divergence Angle: {performance['divergence_angle_deg']:.2f}°
        Nozzle Efficiency: {performance['nozzle_efficiency']:.2%}
        Length to Throat Ratio: {performance['length_to_throat_ratio']:.2f}
        Surface Area: {performance['surface_area']:.4f} m²
        Exit Mach Number: {performance['exit_mach_number']:.2f}
        """
        
        self.nozzle_text.setText(performance_text)
        
        # Render the figure
        self.nozzle_canvas.draw()
        
        # Update performance text
        html = f"""<h2>{nozzle_type} Nozzle Performance</h2>
        <table border='0' cellspacing='5' cellpadding='5'>
            <tr>
                <td><b>Area Ratio (A₍/A*):</b></td>
                <td>{performance['area_ratio']:.2f}</td>
                <td><b>Pressure Ratio:</b></td>
                <td>{performance['pressure_ratio']:.2f}</td>
            </tr>
            <tr>
                <td><b>Thrust Coefficient:</b></td>
                <td>{performance['thrust_coefficient']:.4f}</td>
                <td><b>Nozzle Efficiency:</b></td>
                <td>{performance['nozzle_efficiency']:.2f}</td>
            </tr>
            <tr>
                <td><b>Length/Throat Ratio:</b></td>
                <td>{performance['length_to_throat_ratio']:.2f}</td>
                <td><b>Divergence Loss Factor:</b></td>
                <td>{performance['divergence_loss_factor']:.4f}</td>
            </tr>
        </table>
        <p><small>Based on best performing case: O/F = {best_case['O/F']:.2f}, Pc = {best_case['Pc (bar)']} bar</small></p>
        """
        self.nozzle_text.setHtml(html)
        
    def export_nozzle_coordinates(self):
        """Export nozzle coordinates to a CSV file"""
        if self.df is None or not hasattr(self, 'current_nozzle_coords'):
            return
        
        fname, _ = QFileDialog.getSaveFileName(self, "Export Nozzle Coordinates", "", "CSV Files (*.csv);;Text Files (*.txt)")
        if fname:
            x, r = self.current_nozzle_coords
            success = nozzle.export_nozzle_coordinates(x, r, fname)
            if success:
                self.status.showMessage(f"Nozzle coordinates exported to {fname}", 5000)
            else:
                self.status.showMessage("Error exporting nozzle coordinates", 5000)
