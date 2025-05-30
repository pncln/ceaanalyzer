# CEA Analyzer

A modern tool for analyzing rocket propulsion systems using NASA-CEA data. This application provides advanced rocket nozzle design, analysis, and visualization capabilities.

## Features

- Parse and analyze NASA-CEA output files
- Interactive data visualization and filtering
- Multiple nozzle design methods:
  - Method of Characteristics (MOC)
  - Rao Optimum Nozzle
  - Conical Nozzle
  - Bell Nozzle
  - Truncated Ideal Contour (TIC)
- Performance calculations and optimization
- Export capabilities (CSV, Excel, PDF)
- Graphical user interface with PyQt5

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cea_analyzer.git
cd cea_analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m cea_analyzer
```

Or run directly:

```bash
python cea_analyzer/main.py
```

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## License

[MIT License](LICENSE)
