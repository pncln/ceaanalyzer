# Contributing to CEA Analyzer

Thank you for your interest in contributing to the CEA Analyzer project! This document provides guidelines and information for contributing to the project.

## Development Setup

### Installing for Development

To set up the development environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cea_analyzer.git
   cd cea_analyzer
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

This will install the package in "editable" mode, allowing you to make changes to the code without having to reinstall the package.

### Project Structure

```
cea_analyzer/
├── src/               # Source code directory
├── tests/             # Test directory
├── setup.py           # Package setup script
├── README.md          # Project README
├── requirements.txt   # Package dependencies
└── pytest.ini         # pytest configuration
```

## Running Tests

Run the tests using pytest:

```bash
pytest
```

## Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure your code follows these guidelines.

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run the tests to ensure they pass
5. Submit a pull request

## License

By contributing to CEA Analyzer, you agree that your contributions will be licensed under the project's MIT License.
