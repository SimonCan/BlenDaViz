# Contributing to BlenDaViz

Thank you for your interest in contributing to BlenDaViz! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on [GitHub](https://github.com/SimonCan/BlenDaViz/issues) with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your Blender version and operating system
- Any relevant code snippets or error messages

### Suggesting Features

Feature suggestions are welcome! Please open an issue describing:

- The feature you'd like to see
- Why it would be useful
- Any implementation ideas you have

### Submitting Code Changes

1. **Fork the repository** and create your branch from `master`.

2. **Set up your development environment:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/BlenDaViz.git
   cd BlenDaViz
   ```

3. **Make your changes:**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation if needed

4. **Run the tests:**
   ```bash
   /path/to/blenders/python -m pytest tests/
   ```

5. **Submit a pull request** with a clear description of your changes.

## Code Style

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add docstrings to new functions and classes
- Keep functions focused and modular

## Adding New Plot Types

When adding a new plot type:

1. Create a new class inheriting from `GenericPlot` in `generic.py`
2. Implement the required methods (`plot`, `time_handler` if applicable)
3. Add a convenience function in `__init__.py`
4. Add tests in `tests/test_all.py`
5. Add documentation and examples in `docs/index.rst`

## Testing

- All new features should include tests
- Tests should cover both normal operation and edge cases
- Run the full test suite before submitting a pull request

## Documentation

- Update `docs/index.rst` for user-facing changes
- Include code examples where appropriate
- Keep explanations clear and concise

## Questions?

If you have questions about contributing, feel free to open an issue or contact the maintainers.

Thank you for helping improve BlenDaViz!
