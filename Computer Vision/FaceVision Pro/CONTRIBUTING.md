# Contributing to FaceVision Pro

Thank you for considering contributing! 🎉

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FaceVisionPro.git
   cd FaceVisionPro
   ```
3. **Create a virtual environment** and install dev dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate

   pip install -r requirements-dev.txt
   pip install -e .
   ```
4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes following the code style below.
3. Add or update tests in `tests/`.
4. Run tests:
   ```bash
   pytest tests/ -v
   ```
5. Format your code:
   ```bash
   black .
   isort .
   flake8 .
   ```
6. Commit with a clear message:
   ```bash
   git commit -m "feat: add my feature"
   ```
7. Push and open a Pull Request.

## Code Style

- **Python 3.9+** compatible code only
- **Black** for formatting (line length: 100)
- **isort** for import sorting
- **Type hints** on all public functions
- **Docstrings** on all public classes and methods (Google style)

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring without feature change |
| `chore:` | Build/tooling changes |
| `perf:` | Performance improvement |

## Reporting Issues

Please use the [GitHub Issues](https://github.com/yourusername/FaceVisionPro/issues) page.
Include:
- OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output or screenshots

## License

By contributing, you agree your contributions will be licensed under the [MIT License](LICENSE).
