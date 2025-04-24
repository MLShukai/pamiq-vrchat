# 🚀 Python UV Template

[![GitHub stars](https://img.shields.io/github/stars/Geson-anko/python-uv-template?style=social)](https://github.com/Geson-anko/python-uv-template/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Main workflow](https://github.com/Geson-anko/python-uv-template/actions/workflows/main.yml/badge.svg)](https://github.com/Geson-anko/python-uv-template/actions/workflows/main.yml)

> ✨ A modern Python project template using UV package manager for blazing fast dependency management

## 📋 Features

- 🐍 Python 3.12+ focused with type hints
- 🧪 Pre-configured pytest with coverage
- 🔍 Static type checking with pyright
- 🧹 Code formatting with ruff
- 🔄 CI/CD with GitHub Actions
- 🐳 Dev container configuration for consistent development
- 📦 UV package management for fast dependency resolution
- 📝 Pre-commit hooks for code quality
- 🏗️ Project structure following best practices

## 🛠️ Quick Start

### Create New Repository

[![Use this template](https://img.shields.io/badge/Use%20this%20template-2ea44f?style=for-the-badge)](https://github.com/new?template_name=python-uv-template&template_owner=Geson-anko)

### Clone and Setup

```bash
# Clone your new repository
git clone https://github.com/yourusername/your-new-repo.git
cd your-new-repo

# Rename the project (Linux/macOS)
./rename.sh your-project-name

# Rename the project (Windows)
.\Rename.ps1 your-project-name

# Setup virtual environment and install dependencies
make venv
```

### Development Tools

```bash
# Run all checks (format, test, type check)
make run

# Format code
make format

# Run tests
make test

# Run type check
make type

# Clean up temporary files
make clean
```

## 📂 Project Structure

```
.
├── .devcontainer/      # Dev container configuration
├── .github/            # GitHub workflows and templates
├── src/
│   └── python_uv_template/  # Source code (will be renamed)
├── tests/              # Test files
├── .pre-commit-config.yaml
├── Makefile            # Development commands
├── pyproject.toml      # Project configuration
├── LICENSE
└── README.md
```

## 🏄‍♂️ Using Dev Container

This project includes a dev container configuration for VSCode or GitHub Codespaces.

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VSCode](https://code.visualstudio.com/)
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VSCode
3. Open the project in VSCode and click "Reopen in Container" when prompted

## 🧩 Dependencies

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) - Modern Python package manager
- Development tools installed automatically via `make venv`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- [UV](https://github.com/astral-sh/uv) for the blazing fast package management
- [ruff](https://github.com/astral-sh/ruff) for the powerful Python linter and formatter
- [pyright](https://github.com/microsoft/pyright) for static type checking
- [pre-commit](https://pre-commit.com/) for git hooks management
