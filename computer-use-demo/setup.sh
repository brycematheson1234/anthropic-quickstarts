#!/bin/bash
PYTHON_MINOR_VERSION=$(python3 --version | awk -F. '{print $2}')

if [ "$PYTHON_MINOR_VERSION" -gt 12 ]; then
    echo "Python version 3.$PYTHON_MINOR_VERSION detected. Python 3.12 or lower is required for setup to complete."
    echo "If you have multiple versions of Python installed, you can set the correct one by adjusting setup.sh to use a specific version, for example:"
    echo "'python3 -m venv .venv' -> 'python3.12 -m venv .venv'"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "Cargo (the package manager for Rust) is not present.  This is required for one of this module's dependencies."
    echo "See https://www.rust-lang.org/tools/install for installation instructions."
    exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r dev-requirements.txt
pre-commit install

# Git repository setup for Docker
echo ""
echo "Setting up git repository cache for Docker..."

# Create directories for Docker volume mounting
mkdir -p ./git-cache ./workspace ./host-files

# Set up git repository cache
# Usage: ./setup.sh [REPO_URL] [BRANCH] [REPO_NAME]
REPO_URL="${1:-${CANVA_REPO_URL:-org-2562356@github.com:Canva/canva.git}}"
BRANCH="${2:-${CANVA_BRANCH:-master}}"
REPO_NAME="${3:-canva}"
CACHE_PATH="./git-cache/$REPO_NAME"

echo "Using repository: $REPO_URL"
echo "Using branch: $BRANCH"
echo "Using repo name: $REPO_NAME"

if [[ -d "$CACHE_PATH/.git" ]]; then
    echo "Found existing git cache, updating..."
    cd "$CACHE_PATH"
    git fetch origin "$BRANCH" --depth=1
    git reset --hard "origin/$BRANCH"
    cd ../..
else
    echo "Performing initial shallow clone..."
    git clone --depth=1 --branch="$BRANCH" --single-branch "$REPO_URL" "$CACHE_PATH"
fi
echo "Git cache ready at $CACHE_PATH"

echo ""
echo "Setup complete! You can now run:"
echo "  docker-compose up"
