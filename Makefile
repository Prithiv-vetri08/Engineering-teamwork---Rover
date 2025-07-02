# Variables
PYTHON=python3
PIP=pip
TESTS=test_Final.py
REQ=requirements.txt

# Install dependencies
install:
    $(PIP) install -r $(REQ)

# Run tests
test: 
    pytest $(TESTS)

# Clean up
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete

# Run all steps
all: install test clean
