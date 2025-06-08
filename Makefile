# Makefile for Moodle Cheating Detection System
# Provides convenient commands for running the ML pipeline

.PHONY: help setup test quick-start clean install train detect all

# Default target
help:
	@echo "Moodle Cheating Detection System - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup       - Set up virtual environment and install dependencies"
	@echo "  make install     - Install dependencies only (venv must exist)"
	@echo ""
	@echo "Quick Start:"
	@echo "  make quick-start - Run complete pipeline with default settings"
	@echo "  make quick-small - Run quick demo with smaller dataset"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make generate    - Generate synthetic training data"
	@echo "  make preprocess  - Extract features from synthetic data"
	@echo "  make train       - Train ML models"
	@echo "  make detect      - Run detection on processed features"
	@echo ""
	@echo "Testing and Validation:"
	@echo "  make test        - Run comprehensive pipeline test"
	@echo "  make test-quick  - Run quick pipeline test"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Clean generated files and outputs"
	@echo "  make clean-all   - Clean everything including models"
	@echo ""
	@echo "Pipeline:"
	@echo "  make all         - Run complete pipeline (generate → train → detect)"

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up Moodle Cheating Detection System..."
	python -m venv venv
	@echo ""
	@echo "Virtual environment created. To activate:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate   (Windows)"
	@echo ""
	@echo "After activation, run: make install"

# Install dependencies (requires active virtual environment)
install:
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# Quick start - complete pipeline
quick-start:
	@echo "Running quick start pipeline..."
	python quick_start.py

# Quick start with small dataset
quick-small:
	@echo "Running quick start with small dataset..."
	python quick_start.py --small

# Generate synthetic data
generate:
	@echo "Generating synthetic training data..."
	mkdir -p data/synthetic_moodle_logs
	python generate_case.py

# Preprocess data and extract features
preprocess:
	@echo "Preprocessing data and extracting features..."
	mkdir -p preprocessing/artifacts
	python -m preprocessing.main_preprocessor \
		--data_input_dir data/synthetic_moodle_logs \
		--output_feature_path data/processed_artificial_features_V2.csv \
		--train_mode \
		--ground_truth_path data/synthetic_ground_truth.csv

# Train ML models
train:
	@echo "Training machine learning models..."
	mkdir -p results
	python enhanced_model.py

# Run detection
detect:
	@echo "Running cheating detection..."
	mkdir -p detection_results
	# Copy synthetic features to simulate real data detection
	cp data/processed_artificial_features_V2.csv data/processed_real_features_for_detection_V2.csv
	python cheating_detection.py

# Run comprehensive test
test:
	@echo "Running comprehensive pipeline test..."
	python test_pipeline.py

# Run quick test
test-quick:
	@echo "Running quick pipeline test..."
	python test_pipeline.py --quick

# Complete pipeline: generate → preprocess → train → detect
all: generate preprocess train detect
	@echo "✅ Complete pipeline finished successfully!"
	@echo ""
	@echo "Generated outputs:"
	@echo "  📂 data/synthetic_moodle_logs/     - Synthetic Moodle data"
	@echo "  📂 results/                       - Trained models"
	@echo "  📂 detection_results/             - Detection outputs"
	@echo ""
	@echo "Next steps:"
	@echo "  - Check 'cheating_ground_truth.md' for data analysis"
	@echo "  - Review 'results/report_*.txt' for model performance"
	@echo "  - Examine 'detection_results/*.csv' for detection results"

# Clean generated data and outputs (keep models)
clean:
	@echo "Cleaning generated files..."
	rm -rf data/synthetic_moodle_logs/
	rm -f data/processed_artificial_features_V2.csv
	rm -f data/processed_real_features_for_detection_V2.csv
	rm -f data/synthetic_ground_truth.csv
	rm -f cheating_ground_truth.md
	rm -rf detection_results/
	rm -rf preprocessing/artifacts/
	@echo "✅ Cleaned generated files (models preserved)"

# Clean everything including trained models
clean-all: clean
	@echo "Cleaning all outputs including trained models..."
	rm -rf results/
	rm -rf __pycache__/
	rm -rf preprocessing/__pycache__/
	rm -f test_config*.json
	@echo "✅ Cleaned all generated files and models"

# Validate setup
validate:
	@echo "Validating setup..."
	python -c "import sys; print(f'Python: {sys.version}')"
	python -c "import pandas, numpy, sklearn, xgboost; print('✅ All dependencies available')"
	@echo "✅ Setup validation complete"

# Show system information
info:
	@echo "System Information:"
	@echo "=================="
	python -c "import sys, platform; print(f'OS: {platform.system()} {platform.release()}')"
	python -c "import sys; print(f'Python: {sys.version}')"
	@echo ""
	@echo "File Status:"
	@echo "============"
	@test -f generate_case.py && echo "✅ generate_case.py" || echo "❌ generate_case.py"
	@test -f enhanced_model.py && echo "✅ enhanced_model.py" || echo "❌ enhanced_model.py"
	@test -f cheating_detection.py && echo "✅ cheating_detection.py" || echo "❌ cheating_detection.py"
	@test -f requirements.txt && echo "✅ requirements.txt" || echo "❌ requirements.txt"
	@echo ""
	@echo "Directory Status:"
	@echo "================"
	@test -d data && echo "✅ data/" || echo "❌ data/"
	@test -d results && echo "✅ results/" || echo "❌ results/"
	@test -d preprocessing && echo "✅ preprocessing/" || echo "❌ preprocessing/"

# Generate documentation
docs:
	@echo "Opening documentation..."
	@echo "📖 README.md - Main documentation"
	@echo "📄 Bab 3 in draft-skripsi-tex/ - Methodology"
	@echo "🔬 cheating_ground_truth.md - Data analysis (generated after running)"

# Show usage examples
examples:
	@echo "Usage Examples:"
	@echo "==============="
	@echo ""
	@echo "1. First time setup:"
	@echo "   make setup"
	@echo "   source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
	@echo "   make install"
	@echo ""
	@echo "2. Quick demonstration:"
	@echo "   make quick-start"
	@echo ""
	@echo "3. Step by step:"
	@echo "   make generate"
	@echo "   make preprocess"
	@echo "   make train"
	@echo "   make detect"
	@echo ""
	@echo "4. Testing:"
	@echo "   make test-quick"
	@echo ""
	@echo "5. Clean and restart:"
	@echo "   make clean"
	@echo "   make all" 