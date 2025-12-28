#!/bin/bash

# Render build script for Tesseract OCR
echo "Installing Tesseract OCR dependencies..."

# Update package list and install dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Verify installation
echo "Tesseract version:"
tesseract --version
echo "Tesseract location:"
which tesseract

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Build complete with Tesseract OCR support"
