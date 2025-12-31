#!/bin/bash

# Render build script for Tesseract OCR
echo "Installing Tesseract OCR dependencies..."

# Update package list and install dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Add tesseract to PATH
export PATH="/usr/bin:$PATH"

# Verify installation
echo "Tesseract location: $(which tesseract)"
echo "Tesseract version: $(tesseract --version)"

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Build complete with Tesseract OCR support"
