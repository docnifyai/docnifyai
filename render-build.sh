#!/bin/bash

# Render build script for Tesseract OCR
echo "Installing Tesseract OCR dependencies..."

# Update package list
apt-get update

# Install Tesseract OCR and Poppler utilities
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Set Tesseract path for pytesseract
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Build complete with Tesseract OCR support"
