#!/bin/bash

# OCR Dependencies Installation Script for Docnify
# This script installs tesseract and poppler for PDF OCR processing

echo "🔧 Installing OCR dependencies for Docnify..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "📱 Detected macOS - using Homebrew"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "📦 Installing tesseract..."
    brew install tesseract
    
    echo "📦 Installing poppler..."
    brew install poppler
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Detected Linux"
    
    # Check for package manager
    if command -v apt-get &> /dev/null; then
        echo "📦 Using apt-get..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
    elif command -v yum &> /dev/null; then
        echo "📦 Using yum..."
        sudo yum install -y tesseract poppler-utils
    elif command -v dnf &> /dev/null; then
        echo "📦 Using dnf..."
        sudo dnf install -y tesseract poppler-utils
    else
        echo "❌ No supported package manager found (apt-get, yum, dnf)"
        exit 1
    fi
    
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "Please install tesseract and poppler manually"
    exit 1
fi

echo "✅ OCR dependencies installed successfully!"
echo ""
echo "🧪 Testing installation..."

# Test tesseract
if command -v tesseract &> /dev/null; then
    echo "✅ tesseract: $(tesseract --version | head -n1)"
else
    echo "❌ tesseract not found in PATH"
    exit 1
fi

# Test poppler (pdf2image dependency)
if command -v pdftoppm &> /dev/null; then
    echo "✅ poppler: pdftoppm available"
else
    echo "❌ poppler not found in PATH"
    exit 1
fi

echo ""
echo "🎉 All OCR dependencies are ready!"
echo "You can now process image-based PDFs with Docnify."
