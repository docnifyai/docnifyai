#!/bin/bash

# Render build script for Tesseract OCR
echo "Installing Tesseract OCR dependencies..."

# Update package list and install dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Create startup script to set PATH
cat > start.sh << 'EOF'
#!/bin/bash
export PATH="/usr/bin:/usr/local/bin:$PATH"
exec python main.py
EOF

chmod +x start.sh

# Install Python dependencies
pip install -r requirements.txt

echo "✅ Build complete with Tesseract OCR support"
