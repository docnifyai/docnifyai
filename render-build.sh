#!/bin/bash
set -e

# Make script executable
chmod +x render-build.sh

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Install Python dependencies
pip install -r requirements.txt

# Create start script with proper PATH
cat > start.sh << 'EOF'
#!/bin/bash
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/tessdata/"
python main.py
EOF

chmod +x start.sh

echo "Build complete - tesseract installed"
