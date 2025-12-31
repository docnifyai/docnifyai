#!/bin/bash

# Install system dependencies and Python packages
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
pip install -r requirements.txt

# Create start script
cat > start.sh << 'EOF'
#!/bin/bash
export PATH="/usr/bin:$PATH"
python main.py
EOF
chmod +x start.sh
