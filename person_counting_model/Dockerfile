# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.10-dev \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libfontconfig1 \
    git \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- ĐÓNG GÓI YOLOV8N.PT VÀO IMAGE ---
# Đảm bảo file yolov8n.pt nằm trong thư mục gốc của dự án khi build
COPY yolov8n.pt /app/yolov8n.pt
# ------------------------------------

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy ALL application code and directories
COPY app.py .
COPY trigger.py .
COPY run.bat .
COPY API/ ./API/
COPY known_student_faces/ ./known_student_faces/
COPY secrets/ ./secrets/
COPY static/ ./static/
COPY templates/ ./templates/
COPY -p/ ./-p/

# Create necessary directories for runtime
RUN mkdir -p /var/log \
    /tmp

# Set permissions for application files
RUN chmod +x /app/*.py
RUN chmod 755 /etc/supervisor/conf.d/supervisord.conf

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Start supervisor which will run both Python scripts
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]