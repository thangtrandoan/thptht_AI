# Dockerfile for Student Detection System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy YOLO model (fixed component)
COPY yolov8n.pt /app/yolov8n.pt

# Copy default students list (will be overridden by volume mount)
COPY students_list.json /app/students_list.json

# Create necessary directories
RUN mkdir -p /app/API \
    && mkdir -p /app/known_student_faces \
    && mkdir -p /app/templates \
    && mkdir -p /app/static

# Copy application source code and resources
COPY app.py .
COPY templates/ /app/templates/
COPY static/ /app/static/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1

# Run the application
CMD ["python", "app.py"]
