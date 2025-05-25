# Dockerfile
FROM python:3.10-slim
WORKDIR /app

# Cài đặt các gói hệ thống
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- ĐÓNG GÓI CÁC THÀNH PHẦN CỐ ĐỊNH ----
COPY yolov8n.pt /app/yolov8n.pt
# COPY yolov8s.pt /app/yolov8s.pt # Nếu bạn dùng cả model này
COPY students_list.json /app/students_list.json

# Tạo các thư mục cần thiết có thể rỗng ban đầu
RUN mkdir -p /app/API
RUN mkdir -p /app/known_student_faces
# RUN mkdir -p /app/deepface_database_students # Nếu ứng dụng của bạn dùng đến mà không tự tạo

# ---- SAO CHÉP MÃ NGUỒN VÀ TÀI NGUYÊN CÒN LẠI ----
COPY app.py .
COPY templates /app/templates/
COPY static /app/static/
# COPY .dockerignore . # Thường không cần copy .dockerignore vào image

EXPOSE 5000
CMD ["python", "app.py"]
