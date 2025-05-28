# Person Counting and Face Recognition System

This project provides a person detection and face recognition system for monitoring students on a bus or in a classroom. The system uses YOLOv8 for person detection and FaceNet for face recognition.

## Students:

23020406 - Nguyễn Phương Nam
23020410 - Nguyễn Trọng Hồng Phúc
23020422 - Nguyễn Đình Quyền
23020404 - Kiều Đức Nam
23020438 - Trần Doãn Thắng

## Features

- Real-time person detection and tracking
- Face recognition of known students
- Student management (add, update, delete)
- History tracking of student entry/exit events
- Web interface for monitoring and management

## 📦 Technologies Used

- **YOLOv8**: For person detection.
- **DeepSORT**: For tracking people across frames.
- **FaceNet (MTCNN + InceptionResnetV1)**: For face recognition.
- **OpenCV**: For video processing.
- **Flask**: For the backend API and video streaming.
- **JSON**: For storing student information and event history.
- **Core Libraries**: NumPy, Torch, Ultralytics, facenet-pytorch, deep-sort-realtime.

## Running Locally

- Python 3.10
- Required Python packages (listed in `requirements.txt`)

## Installation & Running

### Using Docker (Recommended)

1. Pull the Docker image:

   ```bash
   docker pull moimoi05/person_counting_model-person_counter
   ```

2. Download `docker-compose.yml` file and save in a folder:

3. Run the application in terminal:

   ```bash
   docker-compose up -d
   ```

4. Access the web interface at `http://localhost:5000`

### Running Locally

1. Clone the repository
2. Set up a virtual environment:

   ```bash
   # Windows
   python -m venv venv_py310
   venv_py310\Scripts\activate

   # Linux/Mac
   python -m venv venv_py310
   source venv_py310/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`

## Usage

1. Open the web interface at `http://localhost:5000`
2. Add students by uploading their photos
3. Monitor students entering and exiting the bus
4. View history of entry/exit events

## Docker Hub Repository

The Docker image is available at: [Docker Hub - moimoi05/person_counting_model-person_counter](https://hub.docker.com/repository/docker/moimoi05/person_counting_model-person_counter/general)

github: https://github.com/thangtrandoan/thptht_AI
