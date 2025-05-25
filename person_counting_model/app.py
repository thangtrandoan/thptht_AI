# app.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time
import os
import json
from facenet_pytorch import InceptionResnetV1, MTCNN # Đã có sẵn
import datetime

# --- CẤU HÌNH CHUNG ---
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.6

HISTORY_JSON_PATH = "API/history.json" # Đường dẫn tới file history.json
MAX_HISTORY_LENGTH_IN_FILE = 500 # Giới hạn số lượng sự kiện trong file để tránh file quá lớn

# Tham số DeepSORT
MAX_AGE_DEEPSORT = 50
N_INIT_DEEPSORT = 3
NMS_MAX_OVERLAP_DEEPSORT = 1.0
MAX_COSINE_DISTANCE = 0.7
NN_BUDGET = None

# Cấu hình Camera và Backend OpenCV
CAMERA_INDEX = 0
OPENCV_VIDEO_CAPTURE_APIPREFERENCE = cv2.CAP_DSHOW
FLIP_CAMERA_HORIZONTAL = True # <--- THÊM CẤU HÌNH LẬT CAMERA

# --- CẤU HÌNH NHẬN DIỆN KHUÔN MẶT TỪ CODE GỐC ---
FACENET_RECOGNITION_THRESHOLD = 0.7 # Ngưỡng từ code gốc, điều chỉnh nếu cần
KNOWN_FACES_DIR = "known_student_faces" # Sử dụng chung
STUDENTS_LIST_JSON_PATH = "students_list.json"

# Cấu hình làm mịn kết quả
RECOGNITION_HISTORY_LENGTH = 5
RECOGNITION_CONFIRM_THRESHOLD = 3

# --- DỮ LIỆU HỌC SINH VÀ TRẠNG THÁI ---
CLASS_STUDENTS_DATA = {}
# CLASS_STUDENTS_DATA sẽ có dạng:
# {
#    "HS001": {"name": "Nguyen Van A", "status": "out",
#              "known_encodings_facenet": [np.array,...], # Encoding từ FaceNet code
#              "original_data": {...}}
# }
students_on_bus_count = 0
strangers_on_bus_count = 0
total_people_on_bus = 0

# --- BIẾN TOÀN CỤC CHO CACHE EMBEDDINGS ---
EMBEDDINGS_CACHE_FILE = "embeddings_cache.npz"
EMBEDDINGS_METADATA_FILE = "embeddings_metadata.json"
embeddings_last_modified = {}  # Lưu thời gian sửa đổi cuối cùng của thư mục ảnh mỗi học sinh
pending_embedding_updates = set()  # Danh sách học sinh cần cập nhật embeddings
last_batch_update_time = 0  # Thời gian cuối cùng cập nhật embeddings theo lô
BATCH_UPDATE_INTERVAL = 300  # Thời gian giữa các lần cập nhật theo lô (5 phút)

# --- BIẾN TOÀN CỤC CHO FLASK VÀ THREAD AI ---
output_frame = None
frame_lock = threading.Lock()

event_history_list = [] # DANH SÁCH MỚI ĐỂ LƯU LỊCH SỬ SỰ KIỆN
MAX_HISTORY_LENGTH = 200 # Giới hạn số lượng sự kiện lưu trữ để tránh đầy bộ nhớ

tracking_data = {
    "students_status": {},
    "students_on_bus_count": 0,
    "strangers_on_bus_count": 0,
    "total_people_on_bus": 0,
    "tracked_objects": [],
    "last_event_message": ""
    # Không cần "event_history" ở đây nữa, sẽ có API riêng
}
data_lock = threading.Lock() # Dùng chung cho tracking_data và event_history_list

# --- KHỞI TẠO FLASK APP ---
app = Flask(__name__)

# --- CÁC HÀM NHẬN DIỆN KHUÔN MẶT TỪ CODE GỐC (ĐÃ SỬA ĐỔI) ---
def initialize_facenet_models_from_original_code():
    global mtcnn_facenet, resnet_facenet, device_facenet
    device_facenet = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'FaceNet models running on device: {device_facenet}')
    try:
        mtcnn_facenet = MTCNN(keep_all=True, device=device_facenet, select_largest=False)
        resnet_facenet = InceptionResnetV1(pretrained='vggface2', device=device_facenet).eval()
        print("MTCNN và ResNet (FaceNet) từ code gốc đã khởi tạo.")
    except Exception as e:
        print(f"LỖI khởi tạo model FaceNet từ code gốc: {e}"); exit()

def get_face_encodings_and_boxes_from_original_code(image_rgb, mtcnn_model, resnet_model, device_):
    """
    Từ code gốc. Phát hiện khuôn mặt, trích xuất bounding boxes và mã hóa chúng.
    """
    boxes, probs = mtcnn_model.detect(image_rgb)
    face_encodings = []
    detected_face_boxes = []
    aligned_face_tensors = None

    if boxes is not None:
        # image_rgb phải là PIL Image hoặc numpy array
        # mtcnn_model.forward() trả về các tensor khuôn mặt đã căn chỉnh
        try:
            aligned_face_tensors = mtcnn_model(image_rgb) # Trả về list các tensor nếu keep_all=True
        except Exception as e:
            # print(f"Lỗi khi gọi mtcnn_model(image_rgb): {e}") # Có thể do ảnh đầu vào không phù hợp
            return [], [], None # Thêm giá trị trả về cho aligned_face_tensors

        if aligned_face_tensors is not None:
            # Nếu aligned_face_tensors là list, cần stack lại thành batch tensor
            if isinstance(aligned_face_tensors, list):
                if not aligned_face_tensors: # List rỗng
                    return [], [], None
                # Lọc bỏ None nếu có
                valid_tensors = [t for t in aligned_face_tensors if t is not None]
                if not valid_tensors:
                    return [], [], None
                try:
                    aligned_face_tensors_batch = torch.stack(valid_tensors).to(device_)
                except RuntimeError as e: # Nếu các tensor có kích thước không đồng nhất
                    # print(f"Lỗi khi stack tensor (kích thước không đồng nhất?): {e}")
                    # Xử lý từng tensor một nếu không stack được
                    temp_encodings = []
                    for tensor_face in valid_tensors:
                        if tensor_face is not None:
                            tensor_face = tensor_face.unsqueeze(0).to(device_) # Thêm batch dim
                            with torch.no_grad():
                                encoding = resnet_model(tensor_face).cpu().numpy().flatten()
                                temp_encodings.append(encoding)
                    # Cần khớp lại encodings với boxes, hơi phức tạp nếu có lỗi này
                    # Tạm thời trả về những gì có
                    face_encodings = temp_encodings
                    # Giả sử detected_face_boxes sẽ có cùng số lượng nếu không có lỗi stack
                    # Điều này cần kiểm tra kỹ nếu lỗi stack xảy ra thường xuyên
                    detected_face_boxes = boxes[:len(face_encodings)].tolist() if boxes is not None else []
                    return face_encodings, detected_face_boxes, aligned_face_tensors # Trả về aligned_face_tensors gốc

            elif isinstance(aligned_face_tensors, torch.Tensor): # Đã là batch tensor
                 aligned_face_tensors_batch = aligned_face_tensors.to(device_)
            else: # Kiểu không mong muốn
                return [], [], None


            with torch.no_grad():
                encodings_batch = resnet_model(aligned_face_tensors_batch).cpu().numpy()

            for i in range(len(boxes)): # Duyệt qua các box gốc
                if i < len(encodings_batch):
                    face_encodings.append(encodings_batch[i].flatten())
                    detected_face_boxes.append(boxes[i])
    return face_encodings, detected_face_boxes, aligned_face_tensors


def load_known_faces_with_facenet_code(known_faces_dir_path, mtcnn_model, resnet_model, device_):
    """
    Sửa đổi từ load_known_faces của code gốc để cập nhật CLASS_STUDENTS_DATA.
    CLASS_STUDENTS_DATA phải được điền bởi load_students_from_json() trước.
    """
    global CLASS_STUDENTS_DATA, embeddings_last_modified
    if not CLASS_STUDENTS_DATA:
        print("CẢNH BÁO: Chưa có dữ liệu học sinh từ JSON. Không thể liên kết ảnh khuôn mặt (FaceNet).")
        return

    if not os.path.exists(known_faces_dir_path):
        print(f"THƯ MỤC KHUÔN MẶT (FaceNet) '{known_faces_dir_path}' KHÔNG TÌM THẤY.")
        return
    
    # Thử tải từ cache trước
    if load_embeddings_cache():
        # Cache đã được tải thành công, không cần xử lý ảnh lại
        return
    
    print(f"Đang tải khuôn mặt học sinh (FaceNet) từ: {known_faces_dir_path}")
    total_encoded_faces = 0
    
    # Lấy thời gian sửa đổi mới nhất của các thư mục
    current_modified_times = get_folder_modified_times()
    
    for student_id_from_folder in os.listdir(known_faces_dir_path):
        person_dir_path = os.path.join(known_faces_dir_path, student_id_from_folder)
        if os.path.isdir(person_dir_path):
            if student_id_from_folder in CLASS_STUDENTS_DATA:
                print(f"  Đang xử lý (FaceNet) cho HS ID: {student_id_from_folder}")
                # Khởi tạo list encoding cho học sinh này nếu chưa có
                if "known_encodings_facenet" not in CLASS_STUDENTS_DATA[student_id_from_folder]:
                    CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"] = []
                else:
                    CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"] = []  # Reset
                
                image_count_for_student = 0
                for image_name in os.listdir(person_dir_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir_path, image_name)
                        known_image = cv2.imread(image_path)
                        if known_image is not None:
                            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
                            # Chỉ cần encoding, không cần box cho known faces
                            encodings, _, _ = get_face_encodings_and_boxes_from_original_code(
                                known_image_rgb, mtcnn_model, resnet_model, device_
                            )
                            if encodings: # Có thể có nhiều mặt trong 1 ảnh, lấy cái đầu tiên
                                CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"].append(encodings[0])
                                image_count_for_student +=1
                                total_encoded_faces +=1
                if image_count_for_student > 0:
                    print(f"    Đã mã hóa {image_count_for_student} khuôn mặt (FaceNet) cho {student_id_from_folder}.")
                    # Lưu thời gian sửa đổi mới nhất cho cache
                    if student_id_from_folder in current_modified_times:
                        embeddings_last_modified[student_id_from_folder] = current_modified_times[student_id_from_folder]
            else:
                print(f"CẢNH BÁO (FaceNet): Thư mục ảnh '{student_id_from_folder}' không khớp ID JSON.")
    
    if total_encoded_faces == 0:
        print("CẢNH BÁO (FaceNet): Không mã hóa được khuôn mặt nào.")
    else:
        print(f"Hoàn tất tải và mã hóa {total_encoded_faces} khuôn mặt (FaceNet).")
        # Lưu cache sau khi đã tải xong
        save_embeddings_cache()
    
    for sid, data in CLASS_STUDENTS_DATA.items():
        if not data.get("known_encodings_facenet"):
            print(f"LƯU Ý (FaceNet): Học sinh {sid} không có ảnh khuôn mặt được mã hóa.")


def recognize_faces_with_facenet_code(current_face_encoding, student_id_to_check, threshold):
    """
    Sửa đổi từ recognize_faces của code gốc.
    Nhận diện MỘT current_face_encoding so với các encoding đã biết của MỘT student_id_to_check.
    Trả về: Tên học sinh (nếu khớp), student_id (nếu khớp), distance nhỏ nhất.
    """
    global CLASS_STUDENTS_DATA
    if student_id_to_check not in CLASS_STUDENTS_DATA or \
       not CLASS_STUDENTS_DATA[student_id_to_check].get("known_encodings_facenet"):
        return "Unknown", None, float('inf') # Không có encoding cho HS này

    known_encodings_for_student = np.array(CLASS_STUDENTS_DATA[student_id_to_check]["known_encodings_facenet"])
    
    if not known_encodings_for_student.size or current_face_encoding is None:
        return "Unknown", None, float('inf')

    distances = np.linalg.norm(known_encodings_for_student - current_face_encoding, axis=1)
    min_distance = np.min(distances)
    
    if min_distance < threshold:
        return CLASS_STUDENTS_DATA[student_id_to_check]["name"], student_id_to_check, min_distance
    
    return "Unknown", None, min_distance


def find_best_match_among_all_students_facenet(current_face_encoding, threshold):
    """
    So sánh một encoding với TẤT CẢ học sinh đã biết.
    Trả về: Tên HS khớp nhất, student_id khớp nhất, distance nhỏ nhất.
    """
    global CLASS_STUDENTS_DATA
    if current_face_encoding is None:
        return "Stranger", None, float('inf')

    best_match_name = "Stranger"
    best_match_student_id = None
    overall_min_distance = float('inf')

    for student_id, data in CLASS_STUDENTS_DATA.items():
        if data.get("known_encodings_facenet"):
            # Sử dụng một phần của recognize_faces_with_facenet_code để lấy distance
            known_encodings_for_student = np.array(data["known_encodings_facenet"])
            distances_to_this_student = np.linalg.norm(known_encodings_for_student - current_face_encoding, axis=1)
            min_dist_to_this_student = np.min(distances_to_this_student)

            if min_dist_to_this_student < overall_min_distance:
                overall_min_distance = min_dist_to_this_student
                # Chưa vội gán best_match_student_id, chỉ gán nếu đạt threshold
                if min_dist_to_this_student < threshold:
                    best_match_name = data["name"]
                    best_match_student_id = student_id
                # else: nếu < overall_min_distance nhưng > threshold, vẫn là Stranger nhưng có distance tốt hơn
                # Điều này giúp phân biệt Stranger "gần giống" và Stranger "hoàn toàn khác"
                # Tuy nhiên, để đơn giản, nếu không đạt threshold thì là Stranger
    
    # Nếu sau khi duyệt hết mà best_match_student_id vẫn là None (tức là không có ai đạt threshold)
    # thì kết quả cuối cùng là Stranger.
    if best_match_student_id is None: # Không có ai đạt ngưỡng
        best_match_name = "Stranger"
        # overall_min_distance đã được cập nhật là khoảng cách nhỏ nhất tới bất kỳ ai

    return best_match_name, best_match_student_id, overall_min_distance


# --- HÀM TẢI DỮ LIỆU HỌC SINH TỪ JSON ---
def load_students_from_json(json_path):
    global CLASS_STUDENTS_DATA
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            students_list = json.load(f)
        
        for student_info in students_list:
            student_id = student_info.get("id")
            if student_id:
                # Sử dụng trực tiếp dữ liệu từ file JSON
                CLASS_STUDENTS_DATA[student_id] = {
                    "name": student_info.get("name", student_id), 
                    "status": student_info.get("status", "out"),  # Lấy trạng thái từ file hoặc mặc định là "out"
                    "known_encodings_facenet": [], # Đổi tên key
                    "original_data": student_info  # Lưu toàn bộ thông tin gốc
                }
            else:
                print(f"CẢNH BÁO: Mục học sinh trong JSON không có ID: {student_info}")
        print(f"Đã tải {len(CLASS_STUDENTS_DATA)} học sinh từ '{json_path}'.")
    except FileNotFoundError:
        print(f"LỖI: File danh sách học sinh '{json_path}' không tìm thấy.")
        CLASS_STUDENTS_DATA = {} # Đảm bảo rỗng nếu lỗi
    except json.JSONDecodeError:
        print(f"LỖI: File '{json_path}' không phải là file JSON hợp lệ.")
        CLASS_STUDENTS_DATA = {}
    except Exception as e:
        print(f"LỖI không xác định khi tải danh sách học sinh từ JSON: {e}")
        CLASS_STUDENTS_DATA = {}

# --- CÁC HÀM HỖ TRỢ LƯU/ĐỌC LỊCH SỬ FILE ---
def load_event_history_from_file():
    """Tải lịch sử sự kiện từ file JSON khi ứng dụng khởi động."""
    global event_history_list
    # Xác định BASE_DIR để đường dẫn file ổn định
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    current_history_json_path = os.path.join(BASE_DIR, HISTORY_JSON_PATH)

    try:
        if os.path.exists(current_history_json_path):
            with open(current_history_json_path, 'r', encoding='utf-8') as f:
                event_history_list = json.load(f)
            print(f"Đã tải {len(event_history_list)} sự kiện từ '{current_history_json_path}'.")
        else:
            print(f"File lịch sử '{current_history_json_path}' không tồn tại. Sẽ tạo mới khi có sự kiện.")
            event_history_list = []
    except json.JSONDecodeError:
        print(f"LỖI: File lịch sử '{current_history_json_path}' không hợp lệ. Tạo danh sách mới.")
        event_history_list = []
    except Exception as e:
        print(f"LỖI không xác định khi tải lịch sử sự kiện từ file: {e}")
        event_history_list = []

def save_event_to_history_file(event_detail):
    """Thêm một sự kiện mới vào danh sách trong bộ nhớ và ghi toàn bộ danh sách vào file."""
    global event_history_list, data_lock, CLASS_STUDENTS_DATA
    # Xác định BASE_DIR để đường dẫn file ổn định
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    current_history_json_path = os.path.join(BASE_DIR, HISTORY_JSON_PATH)
    students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)

    with data_lock: # Bảo vệ cả event_history_list và việc ghi file
        # Thêm vào danh sách trong bộ nhớ trước
        event_history_list.append(event_detail)
        
        # Giữ cho danh sách trong bộ nhớ không quá dài (nếu muốn, hoặc chỉ giới hạn file)
        # if len(event_history_list) > MAX_HISTORY_LENGTH_IN_FILE: # Có thể đặt giới hạn khác cho bộ nhớ
        #     event_history_list.pop(0)

        # Tạo bản sao để ghi, có thể giới hạn số lượng entry trong file
        history_to_write = list(event_history_list)
        if len(history_to_write) > MAX_HISTORY_LENGTH_IN_FILE:
            history_to_write = history_to_write[-MAX_HISTORY_LENGTH_IN_FILE:] # Giữ N mục mới nhất

        # Cập nhật trạng thái học sinh trong file JSON nếu đây là sự kiện của học sinh
        if event_detail.get("type") == "student":
            student_id = event_detail.get("id")
            if student_id and student_id in CLASS_STUDENTS_DATA:
                # Đọc file students_list.json
                try:
                    with open(students_json_path, 'r', encoding='utf-8') as f:
                        students_list_from_file = json.load(f) # Đổi tên biến để tránh nhầm lẫn
                    
                    # Cập nhật trạng thái
                    for student in students_list_from_file:
                        if student.get("id") == student_id:
                            # Cập nhật trạng thái dựa trên hành động
                            if event_detail.get("action") == "Lên xe":
                                student["status"] = "in"
                            elif event_detail.get("action") == "Xuống xe":
                                student["status"] = "out"
                            break
                    
                    # Ghi lại file
                    with open(students_json_path, 'w', encoding='utf-8') as f:
                        json.dump(students_list_from_file, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"LỖI khi cập nhật trạng thái học sinh trong file JSON: {e}")

        try:
            # Đảm bảo thư mục API tồn tại
            os.makedirs(os.path.dirname(current_history_json_path), exist_ok=True)
            with open(current_history_json_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_write, f, ensure_ascii=False, indent=4)
            # print(f"Đã lưu sự kiện vào '{current_history_json_path}'.")
        except Exception as e:
            print(f"LỖI khi lưu sự kiện vào file: {e}")

# --- HÀM KHỞI TẠO CÁC THÀNH PHẦN AI ---
def initialize_ai_components():
    global yolo_model, deepsort_tracker, cap, frame_width, frame_height, counting_line_x
    global mtcnn_facenet, resnet_facenet, device_facenet # Thêm các biến của FaceNet

    # Xác định BASE_DIR để đường dẫn file ổn định
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    current_students_list_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
    current_known_faces_dir = os.path.join(BASE_DIR, KNOWN_FACES_DIR)


    print("Đang khởi tạo các thành phần AI...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"Model YOLO {'đang chạy trên GPU.' if torch.cuda.is_available() else 'đang chạy trên CPU.'}")
    except Exception as e:
        print(f"LỖI tải model YOLO: {e}"); exit()

    # Khởi tạo model FaceNet từ code gốc
    initialize_facenet_models_from_original_code()
    
    # Tải danh sách học sinh từ JSON TRƯỚC
    load_students_from_json(current_students_list_json_path) # Sử dụng đường dẫn đã resolve
    # Sau đó tải ảnh khuôn mặt và liên kết (sử dụng model FaceNet đã khởi tạo)
    load_known_faces_with_facenet_code(current_known_faces_dir, mtcnn_facenet, resnet_facenet, device_facenet)


    cap = cv2.VideoCapture(CAMERA_INDEX, OPENCV_VIDEO_CAPTURE_APIPREFERENCE)
    if not cap.isOpened():
        print(f"LỖI mở camera: {CAMERA_INDEX} với backend {OPENCV_VIDEO_CAPTURE_APIPREFERENCE}")
        print("Đang thử lại với backend mặc định của OpenCV...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened(): print(f"LỖI: Không thể mở camera {CAMERA_INDEX}."); exit()
        else: print("Đã mở camera thành công với backend mặc định.")
    else: print(f"Đã mở camera thành công với backend {OPENCV_VIDEO_CAPTURE_APIPREFERENCE}.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        print("LỖI: Không thể lấy kích thước frame. Gán mặc định 640x480.")
        frame_width, frame_height = 640, 480
    counting_line_x = frame_width // 2
    print(f"Kích thước video: {frame_width}x{frame_height}")

    try:
        deepsort_tracker = DeepSort(
            max_age=MAX_AGE_DEEPSORT, n_init=N_INIT_DEEPSORT,
            nms_max_overlap=NMS_MAX_OVERLAP_DEEPSORT, max_cosine_distance=MAX_COSINE_DISTANCE,
            nn_budget=NN_BUDGET, embedder='mobilenet', embedder_gpu=torch.cuda.is_available()
        )
        print("Tracker DeepSORT đã khởi tạo.")
    except Exception as e:
        print(f"LỖI khởi tạo DeepSORT: {e}"); exit()
    print("Hoàn tất khởi tạo các thành phần AI.")


# --- HÀM CHẠY LOGIC AI VÀ ĐẾM (TRONG THREAD RIÊNG) ---
def process_video_and_count():
    global output_frame, tracking_data, cap, yolo_model, deepsort_tracker
    global frame_width, frame_height, counting_line_x
    global mtcnn_facenet, resnet_facenet, device_facenet, CLASS_STUDENTS_DATA
    global students_on_bus_count, strangers_on_bus_count, total_people_on_bus
    global FLIP_CAMERA_HORIZONTAL # Sử dụng biến cấu hình

    initialize_ai_components()

    tracked_identities = {}
    crossing_status = {}
    previous_positions = {}

    RECOGNITION_ZONE_WIDTH = 100
    RECOGNITION_RETRY_FRAMES_STRANGER = 15
    RECOGNITION_RETRY_FRAMES_LOW_CONF = 25
    
    print(f"Bắt đầu xử lý video trong thread...")
    frame_counter = 0
    last_batch_check_time = time.time()
    
    while True:
        if cap is None or not cap.isOpened(): time.sleep(1); continue
        ret, frame = cap.read()
        if not ret:
            if isinstance(CAMERA_INDEX, str) and cap.isOpened(): # Kiểm tra cap.isOpened() trước khi dùng
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.1); continue
        
        # >>> THAY ĐỔI: Lật camera nếu cấu hình được bật <<<
        if FLIP_CAMERA_HORIZONTAL:
            frame = cv2.flip(frame, 1) 
        # >>> KẾT THÚC THAY ĐỔI <<<
        
        frame_counter += 1
        current_time = time.time()
        
        if current_time - last_batch_check_time > 60:
            process_batch_embedding_updates()
            last_batch_check_time = current_time
        
        yolo_results = yolo_model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections_for_deepsort = []
        person_boxes_from_yolo = []
        if yolo_results and yolo_results[0].boxes:
            for box_data in yolo_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = box_data
                if int(cls_id) == 0: 
                    width, height = int(x2-x1), int(y2-y1)
                    if width >= 20 and height >= 40:
                        detections_for_deepsort.append(
                            ([int(x1), int(y1), width, height], conf, "person")
                        )
                        person_boxes_from_yolo.append([int(x1), int(y1), int(x2), int(y2)])
        
        frame_for_deepsort_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracks = deepsort_tracker.update_tracks(detections_for_deepsort, frame=frame_for_deepsort_rgb)
        
        current_frame_tracked_objects_info = []
        current_frame_tracked_ids = set()
        latest_event_message_this_frame = ""

        frame_for_facenet_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for track in tracks:
            if not track.is_confirmed(): continue
            track_id_str = str(track.track_id)
            current_frame_tracked_ids.add(track_id_str)
            x1_trk, y1_trk, x2_trk, y2_trk = map(int, track.to_ltrb())
            center_x = (x1_trk + x2_trk) // 2

            if track_id_str not in tracked_identities:
                tracked_identities[track_id_str] = {
                    "name_history": [], "confirmed_name": "Processing...", "confirmed_student_id": None,
                    "last_recognized_frame": 0, "last_known_score": 0.0
                }
            track_info = tracked_identities[track_id_str]
            
            should_recognize_again = False
            is_near_counting_line = abs(center_x - counting_line_x) < RECOGNITION_ZONE_WIDTH
            if not track_info["name_history"]: should_recognize_again = True
            elif track_info["confirmed_student_id"] is None and \
                 (frame_counter - track_info["last_recognized_frame"] > RECOGNITION_RETRY_FRAMES_STRANGER):
                should_recognize_again = True
            elif track_info["confirmed_student_id"] is not None and \
                 track_info["last_known_score"] < (1.0 - FACENET_RECOGNITION_THRESHOLD + 0.1) and \
                 (frame_counter - track_info["last_recognized_frame"] > RECOGNITION_RETRY_FRAMES_LOW_CONF):
                should_recognize_again = True
            elif is_near_counting_line and not track_info["confirmed_student_id"] and \
                 (frame_counter - track_info["last_recognized_frame"] > 5):
                 should_recognize_again = True


            if should_recognize_again:
                person_crop_rgb = frame_for_facenet_rgb[max(0,y1_trk):min(frame_height,y2_trk), 
                                                        max(0,x1_trk):min(frame_width,x2_trk)]
                
                if person_crop_rgb.size > 0:
                    face_encodings_in_person, face_boxes_in_person, _ = \
                        get_face_encodings_and_boxes_from_original_code(
                            person_crop_rgb, mtcnn_facenet, resnet_facenet, device_facenet
                        )

                    current_recognized_name = "Stranger"
                    current_student_id = None
                    current_min_distance = float('inf')

                    if face_encodings_in_person:
                        target_encoding = face_encodings_in_person[0]
                        name, sid, dist = find_best_match_among_all_students_facenet(
                            target_encoding, FACENET_RECOGNITION_THRESHOLD
                        )
                        current_recognized_name = name
                        current_student_id = sid
                        current_min_distance = dist

                    current_score = 1.0 - current_min_distance if current_min_distance != float('inf') else 0.0
                    track_info["name_history"].append({
                        "name": current_recognized_name, "student_id": current_student_id, "score": current_score
                    })
                    if len(track_info["name_history"]) > RECOGNITION_HISTORY_LENGTH:
                        track_info["name_history"].pop(0)
                    
                    track_info["last_recognized_frame"] = frame_counter
                    track_info["last_known_score"] = current_score

                    if track_info["name_history"]:
                        id_counts = {}
                        name_counts = {}
                        for rec in track_info["name_history"]:
                            if rec["student_id"]: id_counts[rec["student_id"]] = id_counts.get(rec["student_id"], 0) + 1
                            else: name_counts[rec["name"]] = name_counts.get(rec["name"], 0) + 1
                        
                        best_id_from_history = None
                        max_id_count = 0
                        for stud_id, count in id_counts.items():
                            if count > max_id_count: max_id_count = count; best_id_from_history = stud_id
                        
                        if best_id_from_history and max_id_count >= RECOGNITION_CONFIRM_THRESHOLD:
                            track_info["confirmed_name"] = CLASS_STUDENTS_DATA[best_id_from_history]["name"]
                            track_info["confirmed_student_id"] = best_id_from_history
                        elif name_counts.get("Stranger", 0) >= RECOGNITION_CONFIRM_THRESHOLD or not best_id_from_history:
                            track_info["confirmed_name"] = "Stranger"; track_info["confirmed_student_id"] = None
            
            final_recognized_name = track_info["confirmed_name"]
            final_student_id = track_info["confirmed_student_id"]
            event_details_for_history = None

            if track_id_str in previous_positions:
                prev_center_x, _ = previous_positions[track_id_str]
                
                # >>> THAY ĐỔI: Điều chỉnh logic đếm người khi camera bị lật ngang <<<
                # Hành động dựa trên hướng di chuyển VẬT LÝ.
                # Nếu FLIP_CAMERA_HORIZONTAL = True, thì:
                #   - Di chuyển Trái -> Phải trên frame ĐÃ LẬT tương ứng với XUỐNG XE vật lý (nếu hiệu ứng gương).
                #   - Di chuyển Phải -> Trái trên frame ĐÃ LẬT tương ứng với LÊN XE vật lý (nếu hiệu ứng gương).
                # Nếu FLIP_CAMERA_HORIZONTAL = False, logic như cũ:
                #   - Di chuyển Trái -> Phải trên frame GỐC tương ứng với LÊN XE vật lý.
                #   - Di chuyển Phải -> Trái trên frame GỐC tương ứng với XUỐNG XE vật lý.

                # Điều kiện: Di chuyển từ Trái -> Phải TRÊN FRAME HIỆN TẠI
                if prev_center_x < counting_line_x and center_x >= counting_line_x and \
                   crossing_status.get(track_id_str) != "crossed_LtoR":
                    crossing_status[track_id_str] = "crossed_LtoR"
                    action = "Xuống xe" if FLIP_CAMERA_HORIZONTAL else "Lên xe" # Đảo ngược nếu lật
                    
                    if action == "Lên xe":
                        if final_student_id:
                            if CLASS_STUDENTS_DATA[final_student_id]["status"] == "out":
                                CLASS_STUDENTS_DATA[final_student_id]["status"] = "in"; students_on_bus_count += 1
                                latest_event_message_this_frame = f"HS {final_recognized_name} Lên xe."
                                event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Lên xe"}
                        elif final_recognized_name == "Stranger":
                            strangers_on_bus_count += 1
                            latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Lên xe."
                            event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Lên xe"}
                    elif action == "Xuống xe":
                        if final_student_id:
                            if CLASS_STUDENTS_DATA[final_student_id]["status"] == "in":
                                CLASS_STUDENTS_DATA[final_student_id]["status"] = "out"; students_on_bus_count -= 1
                                latest_event_message_this_frame = f"HS {final_recognized_name} Xuống xe."
                                event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Xuống xe"}
                        elif final_recognized_name == "Stranger":
                            if strangers_on_bus_count > 0: strangers_on_bus_count -= 1
                            latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Xuống xe."
                            event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Xuống xe"}
                    
                    if latest_event_message_this_frame: print(f"EVENT: {latest_event_message_this_frame}")
                    total_people_on_bus = students_on_bus_count + strangers_on_bus_count

                # Điều kiện: Di chuyển từ Phải -> Trái TRÊN FRAME HIỆN TẠI
                elif prev_center_x >= counting_line_x and center_x < counting_line_x and \
                     crossing_status.get(track_id_str) != "crossed_RtoL":
                    crossing_status[track_id_str] = "crossed_RtoL"
                    action = "Lên xe" if FLIP_CAMERA_HORIZONTAL else "Xuống xe" # Đảo ngược nếu lật

                    if action == "Lên xe":
                        if final_student_id:
                            if CLASS_STUDENTS_DATA[final_student_id]["status"] == "out":
                                CLASS_STUDENTS_DATA[final_student_id]["status"] = "in"; students_on_bus_count += 1
                                latest_event_message_this_frame = f"HS {final_recognized_name} Lên xe."
                                event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Lên xe"}
                        elif final_recognized_name == "Stranger":
                            strangers_on_bus_count += 1
                            latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Lên xe."
                            event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Lên xe"}
                    elif action == "Xuống xe":
                        if final_student_id:
                            if CLASS_STUDENTS_DATA[final_student_id]["status"] == "in":
                                CLASS_STUDENTS_DATA[final_student_id]["status"] = "out"; students_on_bus_count -= 1
                                latest_event_message_this_frame = f"HS {final_recognized_name} Xuống xe."
                                event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Xuống xe"}
                        elif final_recognized_name == "Stranger":
                            if strangers_on_bus_count > 0: strangers_on_bus_count -= 1
                            latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Xuống xe."
                            event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Xuống xe"}

                    if latest_event_message_this_frame: print(f"EVENT: {latest_event_message_this_frame}")
                    total_people_on_bus = students_on_bus_count + strangers_on_bus_count
                # >>> KẾT THÚC THAY ĐỔI LOGIC ĐẾM <<<
                
                # Đảm bảo số đếm không âm
                students_on_bus_count = max(0, students_on_bus_count)
                strangers_on_bus_count = max(0, strangers_on_bus_count)
                total_people_on_bus = max(0, total_people_on_bus)

                if event_details_for_history:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    event_details_for_history["timestamp"] = timestamp
                    save_event_to_history_file(event_details_for_history)

            previous_positions[track_id_str] = (center_x, (y1_trk + y2_trk) // 2)
            if abs(center_x - counting_line_x) > RECOGNITION_ZONE_WIDTH / 2 and \
               (crossing_status.get(track_id_str) == "crossed_LtoR" or crossing_status.get(track_id_str) == "crossed_RtoL"):
                crossing_status[track_id_str] = "approaching"
            
            display_name_on_frame = f"{final_recognized_name} (ID:{track_id_str.split('-')[0]})"
            color_on_frame = (0, 255, 0) if final_student_id else ((0, 0, 255) if final_recognized_name == "Stranger" else (255,165,0) )
            cv2.rectangle(frame, (x1_trk, y1_trk), (x2_trk, y2_trk), color_on_frame, 2)
            cv2.putText(frame, display_name_on_frame, (x1_trk, y1_trk - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_on_frame, 2)

            current_frame_tracked_objects_info.append({
                'id': track_id_str, 'bbox': [x1_trk, y1_trk, x2_trk, y2_trk],
                'name': final_recognized_name, 'student_id': final_student_id
            })

        ids_to_remove = set(previous_positions.keys()) - current_frame_tracked_ids
        for old_id in ids_to_remove:
            if old_id in previous_positions: del previous_positions[old_id]
            if old_id in tracked_identities: del tracked_identities[old_id]
            if old_id in crossing_status: del crossing_status[old_id]

        cv2.line(frame, (counting_line_x, 0), (counting_line_x, frame_height), (0, 255, 255), 2)
        cv2.putText(frame, f"HS on bus: {students_on_bus_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Strangers: {strangers_on_bus_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total: {total_people_on_bus}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if latest_event_message_this_frame:
             cv2.putText(frame, latest_event_message_this_frame, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        with frame_lock: output_frame = frame.copy()
        with data_lock:
            tracking_data["students_status"] = {
                sid: {"name": data["name"], "status": data["status"], "original_data": data.get("original_data", {})}
                for sid, data in CLASS_STUDENTS_DATA.items()
            }
            tracking_data["students_on_bus_count"] = students_on_bus_count
            tracking_data["strangers_on_bus_count"] = strangers_on_bus_count
            tracking_data["total_people_on_bus"] = total_people_on_bus
            tracking_data["tracked_objects"] = current_frame_tracked_objects_info
            if latest_event_message_this_frame: tracking_data["last_event_message"] = latest_event_message_this_frame


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames_for_stream():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                current_display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(current_display_frame, "AI Initializing...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                current_display_frame = output_frame.copy()
        
        (flag, encodedImage) = cv2.imencode(".jpg", current_display_frame)
        if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03) # Giảm độ trễ một chút

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_for_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/student_management_info')
def student_management_info():
    global tracking_data, data_lock
    with data_lock:
        data_to_send = json.loads(json.dumps(tracking_data))
    return jsonify(data_to_send)

@app.route('/api/students', methods=['GET'])
def get_all_students_status():
    global CLASS_STUDENTS_DATA, data_lock
    
    students_list_with_status = []
    with data_lock:
        for student_id, data in CLASS_STUDENTS_DATA.items():
            students_list_with_status.append({
                "id": student_id,
                "name": data["name"],
                "status": data["status"],
                "has_face_encodings": True if data.get("known_encodings_facenet") and len(data["known_encodings_facenet"]) > 0 else False, # Sửa key
                "original_data": data.get("original_data", {})
            })
    return jsonify(students_list_with_status)

@app.route('/api/event_history', methods=['GET'])
def get_event_history():
    global event_history_list, data_lock
    
    with data_lock:
        # Trả về một bản sao để tránh thay đổi list gốc nếu có xử lý thêm
        events_to_return = list(event_history_list) 
    return jsonify(events_to_return)

@app.route('/api/add_student', methods=['POST'])
def add_student():
    try:
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        student_class = request.form.get('class')
        age = request.form.get('age')
        address = request.form.get('address')
        father_name = request.form.get('father_name')
        father_age = request.form.get('father_age')
        father_phone = request.form.get('father_phone')
        mother_name = request.form.get('mother_name')
        mother_age = request.form.get('mother_age')
        mother_phone = request.form.get('mother_phone')
        
        if not student_id or not name:
            return jsonify({"success": False, "message": "Thiếu thông tin bắt buộc (ID hoặc tên học sinh)"}), 400
        
        student_data = {
            "id": student_id, "name": name, "class": student_class, "age": age,
            "address": address, "father_name": father_name, "father_age": father_age,
            "father_phone": father_phone, "mother_name": mother_name, "mother_age": mother_age,
            "mother_phone": mother_phone, "status": "out"
        }
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
        
        existing_students = []
        if os.path.exists(students_json_path):
            try:
                with open(students_json_path, 'r', encoding='utf-8') as f:
                    existing_students = json.load(f)
            except json.JSONDecodeError:
                existing_students = [] # Nếu file lỗi, bắt đầu với list rỗng

        student_exists = False
        for i, s in enumerate(existing_students):
            if s.get('id') == student_id:
                existing_students[i] = student_data # Cập nhật nếu ID đã tồn tại
                student_exists = True
                break
        if not student_exists:
            existing_students.append(student_data)
        
        with open(students_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_students, f, ensure_ascii=False, indent=4)
        
        images = request.files.getlist('images')
        has_new_images = False
        if images:
            student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id)
            os.makedirs(student_folder, exist_ok=True)
            for i, image in enumerate(images):
                if image and image.filename:
                    has_new_images = True
                    filename = secure_filename(image.filename)
                    base, ext = os.path.splitext(filename)
                    # Tạo tên file duy nhất để tránh ghi đè
                    unique_filename = f"{base}_{int(time.time())}_{i}{ext}"
                    image_path = os.path.join(student_folder, unique_filename)
                    image.save(image_path)
        
        global CLASS_STUDENTS_DATA, data_lock
        with data_lock:
            # Giữ lại encodings cũ nếu có, chờ cập nhật
            old_encodings = CLASS_STUDENTS_DATA.get(student_id, {}).get("known_encodings_facenet", [])
            CLASS_STUDENTS_DATA[student_id] = {
                "name": name, "status": "out",
                "original_data": student_data,
                "known_encodings_facenet": old_encodings 
            }
        
        if has_new_images:
            schedule_student_for_update(student_id)
        
        return jsonify({"success": True, "message": "Thêm/Cập nhật học sinh thành công", "student_id": student_id})
    
    except Exception as e:
        print(f"Lỗi khi thêm/cập nhật học sinh: {e}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"}), 500

@app.route('/api/students/<student_id>', methods=['GET'])
def get_student_details(student_id):
    global CLASS_STUDENTS_DATA, data_lock
    with data_lock:
        if student_id in CLASS_STUDENTS_DATA:
            student_data = CLASS_STUDENTS_DATA[student_id]
            return jsonify({
                "id": student_id,
                "name": student_data["name"],
                "status": student_data["status"],
                "original_data": student_data.get("original_data", {})
            })
        else:
            # Thử đọc từ file JSON nếu không có trong bộ nhớ (ít xảy ra nếu load_students_from_json chạy đúng)
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
            if os.path.exists(students_json_path):
                try:
                    with open(students_json_path, 'r', encoding='utf-8') as f:
                        students_list = json.load(f)
                    for s_data in students_list:
                        if s_data.get("id") == student_id:
                             return jsonify({
                                "id": student_id,
                                "name": s_data.get("name"),
                                "status": s_data.get("status", "out"), # Lấy status từ file
                                "original_data": s_data
                            })
                except Exception:
                    pass # Bỏ qua nếu đọc file lỗi
            return jsonify({"error": "Không tìm thấy học sinh"}), 404

@app.route('/api/students/<student_id>', methods=['PUT'])
def update_student_details(student_id):
    try:
        data = request.json
        name = data.get('name')
        student_class = data.get('class')
        age = data.get('age')
        address = data.get('address')
        father_name = data.get('father_name')
        father_age = data.get('father_age')
        father_phone = data.get('father_phone')
        mother_name = data.get('mother_name')
        mother_age = data.get('mother_age')
        mother_phone = data.get('mother_phone')
        
        if not name: # ID không đổi, chỉ cần tên
            return jsonify({"success": False, "message": "Thiếu tên học sinh"}), 400
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
        
        if not os.path.exists(students_json_path):
            return jsonify({"success": False, "message": "Không tìm thấy file danh sách học sinh"}), 500

        try:
            with open(students_json_path, 'r', encoding='utf-8') as f:
                students_list = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "Lỗi đọc file danh sách học sinh"}), 500
        
        student_found = False
        for i, student in enumerate(students_list):
            if student.get('id') == student_id:
                current_status = student.get('status', 'out') # Giữ trạng thái hiện tại
                students_list[i] = {
                    "id": student_id, "name": name, "class": student_class, "age": age,
                    "address": address, "father_name": father_name, "father_age": father_age,
                    "father_phone": father_phone, "mother_name": mother_name, "mother_age": mother_age,
                    "mother_phone": mother_phone, "status": current_status
                }
                student_found = True
                break
        
        if not student_found:
            return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id}"}), 404
        
        with open(students_json_path, 'w', encoding='utf-8') as f:
            json.dump(students_list, f, ensure_ascii=False, indent=4)
        
        global CLASS_STUDENTS_DATA, data_lock
        updated_student_original_data = { # Dữ liệu đầy đủ để lưu vào original_data
            "id": student_id, "name": name, "class": student_class, "age": age,
            "address": address, "father_name": father_name, "father_age": father_age,
            "father_phone": father_phone, "mother_name": mother_name, "mother_age": mother_age,
            "mother_phone": mother_phone
        }
        with data_lock:
            if student_id in CLASS_STUDENTS_DATA:
                current_status_mem = CLASS_STUDENTS_DATA[student_id].get("status", "out")
                current_encodings_mem = CLASS_STUDENTS_DATA[student_id].get("known_encodings_facenet", [])
                CLASS_STUDENTS_DATA[student_id] = {
                    "name": name,
                    "status": current_status_mem,
                    "known_encodings_facenet": current_encodings_mem,
                    "original_data": updated_student_original_data # Cập nhật original_data
                }
        
        return jsonify({"success": True, "message": "Cập nhật thông tin học sinh thành công"})
    
    except Exception as e:
        print(f"Lỗi khi cập nhật thông tin học sinh: {e}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"}), 500

@app.route('/api/students/<student_id>/avatar', methods=['POST'])
def update_student_avatar(student_id):
    try:
        if 'avatar' not in request.files:
            return jsonify({"success": False, "message": "Không có file ảnh được gửi lên"}), 400
        
        avatar_file = request.files['avatar']
        if not avatar_file or not avatar_file.filename:
            return jsonify({"success": False, "message": "File ảnh không hợp lệ"}), 400
        
        global CLASS_STUDENTS_DATA # Không cần data_lock ở đây vì chỉ kiểm tra sự tồn tại
        if student_id not in CLASS_STUDENTS_DATA:
             # Kiểm tra thêm trong file JSON nếu không có trong bộ nhớ
            BASE_DIR_CHECK = os.path.dirname(os.path.abspath(__file__))
            students_json_path_check = os.path.join(BASE_DIR_CHECK, STUDENTS_LIST_JSON_PATH)
            student_in_json = False
            if os.path.exists(students_json_path_check):
                try:
                    with open(students_json_path_check, 'r', encoding='utf-8') as f_check:
                        students_list_check = json.load(f_check)
                    if any(s.get('id') == student_id for s in students_list_check):
                        student_in_json = True
                except Exception:
                    pass # Bỏ qua lỗi đọc file
            if not student_in_json:
                 return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id}"}), 404
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id)
        os.makedirs(student_folder, exist_ok=True)
        
        filename = secure_filename(avatar_file.filename)
        name_part, ext_part = os.path.splitext(filename)
        # Tạo tên file duy nhất để tránh ghi đè và dễ quản lý hơn
        unique_filename = f"{name_part}_{int(time.time())}{ext_part}"
        avatar_path = os.path.join(student_folder, unique_filename)
        avatar_file.save(avatar_path)
        
        schedule_student_for_update(student_id) # Lên lịch cập nhật embeddings
        
        return jsonify({
            "success": True, 
            "message": "Cập nhật ảnh đại diện thành công. Embeddings sẽ được cập nhật.",
            "image_url": f"/api/student_image/{student_id}?t={int(time.time())}"
        })
    
    except Exception as e:
        print(f"Lỗi khi cập nhật ảnh đại diện: {e}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"}), 500

@app.route('/api/student_image/<student_id>', methods=['GET'])
def get_student_image(student_id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id)
    default_avatar_path = os.path.join(BASE_DIR, 'static')
    default_avatar_name = 'default_avatar.png'

    if not os.path.exists(student_folder) or not os.path.isdir(student_folder):
        return send_from_directory(default_avatar_path, default_avatar_name)
    
    image_files = [f for f in os.listdir(student_folder) 
                  if os.path.isfile(os.path.join(student_folder, f)) 
                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))] # Bỏ .gif nếu không hỗ trợ
    
    if not image_files:
        return send_from_directory(default_avatar_path, default_avatar_name)
    
    image_files_with_time = [(f, os.path.getmtime(os.path.join(student_folder, f))) for f in image_files] # Sửa getctime thành getmtime
    image_files_with_time.sort(key=lambda x: x[1], reverse=True)
    newest_image = image_files_with_time[0][0]
    
    return send_from_directory(student_folder, newest_image)

@app.route('/api/history.json', methods=['GET']) # API để phục vụ file history.json nếu cần
def serve_history_json():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    history_file_path = os.path.join(BASE_DIR, HISTORY_JSON_PATH)
    if os.path.exists(history_file_path):
        return send_from_directory(os.path.dirname(history_file_path), os.path.basename(history_file_path))
    else:
        return jsonify({"error": "File history.json không tìm thấy."}), 404

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
        
        if not os.path.exists(students_json_path):
             return jsonify({"success": False, "message": "Không tìm thấy file danh sách học sinh"}), 500
        try:
            with open(students_json_path, 'r', encoding='utf-8') as f:
                students_list = json.load(f)
        except json.JSONDecodeError:
             return jsonify({"success": False, "message": "Lỗi đọc file danh sách học sinh"}), 500
        
        original_len = len(students_list)
        students_list = [s for s in students_list if s.get('id') != student_id]
        
        if len(students_list) == original_len:
            return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id} để xóa"}), 404
        
        with open(students_json_path, 'w', encoding='utf-8') as f:
            json.dump(students_list, f, ensure_ascii=False, indent=4)
        
        student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id)
        if os.path.exists(student_folder) and os.path.isdir(student_folder):
            import shutil
            try:
                shutil.rmtree(student_folder) # Xóa thư mục và tất cả nội dung bên trong
                print(f"Đã xóa thư mục ảnh: {student_folder}")
            except Exception as e:
                print(f"Lỗi khi xóa thư mục {student_folder}: {e}")
        
        global CLASS_STUDENTS_DATA, data_lock, embeddings_last_modified, pending_embedding_updates
        with data_lock:
            if student_id in CLASS_STUDENTS_DATA:
                del CLASS_STUDENTS_DATA[student_id]
            if student_id in embeddings_last_modified:
                del embeddings_last_modified[student_id]
            if student_id in pending_embedding_updates:
                pending_embedding_updates.remove(student_id)
        
        # Sau khi xóa, nên lưu lại cache embeddings
        save_embeddings_cache()

        return jsonify({"success": True, "message": "Đã xóa học sinh thành công"})
        
    except Exception as e:
        print(f"Lỗi khi xóa học sinh: {e}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"}), 500

# --- CÁC HÀM CẬP NHẬT EMBEDDING MỚI ---
def update_single_student_embeddings(student_id, mtcnn_model, resnet_model, device_):
    global CLASS_STUDENTS_DATA, embeddings_last_modified
    
    if student_id not in CLASS_STUDENTS_DATA:
        print(f"CẢNH BÁO (update_single): Không tìm thấy HS ID {student_id} trong CLASS_STUDENTS_DATA.")
        # Kiểm tra xem có nên tải lại thông tin học sinh từ JSON không nếu nó không có trong bộ nhớ
        # Điều này có thể xảy ra nếu học sinh được thêm vào JSON nhưng app chưa restart
        BASE_DIR_TEMP = os.path.dirname(os.path.abspath(__file__))
        students_json_path_temp = os.path.join(BASE_DIR_TEMP, STUDENTS_LIST_JSON_PATH)
        try:
            with open(students_json_path_temp, 'r', encoding='utf-8') as f_temp:
                all_students_json = json.load(f_temp)
            student_info_json = next((s for s in all_students_json if s.get("id") == student_id), None)
            if student_info_json:
                print(f"Tìm thấy HS ID {student_id} trong JSON, đang thêm vào CLASS_STUDENTS_DATA.")
                CLASS_STUDENTS_DATA[student_id] = {
                    "name": student_info_json.get("name", student_id),
                    "status": student_info_json.get("status", "out"),
                    "known_encodings_facenet": [],
                    "original_data": student_info_json
                }
            else: # Vẫn không tìm thấy
                return False
        except Exception as e_json:
            print(f"Lỗi khi thử tải lại HS ID {student_id} từ JSON: {e_json}")
            return False

    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    person_dir_path = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id)
    
    if not os.path.exists(person_dir_path) or not os.path.isdir(person_dir_path):
        print(f"CẢNH BÁO: Không tìm thấy thư mục ảnh cho học sinh ID {student_id} tại {person_dir_path}. Sẽ xóa encodings cũ (nếu có).")
        CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"] = []
        if student_id in embeddings_last_modified:
            del embeddings_last_modified[student_id] # Xóa khỏi metadata
        save_embeddings_cache() # Lưu lại cache sau khi xóa
        return False # Không có ảnh để xử lý
    
    print(f"Đang cập nhật embeddings (FaceNet) cho học sinh ID: {student_id}")
    
    # Reset encodings cho học sinh này
    CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"] = []
    
    latest_mod_time = 0
    image_count = 0
    
    for image_name in os.listdir(person_dir_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(person_dir_path, image_name)
            mod_time = os.path.getmtime(image_path)
            if mod_time > latest_mod_time:
                latest_mod_time = mod_time
                
            known_image = cv2.imread(image_path)
            if known_image is not None:
                known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
                encodings, _, _ = get_face_encodings_and_boxes_from_original_code(
                    known_image_rgb, mtcnn_model, resnet_model, device_
                )
                if encodings:
                    CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"].append(encodings[0])
                    image_count += 1
    
    if image_count > 0:
        print(f"Đã cập nhật {image_count} khuôn mặt (FaceNet) cho học sinh ID {student_id}.")
        if latest_mod_time > 0:
            embeddings_last_modified[student_id] = latest_mod_time
        # Không cần save_embeddings_cache() ở đây, hàm process_batch_embedding_updates sẽ gọi
        return True
    else:
        print(f"CẢNH BÁO: Không tìm thấy khuôn mặt nào trong ảnh của học sinh ID {student_id}. Encodings sẽ rỗng.")
        # Vẫn cập nhật thời gian sửa đổi để cache biết là đã xử lý thư mục này
        if latest_mod_time > 0:
             embeddings_last_modified[student_id] = latest_mod_time
        else: # Nếu thư mục rỗng, lấy thời gian sửa đổi của chính thư mục
             embeddings_last_modified[student_id] = os.path.getmtime(person_dir_path)

        return False # Trả về False vì không có encoding nào được tạo

def process_batch_embedding_updates():
    global pending_embedding_updates, last_batch_update_time, mtcnn_facenet, resnet_facenet, device_facenet
    
    if not pending_embedding_updates:
        return
    
    current_time = time.time()
    if current_time - last_batch_update_time < BATCH_UPDATE_INTERVAL and last_batch_update_time != 0 : # Cho phép chạy lần đầu tiên ngay
        return
    
    print(f"Bắt đầu cập nhật embeddings theo lô cho {len(pending_embedding_updates)} học sinh: {list(pending_embedding_updates)}")
    
    students_to_process_now = list(pending_embedding_updates) # Xử lý những gì đang có
    pending_embedding_updates.clear() # Xóa danh sách chờ hiện tại, các yêu cầu mới sẽ được thêm vào sau
    
    updated_any_student = False
    for student_id in students_to_process_now:
        if update_single_student_embeddings(student_id, mtcnn_facenet, resnet_facenet, device_facenet):
            updated_any_student = True # Chỉ cần một học sinh được cập nhật thành công
    
    last_batch_update_time = current_time
    
    if updated_any_student or students_to_process_now: # Lưu cache ngay cả khi không có encoding nào được tạo (để cập nhật metadata)
        print(f"Hoàn thành xử lý lô embeddings. Đang lưu cache...")
        save_embeddings_cache()
    else:
        print("Không có học sinh nào trong lô hiện tại cần cập nhật embeddings.")
    
    if pending_embedding_updates: # Kiểm tra lại nếu có yêu cầu mới trong lúc xử lý
        print(f"Còn {len(pending_embedding_updates)} học sinh chờ cập nhật trong lô tiếp theo: {list(pending_embedding_updates)}")

def schedule_student_for_update(student_id):
    global pending_embedding_updates
    
    # Kiểm tra xem học sinh có tồn tại trong JSON không, phòng trường hợp CLASS_STUDENTS_DATA chưa được cập nhật
    # (ví dụ: thêm học sinh mới và ảnh cùng lúc)
    student_exists_in_json = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
    try:
        with open(students_json_path, 'r', encoding='utf-8') as f:
            all_students_list = json.load(f)
        if any(s.get("id") == student_id for s in all_students_list):
            student_exists_in_json = True
    except Exception:
        pass # Bỏ qua nếu không đọc được file

    if student_id in CLASS_STUDENTS_DATA or student_exists_in_json:
        pending_embedding_updates.add(student_id)
        print(f"Đã lên lịch cập nhật embeddings cho học sinh ID {student_id}.")
        # Có thể trigger chạy batch update sớm hơn nếu cần
        # global last_batch_update_time
        # last_batch_update_time = 0 # Reset để chạy ngay lần kiểm tra tiếp theo
        return True
    else:
        print(f"CẢNH BÁO: Không thể lên lịch cập nhật. HS ID {student_id} không tồn tại trong dữ liệu hoặc file JSON.")
        return False

# --- CÁC HÀM LƯU, ĐỌC CACHE EMBEDDINGS ---
def save_embeddings_cache():
    global CLASS_STUDENTS_DATA, embeddings_last_modified
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
    metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)
    
    try:
        embeddings_data_to_save = {}
        for student_id, data in CLASS_STUDENTS_DATA.items():
            if data.get("known_encodings_facenet"): # Chỉ lưu nếu có encodings
                embeddings_data_to_save[student_id] = np.array(data["known_encodings_facenet"])
        
        if not embeddings_data_to_save and not embeddings_last_modified: # Nếu không có gì để lưu
            print("Không có dữ liệu embeddings hoặc metadata mới để lưu vào cache.")
            # Kiểm tra xem có nên xóa file cache cũ không nếu chúng rỗng
            if os.path.exists(cache_file_path): os.remove(cache_file_path)
            if os.path.exists(metadata_file_path): os.remove(metadata_file_path)
            return False

        # Lưu embeddings
        if embeddings_data_to_save:
            np.savez_compressed(cache_file_path, **embeddings_data_to_save)
            print(f"Đã lưu cache embeddings cho {len(embeddings_data_to_save)} học sinh vào '{cache_file_path}'.")
        elif os.path.exists(cache_file_path): # Nếu không có data nhưng file tồn tại, có thể là đã xóa hết HS
            os.remove(cache_file_path)
            print(f"Đã xóa file cache embeddings rỗng: '{cache_file_path}'.")


        # Lưu metadata (ngay cả khi embeddings_data rỗng, metadata có thể vẫn cần cập nhật)
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_last_modified, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu metadata embeddings vào '{metadata_file_path}'.")
        
        return True
    
    except Exception as e:
        print(f"Lỗi khi lưu cache embeddings: {e}")
        return False

def load_embeddings_cache():
    global CLASS_STUDENTS_DATA, embeddings_last_modified
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
    metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)
    
    if not os.path.exists(cache_file_path) or not os.path.exists(metadata_file_path):
        print("Không tìm thấy tệp cache embeddings hoặc metadata. Sẽ tạo mới từ ảnh.")
        return False
    
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            cached_modified_times = json.load(f)
        
        current_folder_states = get_folder_modified_times() # Lấy trạng thái hiện tại của các thư mục ảnh

        # Kiểm tra những học sinh có trong cache metadata nhưng không còn thư mục ảnh
        # hoặc những học sinh mới có thư mục ảnh nhưng chưa có trong cache metadata
        students_in_cache_metadata = set(cached_modified_times.keys())
        students_with_folders = set(current_folder_states.keys())
        
        # Nếu có sự khác biệt về danh sách học sinh (thêm/xóa thư mục), cần re-cache
        if students_in_cache_metadata != students_with_folders:
            print("Phát hiện thay đổi trong danh sách thư mục học sinh (thêm/xóa). Cần làm mới cache.")
            # Xóa metadata cũ để đảm bảo load_known_faces_with_facenet_code chạy lại toàn bộ
            embeddings_last_modified.clear() 
            return False

        # Kiểm tra thời gian sửa đổi cho từng học sinh có thư mục
        for student_id, current_mod_time in current_folder_states.items():
            if student_id not in cached_modified_times or cached_modified_times[student_id] < current_mod_time:
                print(f"Phát hiện thay đổi trong thư mục ảnh của học sinh {student_id}. Cần làm mới cache.")
                embeddings_last_modified.clear() 
                return False
        
        # Nếu không có thay đổi, tải cache embeddings
        embeddings_cache_content = np.load(cache_file_path, allow_pickle=True)
        
        loaded_count = 0
        for student_id in embeddings_cache_content.files:
            if student_id in CLASS_STUDENTS_DATA: # Chỉ load cho HS đã có trong CLASS_STUDENTS_DATA (từ JSON)
                CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"] = embeddings_cache_content[student_id].tolist()
                loaded_count += 1
            # else: HS có trong cache nhưng không có trong JSON -> bỏ qua, sẽ được dọn dẹp khi save_embeddings_cache
        
        embeddings_last_modified = cached_modified_times # Cập nhật thời gian từ cache
        
        if loaded_count > 0:
            print(f"Đã tải cache embeddings thành công cho {loaded_count} học sinh.")
        else:
            print("Cache embeddings không chứa dữ liệu cho các học sinh hiện tại hoặc cache rỗng.")
            # Nếu cache rỗng nhưng metadata nói rằng nó hợp lệ, vẫn trả về True,
            # load_known_faces_with_facenet_code sẽ không làm gì vì cache hợp lệ.
        return True 
    
    except Exception as e:
        print(f"Lỗi khi tải cache embeddings: {e}. Sẽ tạo mới từ ảnh.")
        embeddings_last_modified.clear() # Xóa metadata nếu có lỗi để đảm bảo re-cache
        return False

def get_folder_modified_times():
    modified_times = {}
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    known_faces_root_dir = os.path.join(BASE_DIR, KNOWN_FACES_DIR)
    
    if not os.path.exists(known_faces_root_dir) or not os.path.isdir(known_faces_root_dir):
        return modified_times
    
    for student_id_folder_name in os.listdir(known_faces_root_dir):
        student_dir_path = os.path.join(known_faces_root_dir, student_id_folder_name)
        if os.path.isdir(student_dir_path):
            latest_time_in_folder = os.path.getmtime(student_dir_path) # Thời gian sửa đổi của thư mục
            # Kiểm tra cả thời gian sửa đổi của các file bên trong
            for item_name in os.listdir(student_dir_path):
                item_path = os.path.join(student_dir_path, item_name)
                if os.path.isfile(item_path):
                    item_mod_time = os.path.getmtime(item_path)
                    if item_mod_time > latest_time_in_folder:
                        latest_time_in_folder = item_mod_time
            
            modified_times[student_id_folder_name] = latest_time_in_folder
    return modified_times

# --- API TRIGGER VÀ STATUS CHO EMBEDDINGS ---
@app.route('/api/update_embeddings', methods=['POST'])
def trigger_embeddings_update_api(): # Đổi tên hàm để tránh trùng
    global mtcnn_facenet, resnet_facenet, device_facenet # Cần các model này
    try:
        data = request.json or {}
        student_id = data.get('student_id')
        force_immediate_update = data.get('force_immediate', False) # Đổi tên param
        
        if student_id:
            if force_immediate_update:
                print(f"API: Yêu cầu cập nhật embeddings ngay cho HS ID: {student_id}")
                # Đảm bảo models đã được khởi tạo
                if not mtcnn_facenet or not resnet_facenet:
                    return jsonify({"success": False, "message": "FaceNet models chưa sẵn sàng."}), 503
                success = update_single_student_embeddings(student_id, mtcnn_facenet, resnet_facenet, device_facenet)
                if success is not None: # update_single_student_embeddings có thể trả về False nếu không có ảnh
                    save_embeddings_cache() # Lưu cache ngay sau khi cập nhật một HS
                    return jsonify({"success": True, "message": f"Đã xử lý cập nhật embeddings cho HS ID {student_id}. Thành công: {success}"})
                else: # Trả về None hoặc lỗi gì đó
                     return jsonify({"success": False, "message": f"Không thể cập nhật embeddings cho HS ID {student_id}. Kiểm tra logs."}), 400
            else:
                if schedule_student_for_update(student_id):
                    return jsonify({"success": True, "message": f"Đã lên lịch cập nhật embeddings cho HS ID {student_id}"})
                else:
                    return jsonify({"success": False, "message": f"Không thể lên lịch cập nhật cho HS ID {student_id}"}), 400
        else: # Cập nhật cho tất cả
            if force_immediate_update:
                print("API: Yêu cầu cập nhật embeddings ngay cho TẤT CẢ học sinh.")
                if not mtcnn_facenet or not resnet_facenet:
                    return jsonify({"success": False, "message": "FaceNet models chưa sẵn sàng."}), 503
                
                all_student_ids = list(CLASS_STUDENTS_DATA.keys()) # Lấy danh sách HS hiện tại
                if not all_student_ids:
                    return jsonify({"success": True, "message": "Không có học sinh nào để cập nhật embeddings."})

                updated_any = False
                for sid_to_update in all_student_ids:
                    if update_single_student_embeddings(sid_to_update, mtcnn_facenet, resnet_facenet, device_facenet):
                        updated_any = True
                
                if updated_any or all_student_ids : # Lưu cache ngay cả khi không có encoding nào được tạo
                    save_embeddings_cache()
                return jsonify({"success": True, "message": f"Đã xử lý cập nhật embeddings cho {len(all_student_ids)} học sinh."})
            else:
                scheduled_count = 0
                all_student_ids_for_schedule = list(CLASS_STUDENTS_DATA.keys())
                if not all_student_ids_for_schedule:
                     return jsonify({"success": True, "message": "Không có học sinh nào để lên lịch cập nhật."})

                for sid_to_schedule in all_student_ids_for_schedule:
                    if schedule_student_for_update(sid_to_schedule):
                        scheduled_count += 1
                return jsonify({"success": True, "message": f"Đã lên lịch cập nhật embeddings cho {scheduled_count} học sinh."})
    
    except Exception as e:
        print(f"Lỗi API /api/update_embeddings: {e}")
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500

@app.route('/api/cache_status', methods=['GET'])
def get_cache_status_api(): # Đổi tên hàm
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
        metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)
        
        cache_exists = os.path.exists(cache_file_path)
        metadata_exists = os.path.exists(metadata_file_path)
        
        pending_count = len(pending_embedding_updates)
        pending_list = list(pending_embedding_updates)
        
        student_count = len(CLASS_STUDENTS_DATA)
        students_with_embeddings = 0
        for data in CLASS_STUDENTS_DATA.values(): # Không cần student_id ở đây
            if data.get("known_encodings_facenet"): # Check list rỗng
                students_with_embeddings += 1
        
        cache_size = os.path.getsize(cache_file_path) if cache_exists else 0
        
        last_update_ts = last_batch_update_time
        last_update_str = datetime.datetime.fromtimestamp(last_update_ts).strftime("%Y-%m-%d %H:%M:%S") if last_update_ts > 0 else "Chưa chạy lần nào"
        
        next_scheduled_ts = last_update_ts + BATCH_UPDATE_INTERVAL if last_update_ts > 0 else time.time() # Nếu chưa chạy, thì lần tiếp theo là "ngay bây giờ" (khi có pending)
        next_scheduled_str = datetime.datetime.fromtimestamp(next_scheduled_ts).strftime("%Y-%m-%d %H:%M:%S")
        if not pending_embedding_updates and last_update_ts > 0:
             next_scheduled_str = "Không có cập nhật đang chờ"


        return jsonify({
            "success": True,
            "cache_file_exists": cache_exists,
            "metadata_file_exists": metadata_exists,
            "cache_file_size_bytes": cache_size,
            "cache_file_size_human": f"{cache_size / (1024*1024):.2f} MB" if cache_size > 0 else "0 B",
            "total_students_in_memory": student_count,
            "students_with_embeddings_in_memory": students_with_embeddings,
            "coverage_percentage_in_memory": f"{(students_with_embeddings / student_count * 100) if student_count > 0 else 0:.2f}%",
            "pending_updates_count": pending_count,
            "pending_student_ids": pending_list,
            "last_batch_update_timestamp": last_update_str,
            "next_scheduled_batch_update_check": next_scheduled_str,
            "batch_update_interval_seconds": BATCH_UPDATE_INTERVAL
        })
    
    except Exception as e:
        print(f"Lỗi API /api/cache_status: {e}")
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    load_event_history_from_file()
    print("Khởi động Flask server...")
    print(f"Lật camera ngang được cấu hình: {FLIP_CAMERA_HORIZONTAL}")
    print("Thread AI sẽ bắt đầu ngay sau đây.")
    
    ai_thread = threading.Thread(target=process_video_and_count, daemon=True)
    ai_thread.start()
    
    # Chờ một chút để thread AI có thể khởi tạo models trước khi API update_embeddings được gọi (nếu cần)
    # time.sleep(15) # Tùy thuộc vào thời gian khởi tạo model

    try:
        # Chạy Flask với reloader=False để tránh thread AI chạy nhiều lần khi debug=True
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Đang tắt ứng dụng...")
    except Exception as e:
        print(f"Lỗi không mong muốn khi chạy Flask app: {e}")
    finally:
        print("Lưu cache embeddings lần cuối trước khi thoát...")
        save_embeddings_cache() # Đảm bảo lưu cache khi ứng dụng kết thúc
        print("Đã lưu cache. Thoát ứng dụng.")