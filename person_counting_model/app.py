import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
# from flask import Flask, render_template, Response, jsonify, request as flask_request, send_from_directory # Original
from flask import Flask, render_template, Response, jsonify, request, send_from_directory # Changed as per request
# Bỏ dòng import request trùng lặp, giữ lại flask_request đã đổi tên # Comment from original, now using `request`
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import threading
import time
import os
import json
from facenet_pytorch import InceptionResnetV1, MTCNN
import datetime
import base64
import queue
from concurrent.futures import ThreadPoolExecutor # Added

# --- CẤU HÌNH CHUNG ---
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.6

HISTORY_JSON_PATH = "API/history.json"
MAX_HISTORY_LENGTH_IN_FILE = 500

MAX_AGE_DEEPSORT = 50
N_INIT_DEEPSORT = 3
NMS_MAX_OVERLAP_DEEPSORT = 1.0
MAX_COSINE_DISTANCE = 0.7
NN_BUDGET = None

FLIP_CAMERA_HORIZONTAL = True

FACENET_RECOGNITION_THRESHOLD = 0.7
KNOWN_FACES_DIR = "known_student_faces"
STUDENTS_LIST_JSON_PATH = "API\students_list.json"

RECOGNITION_HISTORY_LENGTH = 5
RECOGNITION_CONFIRM_THRESHOLD = 3

CLASS_STUDENTS_DATA = {}
students_on_bus_count = 0
strangers_on_bus_count = 0
total_people_on_bus = 0

EMBEDDINGS_CACHE_FILE = "embeddings_cache.npz"
EMBEDDINGS_METADATA_FILE = "embeddings_metadata.json"
embeddings_last_modified = {}
pending_embedding_updates = set()
last_batch_update_time = 0
BATCH_UPDATE_INTERVAL = 300

frame_lock = threading.Lock()

event_history_list = []
MAX_HISTORY_LENGTH = 200

tracking_data = {
    "students_status": {},
    "students_on_bus_count": 0,
    "strangers_on_bus_count": 0,
    "total_people_on_bus": 0,
    "last_event_message": ""
}
data_lock = threading.Lock()

# --- HÀNG ĐỢI FRAME TỪ WEBSOCKET ---
frame_queue = queue.Queue(maxsize=5)
processing_frame_width = None
processing_frame_height = None

# --- KHỞI TẠO FLASK APP VÀ SOCKETIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dieu_bi_mat_cua_ban!'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- KHAI BÁO EXECUTOR VÀ CỜ MODELS_READY ---
mtcnn_facenet, resnet_facenet, device_facenet = None, None, None
yolo_model, deepsort_tracker = None, None
MODELS_READY = False
executor = ThreadPoolExecutor(max_workers=1) # Nên đặt số worker phù hợp, 1 là đủ cho tác vụ cập nhật embedding tuần tự


# --- HÀM KHỞI TẠO MODELS AI CHÍNH ---
def initialize_models_only():
    global mtcnn_facenet, resnet_facenet, device_facenet, yolo_model, deepsort_tracker, MODELS_READY
    if MODELS_READY:
        print("Models đã được khởi tạo trước đó.")
        return

    print("Đang khởi tạo các model AI chính...")
    device_facenet = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'FaceNet models chạy trên: {device_facenet}')
    try:
        # Thêm các tham số tối ưu nếu cần
        mtcnn_facenet = MTCNN(keep_all=True, device=device_facenet, select_largest=False, post_process=False, image_size=160)
        resnet_facenet = InceptionResnetV1(pretrained='vggface2', device=device_facenet).eval()
        print("MTCNN và ResNet (FaceNet) đã khởi tạo.")

        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"Model YOLO {'chạy trên GPU.' if torch.cuda.is_available() else 'chạy trên CPU.'}")

        deepsort_tracker = DeepSort(
            max_age=MAX_AGE_DEEPSORT, n_init=N_INIT_DEEPSORT,
            nms_max_overlap=NMS_MAX_OVERLAP_DEEPSORT, max_cosine_distance=MAX_COSINE_DISTANCE,
            nn_budget=NN_BUDGET, embedder='mobilenet', embedder_gpu=torch.cuda.is_available()
        )
        print("Tracker DeepSORT đã khởi tạo.")
        MODELS_READY = True
        print("Tất cả model chính đã sẵn sàng.")
    except Exception as e:
        print(f"LỖI nghiêm trọng khi khởi tạo model AI chính: {e}")
        MODELS_READY = False
        # Có thể raise lỗi hoặc exit nếu model là bắt buộc để ứng dụng chạy


# --- HÀM TẢI DỮ LIỆU HỌC SINH VÀ ENCODING BAN ĐẦU ---
def initialize_data_and_face_encodings():
    global CLASS_STUDENTS_DATA, MODELS_READY, mtcnn_facenet, resnet_facenet, device_facenet
    print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Bắt đầu initialize_data_and_face_encodings.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    current_students_list_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
    current_known_faces_dir = os.path.join(BASE_DIR, KNOWN_FACES_DIR)

    if MODELS_READY:
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Models sẵn sàng. Trước khi load_students_from_json. CLASS_STUDENTS_DATA có {len(CLASS_STUDENTS_DATA)} entries.")
        load_students_from_json(current_students_list_json_path) # Đọc từ file JSON
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Sau khi load_students_from_json. CLASS_STUDENTS_DATA có {len(CLASS_STUDENTS_DATA)} entries. Vài keys: {list(CLASS_STUDENTS_DATA.keys())[:3]}")

        # Ngay sau khi load_students_from_json, CLASS_STUDENTS_DATA phải có dữ liệu nếu file JSON có.
        # Hàm load_known_faces_with_facenet_code sẽ gọi load_embeddings_cache() bên trong nó.
        load_known_faces_with_facenet_code(current_known_faces_dir, mtcnn_facenet, resnet_facenet, device_facenet)
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Hoàn tất tải dữ liệu học sinh và face encodings ban đầu.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Models FaceNet chưa sẵn sàng, không thể tải dữ liệu học sinh và khuôn mặt ban đầu.")

# --- HÀM NỀN CẬP NHẬT EMBEDDINGS VÀ THÔNG BÁO ---
def _background_update_embeddings_and_notify(student_id, client_sid_to_notify=None):
    global mtcnn_facenet, resnet_facenet, device_facenet, socketio, CLASS_STUDENTS_DATA, MODELS_READY
    action_description = f"cập nhật embeddings cho HS {student_id}"
    try:
        print(f"BG Task: Bắt đầu {action_description}")
        if not MODELS_READY:
            print(f"BG Task: Models chưa sẵn sàng cho {action_description}.")
            if socketio and client_sid_to_notify:
                socketio.emit('model_update_notification',
                              {'status': 'error', 'message': 'Hệ thống models chưa sẵn sàng, vui lòng thử lại sau.', 'student_id': student_id, 'action': 'update_embeddings'},
                              room=client_sid_to_notify)
            elif socketio: # Broadcast if no specific client
                 socketio.emit('model_update_notification',
                              {'status': 'error', 'message': f'Hệ thống models chưa sẵn sàng cho {action_description}.', 'student_id': student_id, 'action': 'update_embeddings'})
            return

        # CLASS_STUDENTS_DATA should be up-to-date from the API route.
        # update_single_student_embeddings itself has a fallback to load from JSON if not in CLASS_STUDENTS_DATA.
        if student_id not in CLASS_STUDENTS_DATA:
             print(f"BG Task: HS {student_id} không có trong CLASS_STUDENTS_DATA khi bắt đầu tác vụ nền. update_single_student_embeddings sẽ thử tải lại.")

        success_update = update_single_student_embeddings(student_id, mtcnn_facenet, resnet_facenet, device_facenet)

        # update_single_student_embeddings returns True if encoding happened, False if no faces found/error for student
        # It doesn't return None unless a very unexpected error occurs before its main logic.
        # Let's assume it returns True/False as per its logic (True for successful encoding, False for no faces/dir issue for that student)

        save_embeddings_cache() # Save cache regardless of whether this specific student had new faces
        message = f'Model cho HS {student_id} đã được xử lý cập nhật.'
        if success_update is False: # Explicitly check for False (meaning processed but no faces found or dir error)
            message += " (Lưu ý: Không tìm thấy khuôn mặt nào để mã hóa trong ảnh hoặc có lỗi thư mục.)"
        elif success_update is True:
            message += " (Đã tìm thấy và mã hóa khuôn mặt.)"


        print(f"BG Task: Hoàn thành {action_description}")
        if socketio and client_sid_to_notify:
            socketio.emit('model_update_notification',
                          {'status': 'success', 'message': message, 'student_id': student_id, 'action': 'update_embeddings'},
                          room=client_sid_to_notify)
        elif socketio: # Broadcast if no specific client
            socketio.emit('model_update_notification',
                          {'status': 'success', 'message': message, 'student_id': student_id, 'action': 'update_embeddings'})


    except Exception as e:
        print(f"BG Task: Lỗi khi {action_description}: {e}")
        import traceback
        traceback.print_exc()
        if socketio and client_sid_to_notify:
            socketio.emit('model_update_notification',
                          {'status': 'error', 'message': f'Lỗi hệ thống khi {action_description}.', 'student_id': student_id, 'action': 'update_embeddings'},
                          room=client_sid_to_notify)
        elif socketio: # Broadcast
            socketio.emit('model_update_notification',
                          {'status': 'error', 'message': f'Lỗi hệ thống khi {action_description}.', 'student_id': student_id, 'action': 'update_embeddings'})


# --- HÀM NỀN LƯU CACHE VÀ THÔNG BÁO ---
def _background_save_cache_and_notify(operation_description, student_id_involved=None, client_sid_to_notify=None):
    global socketio
    try:
        print(f"BG Task: Bắt đầu {operation_description}")
        save_embeddings_cache()
        print(f"BG Task: Hoàn thành {operation_description}")
        if socketio:
            message_text = f"Model embeddings đã được cập nhật sau khi {operation_description}."
            if student_id_involved:
                message_text = f"Model embeddings đã được cập nhật sau khi {operation_description} liên quan đến HS {student_id_involved}."

            if client_sid_to_notify:
                socketio.emit('model_update_notification',
                              {'status': 'success', 'message': message_text, 'student_id': student_id_involved, 'action': 'save_cache'},
                              room=client_sid_to_notify)
            else: # Broadcast
                 socketio.emit('model_update_notification',
                              {'status': 'success', 'message': message_text, 'student_id': student_id_involved, 'action': 'save_cache'})
    except Exception as e:
        print(f"BG Task: Lỗi khi {operation_description}: {e}")
        if socketio:
            error_message = f'Lỗi hệ thống khi {operation_description}.'
            if client_sid_to_notify:
                socketio.emit('model_update_notification',
                              {'status': 'error', 'message': error_message, 'student_id': student_id_involved, 'action': 'save_cache'},
                              room=client_sid_to_notify)
            else: # Broadcast
                socketio.emit('model_update_notification',
                              {'status': 'error', 'message': error_message, 'student_id': student_id_involved, 'action': 'save_cache'})


# --- CÁC HÀM NHẬN DIỆN KHUÔN MẶT (Giữ nguyên) ---
def initialize_facenet_models_from_original_code(): #This seems redundant if initialize_models_only is used
    global mtcnn_facenet, resnet_facenet, device_facenet # Should be already set by initialize_models_only
    if not mtcnn_facenet or not resnet_facenet:
        print("WARNING: initialize_facenet_models_from_original_code called but models not set. This should not happen if initialize_models_only ran.")
        # Fallback or error, but ideally initialize_models_only handles this.
        # For safety, let's keep the core logic if they are somehow None
        device_facenet_local = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'FaceNet models (fallback init) running on device: {device_facenet_local}')
        try:
            mtcnn_facenet = MTCNN(keep_all=True, device=device_facenet_local, select_largest=False, post_process=False, image_size=160)
            resnet_facenet = InceptionResnetV1(pretrained='vggface2', device=device_facenet_local).eval()
            device_facenet = device_facenet_local # Ensure global is set
            print("MTCNN và ResNet (FaceNet) từ code gốc đã khởi tạo (fallback).")
        except Exception as e:
            print(f"LỖI khởi tạo model FaceNet từ code gốc (fallback): {e}"); exit()
    else:
        print("FaceNet models (from original code perspective) already initialized.")


def get_face_encodings_and_boxes_from_original_code(image_rgb, mtcnn_model, resnet_model, device_):
    boxes, probs = mtcnn_model.detect(image_rgb)
    face_encodings = []
    detected_face_boxes = []
    aligned_face_tensors = None
    if boxes is not None:
        try:
            # MTCNN with post_process=False returns a list of PIL Images or None
            # If it returns tensors directly, this logic is fine.
            # If it returns PIL Images, they need to be converted to tensors.
            # The facenet_pytorch MTCNN's __call__ method handles this if input is PIL/numpy
            aligned_face_tensors = mtcnn_model(image_rgb)
        except Exception as e:
            # print(f"Error during MTCNN processing (get_face_encodings): {e}") # Debug
            return [], [], None

        if aligned_face_tensors is not None:
            # aligned_face_tensors will be a tensor or list of tensors if faces are found
            # It will be None if no faces are found or post_process=False and no faces
            if isinstance(aligned_face_tensors, list): # Should not happen if post_process=False and faces are found (returns batch tensor)
                if not aligned_face_tensors: return [], [], None
                valid_tensors = [t for t in aligned_face_tensors if t is not None]
                if not valid_tensors: return [], [], None
                try:
                    aligned_face_tensors_batch = torch.stack(valid_tensors).to(device_)
                except RuntimeError as e: # Handle cases where tensors might not be stackable (e.g. different sizes, though MTCNN should normalize)
                    # print(f"RuntimeError stacking tensors: {e}. Processing individually.") # Debug
                    temp_encodings = []
                    for tensor_face in valid_tensors:
                        if tensor_face is not None:
                            tensor_face = tensor_face.unsqueeze(0).to(device_) # Add batch dim
                            with torch.no_grad():
                                encoding = resnet_model(tensor_face).cpu().numpy().flatten()
                                temp_encodings.append(encoding)
                    face_encodings = temp_encodings
                    # Adjust detected_face_boxes to match the number of successful encodings
                    detected_face_boxes = boxes[:len(face_encodings)].tolist() if boxes is not None else []
                    return face_encodings, detected_face_boxes, aligned_face_tensors # Return original aligned_face_tensors
            elif isinstance(aligned_face_tensors, torch.Tensor):
                 aligned_face_tensors_batch = aligned_face_tensors.to(device_)
            else: # Should not happen
                return [], [], None

            with torch.no_grad():
                encodings_batch = resnet_model(aligned_face_tensors_batch).cpu().numpy()

            # Ensure encodings match detected boxes; MTCNN with keep_all=True should align
            num_detected_faces = len(boxes)
            num_processed_faces = encodings_batch.shape[0]

            for i in range(num_processed_faces): # Iterate over successfully processed faces
                if i < num_detected_faces: # Match with original boxes
                    face_encodings.append(encodings_batch[i].flatten())
                    detected_face_boxes.append(boxes[i])
                # else: # More processed faces than detected boxes - should not happen with keep_all=True
                    # print("Warning: More processed faces than detected boxes.")

    return face_encodings, detected_face_boxes, aligned_face_tensors


def load_known_faces_with_facenet_code(known_faces_dir_path, mtcnn_model, resnet_model, device_):
    global CLASS_STUDENTS_DATA, embeddings_last_modified
    print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Bắt đầu load_known_faces_with_facenet_code. CLASS_STUDENTS_DATA có {len(CLASS_STUDENTS_DATA)} entries.") # Kiểm tra ở đây
    if not MODELS_READY: # Check if dependent models are ready
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: CẢNH BÁO (load_known_faces): Models FaceNet (MTCNN/ResNet) chưa sẵn sàng. Không thể tải khuôn mặt.")
        return
    if not CLASS_STUDENTS_DATA: # Nếu ở đây CLASS_STUDENTS_DATA rỗng thì là vấn đề
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: CẢNH BÁO trong load_known_faces: CLASS_STUDENTS_DATA rỗng!")
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: CẢNH BÁO: Chưa có dữ liệu học sinh từ JSON. Không thể liên kết ảnh khuôn mặt (FaceNet).")
        return
    if not os.path.exists(known_faces_dir_path):
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: THƯ MỤC KHUÔN MẶT (FaceNet) '{known_faces_dir_path}' KHÔNG TÌM THẤY.")
        return
    if load_embeddings_cache():
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Đã tải thành công từ cache. Kết thúc load_known_faces_with_facenet_code.")
        return
    print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Đang tải khuôn mặt học sinh (FaceNet) từ: {known_faces_dir_path} (cache không hợp lệ hoặc không tồn tại).")
    total_encoded_faces = 0
    current_modified_times = get_folder_modified_times()
    for student_id_from_folder_raw in os.listdir(known_faces_dir_path):
        # ĐẢM BẢO ID LÀ STRING VÀ LOẠI BỎ KHOẢNG TRẮNG THỪA
        student_id_from_folder = str(student_id_from_folder_raw).strip()
        if not student_id_from_folder: continue # Bỏ qua nếu tên thư mục rỗng

        person_dir_path = os.path.join(known_faces_dir_path, student_id_from_folder_raw) # Dùng tên gốc để join path
        if os.path.isdir(person_dir_path):
            if student_id_from_folder in CLASS_STUDENTS_DATA: # So sánh str với str
                print(f"  [{time.strftime('%H:%M:%S')}] APP.PY: Đang xử lý (FaceNet) cho HS ID: {student_id_from_folder}")
                if "known_encodings_facenet" not in CLASS_STUDENTS_DATA[student_id_from_folder]:
                    CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"] = []
                else: # Clear old encodings before reloading
                    CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"] = []
                image_count_for_student = 0
                for image_name in os.listdir(person_dir_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir_path, image_name)
                        known_image = cv2.imread(image_path)
                        if known_image is not None:
                            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
                            encodings, _, _ = get_face_encodings_and_boxes_from_original_code(
                                known_image_rgb, mtcnn_model, resnet_model, device_
                            )
                            if encodings: # Should be a list of encodings, take the first one for the primary face in the image
                                CLASS_STUDENTS_DATA[student_id_from_folder]["known_encodings_facenet"].append(encodings[0])
                                image_count_for_student +=1
                                total_encoded_faces +=1
                if image_count_for_student > 0:
                    print(f"    [{time.strftime('%H:%M:%S')}] APP.PY: Đã mã hóa {image_count_for_student} khuôn mặt (FaceNet) cho {student_id_from_folder}.")
                    if student_id_from_folder in current_modified_times:
                        embeddings_last_modified[student_id_from_folder] = current_modified_times[student_id_from_folder]
            else:
                print(f"[{time.strftime('%H:%M:%S')}] APP.PY: CẢNH BÁO (FaceNet): Thư mục ảnh '{student_id_from_folder}' (tên gốc: '{student_id_from_folder_raw}') không khớp ID nào trong CLASS_STUDENTS_DATA.")
    if total_encoded_faces == 0:
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: CẢNH BÁO (FaceNet): Không mã hóa được khuôn mặt nào.")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Hoàn tất tải và mã hóa {total_encoded_faces} khuôn mặt (FaceNet).")
        save_embeddings_cache() # Save after initial load

    # Final check for students without encodings
    for sid, data in CLASS_STUDENTS_DATA.items():
        if not data.get("known_encodings_facenet"):
            print(f"[{time.strftime('%H:%M:%S')}] APP.PY: LƯU Ý (FaceNet): Học sinh {sid} không có ảnh khuôn mặt được mã hóa.")
    print(f"[{time.strftime('%H:%M:%S')}] APP.PY: Kết thúc load_known_faces_with_facenet_code.")


def recognize_faces_with_facenet_code(current_face_encoding, student_id_to_check, threshold):
    global CLASS_STUDENTS_DATA
    if student_id_to_check not in CLASS_STUDENTS_DATA or \
       not CLASS_STUDENTS_DATA[student_id_to_check].get("known_encodings_facenet"):
        return "Unknown", None, float('inf')
    known_encodings_for_student = np.array(CLASS_STUDENTS_DATA[student_id_to_check]["known_encodings_facenet"])
    if not known_encodings_for_student.size or current_face_encoding is None: # current_face_encoding is a single encoding
        return "Unknown", None, float('inf')

    # current_face_encoding is (embedding_dim,), known_encodings_for_student is (num_known, embedding_dim)
    distances = np.linalg.norm(known_encodings_for_student - current_face_encoding, axis=1)
    min_distance = np.min(distances)

    if min_distance < threshold:
        return CLASS_STUDENTS_DATA[student_id_to_check]["name"], student_id_to_check, min_distance
    return "Unknown", None, min_distance

def find_best_match_among_all_students_facenet(current_face_encoding, threshold):
    global CLASS_STUDENTS_DATA
    if current_face_encoding is None: # current_face_encoding is a single encoding
        return "Stranger", None, float('inf')

    best_match_name = "Stranger"
    best_match_student_id = None
    overall_min_distance = float('inf')

    for student_id, data in CLASS_STUDENTS_DATA.items():
        if data.get("known_encodings_facenet"):
            known_encodings_for_student = np.array(data["known_encodings_facenet"])
            if known_encodings_for_student.size == 0: continue # Skip if no known encodings for this student

            # Distances from current_face_encoding to all known_encodings_for_this_student
            distances_to_this_student = np.linalg.norm(known_encodings_for_student - current_face_encoding, axis=1)
            min_dist_to_this_student = np.min(distances_to_this_student)

            if min_dist_to_this_student < overall_min_distance:
                overall_min_distance = min_dist_to_this_student
                # Check if this new overall_min_distance is below threshold for a match
                if min_dist_to_this_student < threshold:
                    best_match_name = data["name"]
                    best_match_student_id = student_id
                # If it's better but still not under threshold, update overall_min_distance but don't assign student yet
                # This means if no student is under threshold, best_match_name remains "Stranger"

    if best_match_student_id is None: # If no match was found under threshold
        best_match_name = "Stranger"
        # overall_min_distance would be the distance to the closest "non-match" or inf if no faces/students

    return best_match_name, best_match_student_id, overall_min_distance


# --- HÀM TẢI DỮ LIỆU HỌC SINH TỪ JSON (Giữ nguyên) ---
def load_students_from_json(json_path):
    global CLASS_STUDENTS_DATA
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            students_list = json.load(f) # Dữ liệu từ JSON có thể là int hoặc str cho ID

        CLASS_STUDENTS_DATA = {}
        for student_info in students_list:
            student_id_from_json = student_info.get("id")
            if student_id_from_json is not None: # Kiểm tra ID có tồn tại không
                # ĐẢM BẢO ID LÀ STRING VÀ LOẠI BỎ KHOẢNG TRẮNG THỪA
                student_id = str(student_id_from_json).strip()
                if not student_id: # Bỏ qua nếu ID rỗng sau khi strip
                    print(f"CẢNH BÁO: ID học sinh rỗng trong JSON sau khi strip: {student_info}")
                    continue

                CLASS_STUDENTS_DATA[student_id] = {
                    "name": student_info.get("name", student_id),
                    "status": student_info.get("status", "out"),
                    "known_encodings_facenet": [],
                    "original_data": student_info
                }
            else:
                print(f"CẢNH BÁO: Mục học sinh trong JSON không có ID hợp lệ: {student_info}")
        print(f"Đã tải {len(CLASS_STUDENTS_DATA)} học sinh từ '{json_path}'. CLASS_STUDENTS_DATA keys: {list(CLASS_STUDENTS_DATA.keys())[:5]}")
    except FileNotFoundError:
        print(f"LỖI: File danh sách học sinh '{json_path}' không tìm thấy. Sẽ tạo file mới rỗng.")
        CLASS_STUDENTS_DATA = {}
        try:
            parent_dir = os.path.dirname(json_path)
            if parent_dir and not os.path.exists(parent_dir):
                 os.makedirs(parent_dir, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            print(f"Đã tạo file '{json_path}' rỗng.")
        except Exception as e_create:
            print(f"LỖI khi cố gắng tạo file '{json_path}': {e_create}")
    except json.JSONDecodeError:
        print(f"LỖI: File '{json_path}' không phải là file JSON hợp lệ.")
        CLASS_STUDENTS_DATA = {}
    except Exception as e:
        print(f"LỖI không xác định khi tải danh sách học sinh từ JSON: {e}")
        CLASS_STUDENTS_DATA = {}

# --- CÁC HÀM HỖ TRỢ LƯU/ĐỌC LỊCH SỬ FILE (Giữ nguyên) ---
def load_event_history_from_file():
    global event_history_list
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
    global event_history_list, data_lock, CLASS_STUDENTS_DATA
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    current_history_json_path = os.path.join(BASE_DIR, HISTORY_JSON_PATH)
    students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
    with data_lock:
        event_history_list.append(event_detail)
        if len(event_history_list) > MAX_HISTORY_LENGTH:
             event_history_list.pop(0)

        history_to_write = list(event_history_list) # Make a copy for writing
        if len(history_to_write) > MAX_HISTORY_LENGTH_IN_FILE:
            history_to_write = history_to_write[-MAX_HISTORY_LENGTH_IN_FILE:]

        # Update student status in the JSON file
        if event_detail.get("type") == "student":
            student_id = event_detail.get("id")
            if student_id and student_id in CLASS_STUDENTS_DATA: # Check if student is known
                try:
                    if not os.path.exists(students_json_path):
                        print(f"CẢNH BÁO: File {students_json_path} không tồn tại, không thể cập nhật trạng thái học sinh.")
                    else:
                        with open(students_json_path, 'r', encoding='utf-8') as f:
                            students_list_from_file = json.load(f)

                        student_found_in_file = False
                        for student_in_file in students_list_from_file:
                            if str(student_in_file.get("id")).strip() == str(student_id).strip(): # Ensure consistent comparison
                                student_found_in_file = True
                                if event_detail.get("action") == "Lên xe":
                                    student_in_file["status"] = "in"
                                elif event_detail.get("action") == "Xuống xe":
                                    student_in_file["status"] = "out"
                                break

                        if student_found_in_file:
                            with open(students_json_path, 'w', encoding='utf-8') as f:
                                json.dump(students_list_from_file, f, ensure_ascii=False, indent=4)
                        # else: # Student in CLASS_STUDENTS_DATA but not in file? Should be rare if loaded correctly.
                            # print(f"Cảnh báo: Học sinh {student_id} có trong bộ nhớ nhưng không có trong file {students_json_path} để cập nhật trạng thái.")
                except json.JSONDecodeError:
                    print(f"LỖI: File {students_json_path} không hợp lệ, không thể cập nhật trạng thái học sinh.")
                except Exception as e:
                    print(f"LỖI khi cập nhật trạng thái học sinh trong file JSON: {e}")

        # Save history
        try:
            os.makedirs(os.path.dirname(current_history_json_path), exist_ok=True)
            with open(current_history_json_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_write, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"LỖI khi lưu sự kiện vào file: {e}")

# --- HÀM KHỞI TẠO CÁC THÀNH PHẦN AI (OLD - To be replaced by new init functions) ---
# def initialize_ai_components(): # This function is effectively replaced
#     global yolo_model, deepsort_tracker
#     # global mtcnn_facenet, resnet_facenet, device_facenet # These are now initialized in initialize_models_only
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     current_students_list_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
#     current_known_faces_dir = os.path.join(BASE_DIR, KNOWN_FACES_DIR)
#     print("Đang khởi tạo các thành phần AI (không bao gồm camera)...")

#     # Model initialization moved to initialize_models_only()
#     # initialize_facenet_models_from_original_code() # This was part of old structure, now handled by initialize_models_only

#     # Data loading moved to initialize_data_and_face_encodings()
#     # load_students_from_json(current_students_list_json_path)
#     # load_known_faces_with_facenet_code(current_known_faces_dir, mtcnn_facenet, resnet_facenet, device_facenet)

#     print("Hoàn tất khởi tạo các thành phần AI.")


# --- XỬ LÝ FRAME TỪ WEBSOCKET ---
@socketio.on('video_frame_to_server')
def handle_video_frame(data_url):
    global frame_queue, processing_frame_width, processing_frame_height
    # Không cần client_sid ở đây nữa nếu emit cho tất cả

    try:
        header, encoded_data = data_url.split(',', 1)
        image_data = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            current_h, current_w = frame.shape[:2]

            if processing_frame_width is None or processing_frame_height is None:
                processing_frame_width = current_w
                processing_frame_height = current_h
                print(f"Backend xử lý frame kích thước: {processing_frame_width}x{processing_frame_height}")

            if current_w != processing_frame_width or current_h != processing_frame_height:
                frame = cv2.resize(frame, (processing_frame_width, processing_frame_height))

            # --- TỐI ƯU FRAME_QUEUE (chỉ với frame) ---
            if frame_queue.full():
                try:
                    dropped_frame = frame_queue.get_nowait() # Lấy ra frame cũ
                    # print("Hàng đợi frame đầy. Đã bỏ frame cũ nhất.")
                except queue.Empty:
                    pass # Should not happen if full() is true, but good for safety
            try:
                frame_queue.put_nowait(frame) # Chỉ đưa frame vào queue
            except queue.Full:
                # print("Hàng đợi frame vẫn đầy sau khi cố gắng dọn. Bỏ qua frame mới.")
                pass
            # --- KẾT THÚC TỐI ƯU ---
        else:
            # print("Lỗi giải mã frame từ client.")
            pass
    except Exception as e:
        # print(f"Lỗi xử lý frame từ WebSocket: {e}")
        pass

# --- HÀM CHẠY LOGIC AI VÀ ĐẾM ---
def process_video_and_count():
    global tracking_data, yolo_model, deepsort_tracker
    global processing_frame_width, processing_frame_height
    global mtcnn_facenet, resnet_facenet, device_facenet, CLASS_STUDENTS_DATA
    global students_on_bus_count, strangers_on_bus_count, total_people_on_bus
    global FLIP_CAMERA_HORIZONTAL, MODELS_READY

    while not MODELS_READY:
        print("Luồng xử lý video đang chờ models chính sẵn sàng...")
        time.sleep(1)

    print("Luồng xử lý video: Models đã sẵn sàng, bắt đầu vòng lặp chính.")

    tracked_identities = {}
    crossing_status = {}
    previous_positions = {}

    RECOGNITION_ZONE_WIDTH = 100 # pixels
    
    frame_counter = 0
    last_batch_check_time = time.time()

    while True:
        if processing_frame_width is None or processing_frame_height is None:
            time.sleep(0.05)
            continue
        try:
            # Lấy chỉ frame từ queue
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        current_counting_line_x_for_logic = processing_frame_width // 2

        frame_counter += 1
        current_time = time.time()

        if current_time - last_batch_check_time > 60: # Check every 60s for batch updates
            process_batch_embedding_updates() 
            last_batch_check_time = current_time

        if not yolo_model or not deepsort_tracker or not mtcnn_facenet or not resnet_facenet:
            print("LỖI: Một hoặc nhiều model AI không khả dụng trong luồng xử lý. Bỏ qua frame.")
            time.sleep(1) 
            continue

        yolo_results = yolo_model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections_for_deepsort = []
        current_frame_tracked_objects_info = [] # Reset for this frame

        if yolo_results and yolo_results[0].boxes:
            for box_data in yolo_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = box_data
                if int(cls_id) == 0: 
                    width, height = int(x2-x1), int(y2-y1)
                    if width >= 20 and height >= 40: 
                        detections_for_deepsort.append(
                            ([int(x1), int(y1), width, height], conf, "person")
                        )

        frame_for_deepsort_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracks = deepsort_tracker.update_tracks(detections_for_deepsort, frame=frame_for_deepsort_rgb)

        current_frame_tracked_ids = set()
        latest_event_message_this_frame = ""
        event_details_for_history = None 


        frame_for_facenet_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        for track in tracks:
            if not track.is_confirmed(): continue
            track_id_str = str(track.track_id)
            current_frame_tracked_ids.add(track_id_str)
            x1_trk, y1_trk, x2_trk, y2_trk = map(int, track.to_ltrb())
            center_x = (x1_trk + x2_trk) // 2

            if track_id_str not in tracked_identities:
                tracked_identities[track_id_str] = {
                    "name_history": [],
                    "confirmed_name": "Processing...",
                    "confirmed_student_id": None,
                    "last_recognition_attempt_frame": 0,
                    "last_known_score": 0.0,
                    "recognition_attempts": 0,
                    "is_student_confirmed": False,
                    "is_stranger_confirmed": False,
                }
            track_info = tracked_identities[track_id_str]
            
            should_recognize_again = False
            is_near_counting_line = abs(center_x - current_counting_line_x_for_logic) < RECOGNITION_ZONE_WIDTH

            RETRY_INTERVAL_PROCESSING = 10
            RETRY_INTERVAL_STRANGER_NEAR_LINE = 5
            RETRY_INTERVAL_STRANGER_FAR_LINE = 20
            MAX_RECOGNITION_ATTEMPTS_FOR_STRANGER = 5

            current_frame_since_last_attempt = frame_counter - track_info["last_recognition_attempt_frame"]

            if not track_info["is_student_confirmed"] and not track_info["is_stranger_confirmed"]:
                if track_info["confirmed_name"] == "Processing...":
                    if current_frame_since_last_attempt >= RETRY_INTERVAL_PROCESSING or track_info["recognition_attempts"] == 0:
                        should_recognize_again = True
                elif track_info["confirmed_student_id"] is None: 
                    if is_near_counting_line:
                        if current_frame_since_last_attempt >= RETRY_INTERVAL_STRANGER_NEAR_LINE:
                            should_recognize_again = True
                    else: 
                        if current_frame_since_last_attempt >= RETRY_INTERVAL_STRANGER_FAR_LINE:
                            should_recognize_again = True
            
            if is_near_counting_line and not track_info["is_student_confirmed"] and not track_info["is_stranger_confirmed"]:
                if track_info["confirmed_name"] == "Processing..." and current_frame_since_last_attempt > 3:
                     should_recognize_again = True
                elif track_info["confirmed_student_id"] is None and current_frame_since_last_attempt > RETRY_INTERVAL_STRANGER_NEAR_LINE: 
                     should_recognize_again = True


            if should_recognize_again:
                track_info["recognition_attempts"] += 1
                track_info["last_recognition_attempt_frame"] = frame_counter

                person_crop_rgb = frame_for_facenet_rgb[max(0,y1_trk):min(processing_frame_height,y2_trk),
                                                        max(0,x1_trk):min(processing_frame_width,x2_trk)]

                if person_crop_rgb.size == 0 or person_crop_rgb.shape[0] < 20 or person_crop_rgb.shape[1] < 20: 
                    print(f"Track {track_id_str[:5]}: Crop không hợp lệ hoặc quá nhỏ để nhận diện ({person_crop_rgb.shape}). Bỏ qua nhận diện lần này.")
                    track_info["recognition_attempts"] = max(0, track_info["recognition_attempts"] - 1)
                    face_encodings_in_person = [] 
                else:
                    face_encodings_in_person, _, _ = get_face_encodings_and_boxes_from_original_code(
                        person_crop_rgb, mtcnn_facenet, resnet_facenet, device_facenet
                    )

                current_recognized_name = "Stranger" 
                current_student_id = None
                current_min_distance = float('inf')
                current_score = 0.0

                if face_encodings_in_person: 
                    target_encoding = face_encodings_in_person[0]
                    current_recognized_name, current_student_id, current_min_distance = find_best_match_among_all_students_facenet(
                        target_encoding, FACENET_RECOGNITION_THRESHOLD
                    )
                    current_score = (1.0 - current_min_distance) if current_min_distance != float('inf') else 0.0

                track_info["last_known_score"] = current_score 
                track_info["name_history"].append({
                    "name": current_recognized_name, 
                    "student_id": current_student_id,
                    "score": current_score
                })
                if len(track_info["name_history"]) > RECOGNITION_HISTORY_LENGTH:
                    track_info["name_history"].pop(0)
                
                id_counts = {}
                name_counts = {}
                max_id_count = 0
                best_id_from_history = None

                for rec in track_info["name_history"]:
                    if rec["student_id"]:
                        id_counts[rec["student_id"]] = id_counts.get(rec["student_id"], 0) + 1
                        if id_counts[rec["student_id"]] > max_id_count:
                            max_id_count = id_counts[rec["student_id"]]
                            best_id_from_history = rec["student_id"]
                    else:
                        name_counts[rec["name"]] = name_counts.get(rec["name"], 0) + 1
                
                if best_id_from_history and max_id_count >= RECOGNITION_CONFIRM_THRESHOLD:
                    track_info["confirmed_name"] = CLASS_STUDENTS_DATA.get(best_id_from_history, {}).get("name", "ID không tìm thấy")
                    track_info["confirmed_student_id"] = best_id_from_history
                    track_info["is_student_confirmed"] = True
                    track_info["is_stranger_confirmed"] = False
                elif (name_counts.get("Stranger", 0) >= RECOGNITION_CONFIRM_THRESHOLD or
                      (not best_id_from_history and len(track_info["name_history"]) >= RECOGNITION_HISTORY_LENGTH)):
                    track_info["confirmed_name"] = "Stranger"
                    track_info["confirmed_student_id"] = None
                    if track_info["recognition_attempts"] >= MAX_RECOGNITION_ATTEMPTS_FOR_STRANGER or \
                       (is_near_counting_line and len(track_info["name_history"]) >= RECOGNITION_HISTORY_LENGTH // 2):
                        track_info["is_stranger_confirmed"] = True
                        track_info["is_student_confirmed"] = False
                else:
                    track_info["confirmed_name"] = "Processing..."
                    track_info["confirmed_student_id"] = None
                    track_info["is_student_confirmed"] = False
                    track_info["is_stranger_confirmed"] = False

            final_recognized_name = track_info["confirmed_name"]
            final_student_id = track_info["confirmed_student_id"]
            
            if track_id_str in previous_positions:
                prev_center_x, _ = previous_positions[track_id_str]
                action_occurred = False
                action_type = "" 

                if prev_center_x < current_counting_line_x_for_logic and center_x >= current_counting_line_x_for_logic and \
                   crossing_status.get(track_id_str) != "crossed_LtoR": 
                    crossing_status[track_id_str] = "crossed_LtoR"
                    action_occurred = True
                    action_type = "Xuống xe" if FLIP_CAMERA_HORIZONTAL else "Lên xe"
                elif prev_center_x >= current_counting_line_x_for_logic and center_x < current_counting_line_x_for_logic and \
                     crossing_status.get(track_id_str) != "crossed_RtoL":
                    crossing_status[track_id_str] = "crossed_RtoL"
                    action_occurred = True
                    action_type = "Lên xe" if FLIP_CAMERA_HORIZONTAL else "Xuống xe"

                if action_occurred:
                    with data_lock: 
                        if action_type == "Lên xe":
                            if final_student_id and track_info["is_student_confirmed"]: 
                                if final_student_id in CLASS_STUDENTS_DATA and CLASS_STUDENTS_DATA[final_student_id]["status"] == "out":
                                    CLASS_STUDENTS_DATA[final_student_id]["status"] = "in"
                                    students_on_bus_count += 1
                                    latest_event_message_this_frame = f"HS {final_recognized_name} Lên xe."
                                    event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Lên xe"}
                            elif track_info["is_stranger_confirmed"]: 
                                strangers_on_bus_count += 1
                                latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Lên xe."
                                event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Lên xe"}
                        
                        elif action_type == "Xuống xe":
                            if final_student_id and track_info["is_student_confirmed"]:
                                if final_student_id in CLASS_STUDENTS_DATA and CLASS_STUDENTS_DATA[final_student_id]["status"] == "in":
                                    CLASS_STUDENTS_DATA[final_student_id]["status"] = "out"
                                    students_on_bus_count -= 1
                                    latest_event_message_this_frame = f"HS {final_recognized_name} Xuống xe."
                                    event_details_for_history = {"id": final_student_id, "name": final_recognized_name, "type": "student", "action": "Xuống xe"}
                            elif track_info["is_stranger_confirmed"]:
                                if strangers_on_bus_count > 0: 
                                    strangers_on_bus_count -= 1
                                latest_event_message_this_frame = f"Người lạ (ID:{track_id_str[:5]}) Xuống xe."
                                event_details_for_history = {"id": track_id_str, "name": "Stranger", "type": "stranger", "action": "Xuống xe"}

                    if latest_event_message_this_frame: print(f"EVENT: {latest_event_message_this_frame}")

                    students_on_bus_count = max(0, students_on_bus_count)
                    strangers_on_bus_count = max(0, strangers_on_bus_count)
                    total_people_on_bus = students_on_bus_count + strangers_on_bus_count

                    if event_details_for_history:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        event_details_for_history["timestamp"] = timestamp
                        save_event_to_history_file(event_details_for_history)

            previous_positions[track_id_str] = (center_x, (y1_trk + y2_trk) // 2)

            if abs(center_x - current_counting_line_x_for_logic) > RECOGNITION_ZONE_WIDTH / 2 and \
               (crossing_status.get(track_id_str) == "crossed_LtoR" or crossing_status.get(track_id_str) == "crossed_RtoL"):
                crossing_status[track_id_str] = "approaching"

            current_frame_tracked_objects_info.append({
                'id_track': track_id_str.split('-')[0],
                'bbox': [x1_trk, y1_trk, x2_trk, y2_trk],
                'name': final_recognized_name, 
                'student_id': final_student_id 
            })

        ids_to_remove = set(previous_positions.keys()) - current_frame_tracked_ids
        for old_id in ids_to_remove:
            if old_id in previous_positions: del previous_positions[old_id]
            if old_id in tracked_identities: del tracked_identities[old_id]
            if old_id in crossing_status: del crossing_status[old_id]

        # ------ PHẦN LOGIC XỬ LÝ AI GIỮ NGUYÊN (current_frame_tracked_objects_info được tạo ở trên) ------
        client_drawing_data = {
            "counting_line_x_on_frame": current_counting_line_x_for_logic,
            "tracked_objects_on_frame": current_frame_tracked_objects_info,
        }
        # Gửi dữ liệu vẽ này về TẤT CẢ client đang kết nối.
        socketio.emit('tracking_update_to_client', client_drawing_data) # Không có room

        if event_details_for_history:
            with data_lock:
                students_status_payload = {
                    sid: {"name": data.get("name", "N/A"), "status": data.get("status", "out")}
                    for sid, data in CLASS_STUDENTS_DATA.items()
                }
            comprehensive_status_update = {
                "students_on_bus_count": students_on_bus_count,
                "strangers_on_bus_count": strangers_on_bus_count,
                "total_people_on_bus": total_people_on_bus,
                "students_status": students_status_payload,
                "last_event_triggered": event_details_for_history,
            }
            socketio.emit('main_status_updated', comprehensive_status_update)

        with data_lock:
            tracking_data["students_status"] = {
                sid: {"name": data.get("name", "N/A"), "status": data.get("status", "out"), "original_data": data.get("original_data", {})}
                for sid, data in CLASS_STUDENTS_DATA.items()
            }
            tracking_data["students_on_bus_count"] = students_on_bus_count
            tracking_data["strangers_on_bus_count"] = strangers_on_bus_count
            tracking_data["total_people_on_bus"] = total_people_on_bus
            if event_details_for_history:
                tracking_data["last_event_message"] = f"{event_details_for_history.get('name')} {event_details_for_history.get('action')}"
            elif latest_event_message_this_frame:
                 tracking_data["last_event_message"] = latest_event_message_this_frame


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student_management_info')
def student_management_info():
    global tracking_data, data_lock
    with data_lock:
        # Deep copy for thread safety if complex objects are involved, json.dumps then loads is one way
        data_to_send = json.loads(json.dumps(tracking_data))
    return jsonify(data_to_send)

@app.route('/api/students', methods=['GET'])
def get_all_students_status():
    global CLASS_STUDENTS_DATA, data_lock
    students_list_with_status = []
    with data_lock:
        # Create a copy of items to iterate over if modifications could happen elsewhere, though data_lock helps
        items = list(CLASS_STUDENTS_DATA.items())

    for student_id, data in items:
        students_list_with_status.append({
            "id": student_id,
            "name": data.get("name", "N/A"),
            "status": data.get("status", "out"),
            "has_face_encodings": True if data.get("known_encodings_facenet") and len(data["known_encodings_facenet"]) > 0 else False,
            "original_data": data.get("original_data", {}) # Send all original data
        })
    return jsonify(students_list_with_status)

@app.route('/api/event_history', methods=['GET'])
def get_event_history():
    global event_history_list, data_lock
    with data_lock:
        events_to_return = list(event_history_list) # Return a copy
    return jsonify(events_to_return)

@app.route('/api/add_student', methods=['POST'])
def add_student():
    try:
        student_id_form = request.form.get('student_id')
        student_id = ""
        if student_id_form:
            student_id = str(student_id_form).strip()
            if not student_id:
                return jsonify({"success": False, "message": "ID học sinh không được rỗng"}), 400
        else:
            return jsonify({"success": False, "message": "Thiếu ID học sinh"}), 400
        
        name = request.form.get('name')
        if not name:
            return jsonify({"success": False, "message": "Thiếu tên học sinh"}), 400
        name = str(name).strip()
        if not name:
            return jsonify({"success": False, "message": "Tên học sinh không được rỗng"}), 400


        student_class = request.form.get('class')
        age = request.form.get('age')
        address = request.form.get('address')
        father_name = request.form.get('father_name')
        father_age = request.form.get('father_age')
        father_phone = request.form.get('father_phone')
        mother_name = request.form.get('mother_name')
        mother_age = request.form.get('mother_age')
        mother_phone = request.form.get('mother_phone')


        # Prepare student data for JSON file
        student_data_for_file = {
            "id": student_id, "name": name, "class": student_class, "age": age,
            "address": address, "father_name": father_name, "father_age": father_age,
            "father_phone": father_phone, "mother_name": mother_name, "mother_age": mother_age,
            "mother_phone": mother_phone, "status": "out" # Default status for new student
        }

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)

        existing_students_from_file = []
        if os.path.exists(students_json_path):
            try:
                with open(students_json_path, 'r', encoding='utf-8') as f:
                    existing_students_from_file = json.load(f)
            except json.JSONDecodeError:
                print(f"CẢNH BÁO: File {students_json_path} lỗi JSON. Sẽ ghi đè.")
                existing_students_from_file = []
        # else: file will be created if it doesn't exist

        student_exists_in_file = False
        for i, s_file in enumerate(existing_students_from_file):
            if str(s_file.get('id')).strip() == student_id: # Compare normalized IDs
                # Update existing student in file list
                existing_students_from_file[i] = student_data_for_file
                student_exists_in_file = True
                break
        if not student_exists_in_file:
            existing_students_from_file.append(student_data_for_file)

        try:
            with open(students_json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_students_from_file, f, ensure_ascii=False, indent=4)
        except Exception as e_write:
            print(f"LỖI khi ghi file {students_json_path}: {e_write}")
            return jsonify({"success": False, "message": f"Lỗi ghi file JSON: {str(e_write)}"}), 500

        # Save images
        images = request.files.getlist('images')
        has_new_images = False
        if images:
            student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id) # Use normalized student_id for folder name
            os.makedirs(student_folder, exist_ok=True)
            for i, image_file in enumerate(images):
                if image_file and image_file.filename:
                    has_new_images = True
                    filename = secure_filename(image_file.filename)
                    # Create a more unique filename to avoid overwrites if multiple default names
                    base, ext = os.path.splitext(filename)
                    unique_filename = f"{base}_{int(time.time())}_{i}{ext}"
                    image_path = os.path.join(student_folder, unique_filename)
                    image_file.save(image_path)

        # Update in-memory CLASS_STUDENTS_DATA
        global CLASS_STUDENTS_DATA, data_lock
        with data_lock:
            # Preserve old encodings if student existed, otherwise it's a new student
            old_encodings = CLASS_STUDENTS_DATA.get(student_id, {}).get("known_encodings_facenet", [])
            CLASS_STUDENTS_DATA[student_id] = {
                "name": name,
                "status": student_data_for_file["status"], # Use status from what was written to file
                "original_data": student_data_for_file, # Store the full data written to file
                "known_encodings_facenet": old_encodings # Will be updated by background task if new images
            }

        response_message = ""
        # If new images were uploaded OR it's a brand new student (even if no images initially)
        # We should schedule an embedding update.
        if has_new_images or not student_exists_in_file:
            print(f"API add_student: Lên lịch cập nhật embeddings cho HS ID {student_id}")
            # Get client SID if available (might not be for pure HTTP requests)
            current_client_sid = request.sid if hasattr(request, 'sid') else None
            executor.submit(_background_update_embeddings_and_notify, student_id, current_client_sid)
            response_message = f"Yêu cầu thêm/cập nhật HS {student_id} đã được tiếp nhận. Model sẽ được cập nhật trong nền."
        else: # Existing student, no new images, just info update
            response_message = f"Thông tin HS {student_id} đã được cập nhật (không có thay đổi về ảnh)."

        return jsonify({"success": True, "message": response_message, "student_id": student_id})

    except Exception as e:
        print(f"Lỗi khi thêm/cập nhật học sinh: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500


@app.route('/api/students/<student_id_path>', methods=['GET'])
def get_student_details(student_id_path):
    student_id = str(student_id_path).strip() # Normalize ID from path
    global CLASS_STUDENTS_DATA, data_lock
    with data_lock: # Access CLASS_STUDENTS_DATA under lock
        if student_id in CLASS_STUDENTS_DATA:
            student_data_mem = CLASS_STUDENTS_DATA[student_id]
            # Return a copy of the original_data to avoid external modification issues
            return jsonify({
                "id": student_id,
                "name": student_data_mem.get("name"),
                "status": student_data_mem.get("status"),
                "original_data": dict(student_data_mem.get("original_data", {}))
            })
        # else: Fall through to load from file if not in memory (e.g., after a restart before full load)

    # If not in memory, try loading from JSON file (less ideal, but a fallback)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
    if os.path.exists(students_json_path):
        try:
            with open(students_json_path, 'r', encoding='utf-8') as f:
                students_list_from_file = json.load(f)
            for s_data_file in students_list_from_file:
                if str(s_data_file.get("id")).strip() == student_id: # Normalize ID from file
                        # This student exists in file but not in CLASS_STUDENTS_DATA.
                        # This indicates a potential inconsistency or state where data hasn't fully loaded.
                        # For now, just return the file data.
                        return jsonify({
                        "id": student_id,
                        "name": s_data_file.get("name"),
                        "status": s_data_file.get("status", "out"),
                        "original_data": s_data_file # The whole record from JSON
                    })
        except Exception as e_read:
            print(f"Lỗi đọc file JSON khi lấy chi tiết học sinh '{student_id}': {e_read}")
            # Fall through to 404 if error or not found in file either

    return jsonify({"error": "Không tìm thấy học sinh"}), 404


@app.route('/api/students/<student_id_path>', methods=['PUT'])
def update_student_details(student_id_path):
    student_id = str(student_id_path).strip() # Normalize ID from path
    try:
        data_req = request.json # Using request from Flask
        name = data_req.get('name')
        if name: name = str(name).strip() # Normalize name

        student_class_req = data_req.get('class')
        age = data_req.get('age')
        address = data_req.get('address')
        father_name = data_req.get('father_name')
        father_age = data_req.get('father_age')
        father_phone = data_req.get('father_phone')
        mother_name = data_req.get('mother_name')
        mother_age = data_req.get('mother_age')
        mother_phone = data_req.get('mother_phone')

        if not name: # ID is from URL, name is essential from body
            return jsonify({"success": False, "message": "Thiếu tên học sinh"}), 400

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)

        if not os.path.exists(students_json_path):
            # This implies an issue, as students should be in a file if they exist.
            return jsonify({"success": False, "message": "Không tìm thấy file danh sách học sinh"}), 500

        try:
            with open(students_json_path, 'r', encoding='utf-8') as f:
                students_list_from_file = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "Lỗi đọc file danh sách học sinh"}), 500

        student_found_in_file = False
        updated_student_data_for_file = {} # To store the data that will be in CLASS_STUDENTS_DATA
        for i, student_file_data_item in enumerate(students_list_from_file):
            if str(student_file_data_item.get('id')).strip() == student_id: # Normalize ID from file
                # Preserve current status from file, update other fields
                current_status_from_file = student_file_data_item.get('status', 'out')

                updated_student_data_for_file = {
                    "id": student_id, "name": name, "class": student_class_req, "age": age,
                    "address": address, "father_name": father_name, "father_age": father_age,
                    "father_phone": father_phone, "mother_name": mother_name, "mother_age": mother_age,
                    "mother_phone": mother_phone, "status": current_status_from_file # Keep existing status
                }
                students_list_from_file[i] = updated_student_data_for_file # Update in the list to be saved
                student_found_in_file = True
                break

        if not student_found_in_file:
            return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id} trong file"}), 404

        # Write updated list back to JSON file
        try:
            with open(students_json_path, 'w', encoding='utf-8') as f:
                json.dump(students_list_from_file, f, ensure_ascii=False, indent=4)
        except Exception as e_write:
            print(f"Lỗi khi ghi file {students_json_path} (update): {e_write}")
            return jsonify({"success": False, "message": f"Lỗi ghi file JSON: {str(e_write)}"}), 500

        # Update in-memory CLASS_STUDENTS_DATA
        global CLASS_STUDENTS_DATA, data_lock
        with data_lock:
            if student_id in CLASS_STUDENTS_DATA:
                CLASS_STUDENTS_DATA[student_id]["name"] = name
                # Update all fields in original_data
                CLASS_STUDENTS_DATA[student_id]["original_data"] = updated_student_data_for_file
                # Status is part of original_data and should be consistent
                CLASS_STUDENTS_DATA[student_id]["status"] = updated_student_data_for_file["status"]
            else:
                # Student was in file but not in memory. Add them.
                # This case should be less common if initial load is robust.
                CLASS_STUDENTS_DATA[student_id] = {
                    "name": name,
                    "status": updated_student_data_for_file["status"],
                    "known_encodings_facenet": [], # No encodings known for this new memory entry yet
                    "original_data": updated_student_data_for_file
                }
                # Potentially schedule an embedding update if their folder exists, though this PUT is for text data
                # For simplicity, embedding updates are triggered by add_student with images or avatar update.

        return jsonify({"success": True, "message": "Cập nhật thông tin học sinh thành công"})

    except Exception as e:
        print(f"Lỗi khi cập nhật thông tin học sinh: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500


@app.route('/api/students/<student_id_path>/avatar', methods=['POST'])
def update_student_avatar(student_id_path):
    student_id = str(student_id_path).strip() # Normalize ID from path
    try:
        if 'avatar' not in request.files: # Using request from Flask
            return jsonify({"success": False, "message": "Không có file ảnh được gửi lên"}), 400

        avatar_file = request.files['avatar']
        if not avatar_file or not avatar_file.filename:
            return jsonify({"success": False, "message": "File ảnh không hợp lệ"}), 400

        # Check if student exists (in memory or file) before saving avatar
        student_exists = False
        if student_id in CLASS_STUDENTS_DATA:
            student_exists = True
        else: # Check file if not in memory
            BASE_DIR_CHECK = os.path.dirname(os.path.abspath(__file__))
            students_json_path_check = os.path.join(BASE_DIR_CHECK, STUDENTS_LIST_JSON_PATH)
            if os.path.exists(students_json_path_check):
                try:
                    with open(students_json_path_check, 'r', encoding='utf-8') as f_check:
                        students_list_check = json.load(f_check)
                    if any(str(s.get('id')).strip() == student_id for s in students_list_check): # Normalize
                        student_exists = True
                except Exception as e_read_check:
                    print(f"Lỗi đọc file JSON khi kiểm tra học sinh (avatar update for {student_id}): {e_read_check}")

        if not student_exists:
             return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id} để cập nhật avatar"}), 404

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id) # Use normalized ID for folder
        os.makedirs(student_folder, exist_ok=True) # Ensure folder exists

        filename = secure_filename(avatar_file.filename)
        name_part, ext_part = os.path.splitext(filename)
        # Unique filename to avoid overwriting, good for history/multiple images
        unique_filename = f"{name_part}_{int(time.time())}{ext_part}"
        avatar_path = os.path.join(student_folder, unique_filename)
        avatar_file.save(avatar_path)

        # Schedule background task to update embeddings
        print(f"API update_avatar: Lên lịch cập nhật embeddings cho HS ID {student_id}")
        current_client_sid = request.sid if hasattr(request, 'sid') else None
        executor.submit(_background_update_embeddings_and_notify, student_id, current_client_sid)

        return jsonify({
            "success": True,
            "message": "Yêu cầu cập nhật ảnh đại diện đã tiếp nhận. Model sẽ được cập nhật trong nền.",
            "image_url": f"/api/student_image/{student_id}?t={int(time.time())}" # Add timestamp to bust cache
        })

    except Exception as e:
        print(f"Lỗi khi cập nhật ảnh đại diện cho HS {student_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500


@app.route('/api/student_image/<student_id_path>', methods=['GET'])
def get_student_image(student_id_path):
    student_id = str(student_id_path).strip() # Normalize ID
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    student_folder = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id) # Use normalized ID
    # Path to default avatar, assuming it's in static/
    default_avatar_dir = os.path.join(BASE_DIR, 'static')
    default_avatar_name = 'default_avatar.png' # Ensure this file exists

    if not os.path.exists(student_folder) or not os.path.isdir(student_folder):
        return send_from_directory(default_avatar_dir, default_avatar_name, as_attachment=False)

    # Find all image files in the student's folder
    image_files = [f for f in os.listdir(student_folder)
                  if os.path.isfile(os.path.join(student_folder, f))
                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files: # No images in folder
        return send_from_directory(default_avatar_dir, default_avatar_name, as_attachment=False)

    # Get the most recently modified image file to serve as the avatar
    image_files_with_time = []
    for img_f in image_files:
        try:
            img_path = os.path.join(student_folder, img_f)
            mod_time = os.path.getmtime(img_path)
            image_files_with_time.append((img_f, mod_time))
        except FileNotFoundError:
            # Should not happen if os.listdir just listed it, but good for robustness
            continue

    if not image_files_with_time: # Should not be reached if image_files was populated
        return send_from_directory(default_avatar_dir, default_avatar_name, as_attachment=False)

    image_files_with_time.sort(key=lambda x: x[1], reverse=True) # Sort by mod time, newest first
    newest_image = image_files_with_time[0][0]

    return send_from_directory(student_folder, newest_image, as_attachment=False)

@app.route('/api/history.json', methods=['GET'])
def serve_history_json():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # HISTORY_JSON_PATH is relative to project root, e.g., "API/history.json"
    # Construct absolute path for existence check
    absolute_history_file_path = os.path.join(BASE_DIR, HISTORY_JSON_PATH)

    if os.path.exists(absolute_history_file_path):
        # send_from_directory needs directory and filename separately
        history_file_dir = os.path.join(BASE_DIR, os.path.dirname(HISTORY_JSON_PATH))
        history_file_name = os.path.basename(HISTORY_JSON_PATH)
        return send_from_directory(history_file_dir, history_file_name)
    else:
        return jsonify({"error": "File history.json không tìm thấy."}), 404

@app.route('/api/students/<student_id_path>', methods=['DELETE'])
def delete_student(student_id_path):
    student_id = str(student_id_path).strip() # Normalize ID
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)

        if not os.path.exists(students_json_path):
             # Though if no file, student likely doesn't exist anyway.
             return jsonify({"success": False, "message": "Không tìm thấy file danh sách học sinh"}), 500
        try:
            with open(students_json_path, 'r', encoding='utf-8') as f:
                students_list_from_file = json.load(f)
        except json.JSONDecodeError:
             return jsonify({"success": False, "message": "Lỗi đọc file danh sách học sinh"}), 500

        original_len = len(students_list_from_file)
        # Filter out the student to be deleted
        students_list_after_deletion = [s for s in students_list_from_file if str(s.get('id')).strip() != student_id] # Normalize

        if len(students_list_after_deletion) == original_len:
            # No student with that ID was found in the file
            return jsonify({"success": False, "message": f"Không tìm thấy học sinh có ID {student_id} để xóa trong file"}), 404

        # Write the modified list back to the JSON file
        try:
            with open(students_json_path, 'w', encoding='utf-8') as f:
                json.dump(students_list_after_deletion, f, ensure_ascii=False, indent=4)
        except Exception as e_write:
            print(f"Lỗi khi ghi file {students_json_path} (delete student {student_id}): {e_write}")
            return jsonify({"success": False, "message": f"Lỗi ghi file JSON: {str(e_write)}"}), 500

        # Delete student's image folder
        student_folder_to_delete = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id) # Use normalized ID
        if os.path.exists(student_folder_to_delete) and os.path.isdir(student_folder_to_delete):
            import shutil
            try:
                shutil.rmtree(student_folder_to_delete)
                print(f"Đã xóa thư mục ảnh: {student_folder_to_delete}")
            except Exception as e_rmtree:
                print(f"Lỗi khi xóa thư mục {student_folder_to_delete}: {e_rmtree}")
                # Continue with other cleanup even if folder deletion fails.

        # Remove from in-memory data structures (synchronized)
        global CLASS_STUDENTS_DATA, data_lock, embeddings_last_modified, pending_embedding_updates
        with data_lock:
            if student_id in CLASS_STUDENTS_DATA:
                del CLASS_STUDENTS_DATA[student_id]
            if student_id in embeddings_last_modified:
                del embeddings_last_modified[student_id]
            if student_id in pending_embedding_updates: # Remove from pending batch updates if any
                pending_embedding_updates.remove(student_id)

        # Schedule background task to save the (now modified) embeddings cache
        print(f"API delete_student: Lên lịch lưu cache sau khi xóa HS {student_id}")
        current_client_sid = request.sid if hasattr(request, 'sid') else None
        # The operation description should be generic for cache saving post-deletion
        executor.submit(_background_save_cache_and_notify,
                        f"xóa dữ liệu HS {student_id}",
                        student_id,
                        current_client_sid)

        return jsonify({"success": True, "message": f"Yêu cầu xóa HS {student_id} đã tiếp nhận. Dữ liệu và model cache sẽ được cập nhật."})

    except Exception as e:
        print(f"Lỗi khi xóa học sinh ID {student_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500

# --- CÁC HÀM CẬP NHẬT EMBEDDING (Giữ nguyên logic, called by background tasks or batch) ---
def update_single_student_embeddings(student_id, mtcnn_model, resnet_model, device_):
    global CLASS_STUDENTS_DATA, embeddings_last_modified, MODELS_READY
    student_id = str(student_id).strip() # Normalize ID

    if not MODELS_READY: # Should be checked by caller, but good for safety
        print(f"CẢNH BÁO (update_single for {student_id}): Models FaceNet chưa sẵn sàng. Bỏ qua.")
        return None # Indicate an issue preventing update

    if student_id not in CLASS_STUDENTS_DATA:
        print(f"CẢNH BÁO (update_single): Không tìm thấy HS ID {student_id} trong CLASS_STUDENTS_DATA. Thử tải lại từ JSON.")
        BASE_DIR_TEMP = os.path.dirname(os.path.abspath(__file__))
        students_json_path_temp = os.path.join(BASE_DIR_TEMP, STUDENTS_LIST_JSON_PATH)
        student_loaded_from_json = False
        try:
            if os.path.exists(students_json_path_temp):
                with open(students_json_path_temp, 'r', encoding='utf-8') as f_temp:
                    all_students_json = json.load(f_temp)
                student_info_json = next((s for s in all_students_json if str(s.get("id")).strip() == student_id), None) # Normalize
                if student_info_json:
                    # Add to CLASS_STUDENTS_DATA if found in JSON
                    with data_lock: # Ensure thread-safe modification
                        CLASS_STUDENTS_DATA[student_id] = {
                            "name": student_info_json.get("name", student_id),
                            "status": student_info_json.get("status", "out"),
                            "known_encodings_facenet": [], # Will be populated
                            "original_data": student_info_json
                        }
                    student_loaded_from_json = True
                    print(f"Đã tải thông tin HS ID {student_id} từ JSON vào CLASS_STUDENTS_DATA.")
        except Exception as e_json:
            print(f"Lỗi khi thử tải lại HS ID {student_id} từ JSON: {e_json}")

        if not student_loaded_from_json:
            print(f"Không thể tìm thấy hoặc tải HS ID {student_id} từ JSON. Bỏ qua cập nhật embedding.")
            return False # False indicates student not processed, distinct from None for system error

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    person_dir_path = os.path.join(BASE_DIR, KNOWN_FACES_DIR, student_id) # Use normalized ID

    if not os.path.exists(person_dir_path) or not os.path.isdir(person_dir_path):
        print(f"CẢNH BÁO (update_single for {student_id}): Không tìm thấy thư mục ảnh tại {person_dir_path}. Sẽ xóa encodings cũ (nếu có).")
        with data_lock: # Protect CLASS_STUDENTS_DATA
            if student_id in CLASS_STUDENTS_DATA: # Student might have been added just above
                CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"] = []
            if student_id in embeddings_last_modified:
                del embeddings_last_modified[student_id]
        return False # False: processed student's record, but no images/dir to encode

    print(f"Đang cập nhật embeddings (FaceNet) cho học sinh ID: {student_id}")

    # Ensure student entry exists in CLASS_STUDENTS_DATA and clear old encodings
    with data_lock: # Protect CLASS_STUDENTS_DATA
        if student_id not in CLASS_STUDENTS_DATA:
            # This should ideally not happen if the logic above worked, but as a safeguard:
            print(f"LỖI NGHIÊM TRỌNG: HS ID {student_id} không có trong CLASS_STUDENTS_DATA ngay trước khi xử lý ảnh. Điều này không nên xảy ra.")
            return None # Indicates a more severe logic flaw
        CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"] = [] # Clear old ones

    latest_mod_time_in_folder = 0
    image_count_encoded = 0 # Renamed from image_count to be specific

    # Iterate through images in the student's folder
    for image_name in os.listdir(person_dir_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(person_dir_path, image_name)
            try:
                mod_time = os.path.getmtime(image_path)
                if mod_time > latest_mod_time_in_folder:
                    latest_mod_time_in_folder = mod_time

                known_image = cv2.imread(image_path)
                if known_image is not None:
                    known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
                    # Use the global models passed as arguments
                    encodings, _, _ = get_face_encodings_and_boxes_from_original_code(
                        known_image_rgb, mtcnn_model, resnet_model, device_
                    )
                    if encodings: # encodings is a list, take the first one
                        with data_lock: # Append to the student's encodings list safely
                            CLASS_STUDENTS_DATA[student_id]["known_encodings_facenet"].append(encodings[0])
                        image_count_encoded += 1
            except FileNotFoundError: # Should be rare as listdir found it
                print(f"CẢNH BÁO: File {image_path} không tìm thấy trong khi xử lý cho HS {student_id}.")
                continue
            except Exception as e_img_proc:
                print(f"Lỗi khi xử lý ảnh {image_path} cho HS {student_id}: {e_img_proc}")

    if image_count_encoded > 0:
        print(f"Đã cập nhật {image_count_encoded} khuôn mặt (FaceNet) cho học sinh ID {student_id}.")
        if latest_mod_time_in_folder > 0: # Store the latest modification time of successfully processed images
            with data_lock: embeddings_last_modified[student_id] = latest_mod_time_in_folder
        return True # True: successfully encoded at least one face
    else:
        print(f"CẢNH BÁO (update_single for {student_id}): Không tìm thấy khuôn mặt nào trong ảnh. Encodings sẽ rỗng.")
        # Still update modification time based on folder, as an attempt was made
        try:
            folder_mod_time = os.path.getmtime(person_dir_path) # time of last modification to folder content/metadata
            # Use the more recent of image mod times (if any) or folder mod time
            final_mod_time_for_record = max(latest_mod_time_in_folder, folder_mod_time)
            with data_lock: embeddings_last_modified[student_id] = final_mod_time_for_record
        except FileNotFoundError: # If folder was deleted during processing (unlikely)
             with data_lock:
                if student_id in embeddings_last_modified:
                    del embeddings_last_modified[student_id]
        return False # False: processed, but no faces found/encoded

def process_batch_embedding_updates():
    global pending_embedding_updates, last_batch_update_time, mtcnn_facenet, resnet_facenet, device_facenet, data_lock, MODELS_READY

    if not pending_embedding_updates: return # Nothing to do

    current_time = time.time()
    # Allow first run immediately if pending, otherwise respect interval
    if last_batch_update_time != 0 and (current_time - last_batch_update_time < BATCH_UPDATE_INTERVAL):
        return

    if not MODELS_READY: # Check if models are ready for batch processing
        print("LỖI (process_batch): Models FaceNet chưa được khởi tạo. Bỏ qua cập nhật batch embeddings.")
        return

    # Take a snapshot of pending updates to process
    with data_lock:
        students_to_process_now = list(pending_embedding_updates)
        pending_embedding_updates.clear() # Clear the global pending set for this batch

    if not students_to_process_now: # Should not happen if initial check passed, but for safety
        last_batch_update_time = current_time # Still update time to avoid rapid checks
        return

    print(f"Bắt đầu cập nhật embeddings theo lô cho {len(students_to_process_now)} học sinh: {students_to_process_now}")

    updated_any_student_in_batch = False
    for student_id_raw in students_to_process_now:
        student_id = str(student_id_raw).strip() # Normalize ID
        # mtcnn_facenet, etc. are global and checked by MODELS_READY
        update_status = update_single_student_embeddings(student_id, mtcnn_facenet, resnet_facenet, device_facenet)
        if update_status is True: # True means successful encoding for this student
            updated_any_student_in_batch = True
        # update_status could be False (no faces) or None (system error for that student)
        # We continue processing other students in the batch

    last_batch_update_time = current_time # Update timestamp after processing this batch

    if updated_any_student_in_batch or students_to_process_now: # Save cache if any student was processed or attempted
        print(f"Hoàn thành xử lý lô embeddings. Đang lưu cache...")
        save_embeddings_cache()
    else: # Should not happen if students_to_process_now was non-empty
        print("Không có học sinh nào trong lô hiện tại cần cập nhật embeddings (hoặc không có thay đổi).")

    with data_lock: # Check if new items were added to pending_embedding_updates while this batch ran
        if pending_embedding_updates:
            print(f"Còn {len(pending_embedding_updates)} học sinh chờ cập nhật trong lô tiếp theo: {list(pending_embedding_updates)}")


def schedule_student_for_update(student_id_raw):
    global pending_embedding_updates, data_lock, CLASS_STUDENTS_DATA
    if not student_id_raw:
        print("CẢNH BÁO: Không thể lên lịch cập nhật. ID học sinh không hợp lệ (rỗng).")
        return False # Indicate failure to schedule
    
    student_id = str(student_id_raw).strip() # Normalize ID
    if not student_id:
        print(f"CẢNH BÁO: ID học sinh rỗng sau khi chuẩn hóa ('{student_id_raw}'). Không thể lên lịch cập nhật.")
        return False

    # Check if student exists either in memory or in the JSON file
    student_exists_in_memory = student_id in CLASS_STUDENTS_DATA
    student_exists_in_json_file = False

    if not student_exists_in_memory: # If not in memory, check the source of truth (JSON file)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        students_json_path = os.path.join(BASE_DIR, STUDENTS_LIST_JSON_PATH)
        try:
            if os.path.exists(students_json_path):
                with open(students_json_path, 'r', encoding='utf-8') as f:
                    all_students_list_from_file = json.load(f)
                if any(str(s.get("id")).strip() == student_id for s in all_students_list_from_file): # Normalize
                    student_exists_in_json_file = True
        except Exception as e_read_json:
            print(f"Lỗi khi kiểm tra sự tồn tại của HS {student_id} trong file JSON (schedule_student_for_update): {e_read_json}")
            # Proceed cautiously; if file error, we might not know if student exists there.

    if student_exists_in_memory or student_exists_in_json_file:
        with data_lock:
            pending_embedding_updates.add(student_id)
        print(f"Đã lên lịch cập nhật embeddings (batch) cho học sinh ID {student_id}.")
        return True # Successfully scheduled
    else:
        print(f"CẢNH BÁO (schedule_student): Không thể lên lịch cập nhật. HS ID {student_id} không tồn tại trong dữ liệu bộ nhớ hoặc file JSON.")
        return False # Failed to schedule


# --- CÁC HÀM LƯU, ĐỌC CACHE EMBEDDINGS (Giữ nguyên logic) ---
def save_embeddings_cache():
    global CLASS_STUDENTS_DATA, embeddings_last_modified, data_lock
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
    metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)

    try:
        embeddings_data_to_save = {}
        # Work with a copy of critical data under lock
        with data_lock:
            students_data_copy = dict(CLASS_STUDENTS_DATA) # Shallow copy of student dict
            metadata_to_save = dict(embeddings_last_modified)

        for student_id, data in students_data_copy.items():
            # Ensure student_id is string for key consistency in npz
            student_id_str = str(student_id).strip()
            # Ensure known_encodings_facenet is a list of numpy arrays or can be converted
            if data.get("known_encodings_facenet"):
                # Convert list of 1D arrays into a 2D numpy array for this student
                try:
                    embeddings_array = np.array(data["known_encodings_facenet"])
                    if embeddings_array.ndim == 1 and embeddings_array.size == 0: # Empty list resulted in empty array
                        pass # Don't save if no encodings
                    elif embeddings_array.ndim == 2 and embeddings_array.shape[0] > 0 : # Valid 2D array of encodings
                        embeddings_data_to_save[student_id_str] = embeddings_array
                    # Handle cases where it might be a list of empty lists or other malformed data
                    # For now, assume data["known_encodings_facenet"] is a list of 1D numpy arrays
                except Exception as e_np_array:
                    print(f"Lỗi khi chuyển đổi embeddings cho HS {student_id_str} sang np.array: {e_np_array}")


        if not embeddings_data_to_save and not metadata_to_save: # If nothing to save
            print("Không có dữ liệu embeddings hoặc metadata mới để lưu vào cache.")
            # Optionally remove old cache files if they exist and current state is empty
            if os.path.exists(cache_file_path):
                try: os.remove(cache_file_path); print(f"Đã xóa cache file cũ: {cache_file_path}")
                except OSError as e: print(f"Lỗi xóa cache file cũ '{cache_file_path}': {e}")
            if os.path.exists(metadata_file_path):
                try: os.remove(metadata_file_path); print(f"Đã xóa metadata file cũ: {metadata_file_path}")
                except OSError as e: print(f"Lỗi xóa metadata file cũ '{metadata_file_path}': {e}")
            return False # Indicate nothing was saved

        if embeddings_data_to_save:
            np.savez_compressed(cache_file_path, **embeddings_data_to_save)
            print(f"Đã lưu cache embeddings cho {len(embeddings_data_to_save)} học sinh vào '{cache_file_path}'.")
        elif os.path.exists(cache_file_path): # No embeddings to save, but cache file exists
            try:
                os.remove(cache_file_path)
                print(f"Đã xóa file cache embeddings rỗng (vì không có data mới): '{cache_file_path}'.")
            except OSError as e: print(f"Lỗi xóa cache file embeddings rỗng: {e}")

        # Always save metadata, even if it's empty (to signify an empty but valid state)
        # Ensure keys in metadata_to_save are also strings
        metadata_to_save_str_keys = {str(k).strip(): v for k, v in metadata_to_save.items()}
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save_str_keys, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu metadata embeddings vào '{metadata_file_path}'.")
        return True # Indicate successful save operation

    except Exception as e:
        print(f"Lỗi khi lưu cache embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False # Indicate failure


def load_embeddings_cache():
    global CLASS_STUDENTS_DATA, embeddings_last_modified, data_lock
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
    metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)

    if not os.path.exists(cache_file_path) or not os.path.exists(metadata_file_path):
        print("Không tìm thấy tệp cache embeddings hoặc metadata. Sẽ tạo mới từ ảnh nếu có.")
        return False # False means cache was not loaded, proceed with full encoding

    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            cached_modified_times_raw = json.load(f)
        # Normalize keys from metadata to string and strip
        cached_modified_times = {str(k).strip(): v for k, v in cached_modified_times_raw.items()}


        current_folder_states = get_folder_modified_times() # Returns keys as normalized strings

        # --- Cache Invalidation Logic ---
        # 1. Check if the set of students in metadata matches students with image folders
        students_in_cache_metadata = set(cached_modified_times.keys())
        students_with_folders_now = set(current_folder_states.keys()) # Already normalized

        if students_in_cache_metadata != students_with_folders_now:
            print("Phát hiện thay đổi trong danh sách thư mục học sinh (thêm/xóa so với metadata). Cần làm mới cache.")
            with data_lock: embeddings_last_modified.clear() # Clear old in-memory metadata
            return False # Force re-encoding

        # 2. Check modification times for each student folder against cached metadata
        for student_id, current_latest_mod_time_in_folder in current_folder_states.items(): # student_id is normalized
            cached_mod_time_for_student = cached_modified_times.get(student_id)
            if cached_mod_time_for_student is None: 
                print(f"HS {student_id} có thư mục ảnh nhưng không có trong metadata cache. Cần làm mới.")
                with data_lock: embeddings_last_modified.clear()
                return False

            if isinstance(cached_mod_time_for_student, (int, float)) and \
               isinstance(current_latest_mod_time_in_folder, (int, float)):
                if current_latest_mod_time_in_folder > cached_mod_time_for_student:
                    print(f"Phát hiện thay đổi trong thư mục ảnh của học sinh {student_id} (mới hơn cache). Cần làm mới cache.")
                    with data_lock: embeddings_last_modified.clear()
                    return False

        # --- Load Embeddings if Cache is Valid ---
        embeddings_cache_content = np.load(cache_file_path, allow_pickle=True)
        loaded_count = 0
        with data_lock: 
            for student_id_in_cache_raw in embeddings_cache_content.files:
                student_id_in_cache = str(student_id_in_cache_raw).strip() # ĐẢM BẢO ID LÀ STRING
                if student_id_in_cache in CLASS_STUDENTS_DATA: # So sánh str với str (CLASS_STUDENTS_DATA keys are normalized)
                    loaded_array = embeddings_cache_content[student_id_in_cache_raw] # Use original key for npz access
                    if loaded_array.ndim == 2:
                        CLASS_STUDENTS_DATA[student_id_in_cache]["known_encodings_facenet"] = [row for row in loaded_array]
                    elif loaded_array.ndim == 1 and loaded_array.size > 0 : 
                        CLASS_STUDENTS_DATA[student_id_in_cache]["known_encodings_facenet"] = [loaded_array]
                    else: 
                        CLASS_STUDENTS_DATA[student_id_in_cache]["known_encodings_facenet"] = []
                    loaded_count += 1

            embeddings_last_modified.clear() 
            embeddings_last_modified.update(cached_modified_times) # Use normalized keys from metadata

        if loaded_count > 0:
            print(f"Đã tải cache embeddings thành công cho {loaded_count} học sinh.")
        else:
            print("Cache embeddings hợp lệ nhưng không chứa dữ liệu cho các học sinh hiện tại, hoặc cache rỗng.")

        return True 

    except FileNotFoundError: 
        print("Lỗi FileNotFoundError khi tải cache embeddings (file có thể đã bị xóa). Sẽ tạo mới.")
        with data_lock: embeddings_last_modified.clear()
        return False
    except Exception as e:
        print(f"Lỗi không xác định khi tải cache embeddings: {e}")
        import traceback
        traceback.print_exc()
        with data_lock: embeddings_last_modified.clear() # Invalidate on any error
        return False


def get_folder_modified_times():
    modified_times = {} # Stores {student_id: latest_mod_time_for_that_student_folder_or_its_images}
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    known_faces_root_dir = os.path.join(BASE_DIR, KNOWN_FACES_DIR)

    if not os.path.exists(known_faces_root_dir) or not os.path.isdir(known_faces_root_dir):
        # print(f"Thư mục khuôn mặt gốc '{known_faces_root_dir}' không tồn tại.")
        return modified_times # Return empty if root dir doesn't exist

    for student_id_folder_name_raw in os.listdir(known_faces_root_dir):
        # ĐẢM BẢO ID LÀ STRING VÀ LOẠI BỎ KHOẢNG TRẮNG THỪA
        student_id_folder_name = str(student_id_folder_name_raw).strip()
        if not student_id_folder_name: continue

        student_dir_path = os.path.join(known_faces_root_dir, student_id_folder_name_raw) # Dùng tên gốc để join path
        if os.path.isdir(student_dir_path): # Process only if it's a directory (student's folder)
            try:
                # Initialize with the modification time of the directory itself
                latest_time_in_folder = os.path.getmtime(student_dir_path)

                # Check modification times of image files within the student's directory
                # has_images = False # Unused
                for item_name_in_student_folder in os.listdir(student_dir_path):
                    item_path = os.path.join(student_dir_path, item_name_in_student_folder)
                    if os.path.isfile(item_path) and item_name_in_student_folder.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # has_images = True # Unused
                        item_mod_time = os.path.getmtime(item_path)
                        if item_mod_time > latest_time_in_folder:
                            latest_time_in_folder = item_mod_time
                modified_times[student_id_folder_name] = latest_time_in_folder # Key là str

            except FileNotFoundError: # Should be rare if os.listdir worked
                print(f"CẢNH BÁO: Thư mục/File không tìm thấy trong khi lấy thời gian sửa đổi: {student_dir_path}")
                continue # Skip this student_id folder
            except Exception as e_mod_time:
                print(f"Lỗi lấy thời gian sửa đổi cho {student_dir_path}: {e_mod_time}")
                # Decide if to skip or record a default/error value. Skipping is safer.
    return modified_times

# --- API TRIGGER VÀ STATUS CHO EMBEDDINGS (Giữ nguyên) ---
@app.route('/api/update_embeddings', methods=['POST'])
def trigger_embeddings_update_api():
    global mtcnn_facenet, resnet_facenet, device_facenet, data_lock, CLASS_STUDENTS_DATA, MODELS_READY

    try:
        data_req = request.json or {} # Using request from Flask
        student_id_req_raw = data_req.get('student_id') # Specific student or all
        force_immediate_req = data_req.get('force_immediate', False) # Process now or schedule for batch

        if not MODELS_READY: # Check if core models are ready
            return jsonify({"success": False, "message": "FaceNet models (MTCNN/ResNet) chưa sẵn sàng."}), 503

        if student_id_req_raw: # Request for a specific student
            student_id_req = str(student_id_req_raw).strip()
            if not student_id_req:
                return jsonify({"success": False, "message": "ID học sinh không được rỗng."}), 400
            
            if force_immediate_req:
                print(f"API: Yêu cầu cập nhật embeddings ngay cho HS ID: {student_id_req}")
                update_result = update_single_student_embeddings(student_id_req, mtcnn_facenet, resnet_facenet, device_facenet)
                save_embeddings_cache() 

                status_message = f"Đã xử lý cập nhật embeddings cho HS ID {student_id_req}."
                if update_result is True: status_message += " (Có tạo embedding mới.)"
                elif update_result is False: status_message += " (Không tạo embedding mới - không có khuôn mặt hoặc lỗi thư mục.)"
                else: status_message += " (Lỗi hệ thống trong quá trình cập nhật.)" 
                return jsonify({"success": True, "message": status_message})
            else: # Schedule for batch update
                if schedule_student_for_update(student_id_req):
                    return jsonify({"success": True, "message": f"Đã lên lịch cập nhật embeddings (batch) cho HS ID {student_id_req}"})
                else: 
                    return jsonify({"success": False, "message": f"Không thể lên lịch cập nhật cho HS ID {student_id_req} (ID không tồn tại hoặc lỗi)."}), 400
        else: # Request for all students
            with data_lock: 
                all_student_ids_current = list(CLASS_STUDENTS_DATA.keys()) # Keys are already normalized

            if not all_student_ids_current:
                return jsonify({"success": True, "message": "Không có học sinh nào trong bộ nhớ để cập nhật embeddings."})

            if force_immediate_req:
                print(f"API: Yêu cầu cập nhật embeddings ngay cho TẤT CẢ ({len(all_student_ids_current)}) học sinh.")
                updated_any_forced = False
                for sid_to_update in all_student_ids_current: # sid_to_update is already normalized
                    if update_single_student_embeddings(sid_to_update, mtcnn_facenet, resnet_facenet, device_facenet) is True:
                        updated_any_forced = True
                if updated_any_forced or all_student_ids_current:
                    save_embeddings_cache()
                return jsonify({"success": True, "message": f"Đã xử lý cập nhật embeddings cho {len(all_student_ids_current)} học sinh."})
            else: # Schedule all for batch update
                scheduled_count = 0
                for sid_to_schedule in all_student_ids_current: # sid_to_schedule is already normalized
                    if schedule_student_for_update(sid_to_schedule):
                        scheduled_count += 1
                return jsonify({"success": True, "message": f"Đã lên lịch cập nhật embeddings (batch) cho {scheduled_count}/{len(all_student_ids_current)} học sinh."})

    except Exception as e:
        print(f"Lỗi API /api/update_embeddings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500


@app.route('/api/cache_status', methods=['GET'])
def get_cache_status_api():
    global pending_embedding_updates, last_batch_update_time, CLASS_STUDENTS_DATA, data_lock
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_file_path = os.path.join(BASE_DIR, EMBEDDINGS_CACHE_FILE)
        metadata_file_path = os.path.join(BASE_DIR, EMBEDDINGS_METADATA_FILE)

        cache_exists = os.path.exists(cache_file_path)
        metadata_exists = os.path.exists(metadata_file_path)

        # Get data under lock for consistency
        with data_lock:
            pending_list_copy = [str(sid).strip() for sid in pending_embedding_updates] # Normalize for display
            pending_count = len(pending_list_copy)
            student_count_mem = len(CLASS_STUDENTS_DATA)
            students_with_embeddings_mem = 0
            for data_mem_item in CLASS_STUDENTS_DATA.values(): # Iterate over values
                if data_mem_item.get("known_encodings_facenet") and len(data_mem_item["known_encodings_facenet"]) > 0:
                    students_with_embeddings_mem += 1
            last_update_ts_copy = last_batch_update_time # Copy timestamp

        cache_size_bytes = 0
        if cache_exists:
            try: cache_size_bytes = os.path.getsize(cache_file_path)
            except OSError: pass # File might be deleted between exists and getsize

        last_update_str = "Chưa chạy lần nào"
        if last_update_ts_copy > 0:
            try: last_update_str = datetime.datetime.fromtimestamp(last_update_ts_copy).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError: pass # Invalid timestamp format

        next_scheduled_str = "Không có cập nhật đang chờ hoặc chưa chạy lần nào"
        # Determine next scheduled time for batch processing
        if pending_count > 0 or last_update_ts_copy == 0 : # If pending or never run
            # Estimate next run time for batch job
            next_scheduled_ts = (last_update_ts_copy + BATCH_UPDATE_INTERVAL) if last_update_ts_copy > 0 else time.time() # If never run, assume "now" as base
            try: next_scheduled_str = datetime.datetime.fromtimestamp(next_scheduled_ts).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError: pass
            if last_update_ts_copy == 0 and pending_count > 0:
                next_scheduled_str += " (ước tính, sẽ chạy sớm nếu có pending và models sẵn sàng)"

        return jsonify({
            "success": True,
            "cache_file_exists": cache_exists,
            "metadata_file_exists": metadata_exists,
            "cache_file_size_bytes": cache_size_bytes,
            "cache_file_size_human": f"{cache_size_bytes / (1024*1024):.2f} MB" if cache_size_bytes > 0 else "0 B",
            "total_students_in_memory": student_count_mem,
            "students_with_embeddings_in_memory": students_with_embeddings_mem,
            "coverage_percentage_in_memory": f"{(students_with_embeddings_mem / student_count_mem * 100) if student_count_mem > 0 else 0:.2f}%",
            "pending_updates_count": pending_count,
            "pending_student_ids": pending_list_copy,
            "last_batch_update_timestamp": last_update_str,
            "next_scheduled_batch_update_check": next_scheduled_str,
            "batch_update_interval_seconds": BATCH_UPDATE_INTERVAL
        })
    except Exception as e:
        print(f"Lỗi API /api/cache_status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi server: {str(e)}"}), 500

# --- CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    print("Bắt đầu khởi tạo ứng dụng...")

    initialize_models_only() # QUAN TRỌNG: Khởi tạo model chính ở đây SỚM

    load_event_history_from_file() # Load history early

    # Tải dữ liệu học sinh và encoding ban đầu SAU KHI model sẵn sàng
    # This needs MODELS_READY to be true. initialize_models_only sets it.
    if MODELS_READY:
        initialize_data_and_face_encodings()
    else:
        print("LỖI NGHIÊM TRỌNG: Models AI chính không khởi tạo được. Dữ liệu học sinh và embeddings sẽ không được tải.")
        # Application might be in a non-functional state for AI processing.
        # Consider if the app should exit or run in a degraded mode.

    print(f"Diễn giải hướng camera khi FLIP_CAMERA_HORIZONTAL={FLIP_CAMERA_HORIZONTAL}")
    print("Khởi động Flask server với SocketIO...")
    print("Thread AI cho xử lý video sẽ bắt đầu (và chờ models nếu cần).")

    ai_thread = threading.Thread(target=process_video_and_count, daemon=True)
    ai_thread.start()    
    try:
        # use_reloader=False is important for multi-threaded/background task apps to avoid re-init issues
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Đang tắt ứng dụng (KeyboardInterrupt)...")
        if executor:
            print("Đang chờ các tác vụ nền hoàn tất...")
            executor.shutdown(wait=True) # Chờ các tác vụ trong executor xong
    except Exception as e_flask:
        print(f"Lỗi không mong muốn khi chạy Flask app: {e_flask}")
        import traceback
        traceback.print_exc()
    finally:
        print("Ứng dụng đang thoát...")
        if executor and not executor._shutdown: # Check if not already shut down
            print("Đảm bảo executor đã tắt.")
            executor.shutdown(wait=False) # Non-blocking shutdown if not already done

        # Save cache one last time if models were ready (might catch unsaved changes)
        if MODELS_READY:
            print("Lưu cache embeddings lần cuối (nếu có thay đổi)...")
            save_embeddings_cache()
        else:
             print("Models FaceNet không được khởi tạo, không thể lưu cache embeddings khi thoát.")
        print("Đã thực hiện các tác vụ dọn dẹp. Thoát ứng dụng.")