import gspread
import json
import time
import os
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# from google.oauth2.service_account import Credentials # Nếu dùng google-auth
# import google.auth # Nếu dùng google-auth

# --- Cấu hình từ biến môi trường với giá trị mặc định ---
# Nên sử dụng đường dẫn tuyệt đối trong container, ví dụ /app/data/file.json
JSON_FILE_PATH_IN_CONTAINER = os.getenv("SYNC_JSON_FILE", "API\students_list.json")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "database")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "sheet1")
SERVICE_ACCOUNT_FILE_IN_CONTAINER = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "secrets\credentials.json")
UNIQUE_ID_COLUMN_HEADER = os.getenv("UNIQUE_ID_COLUMN_HEADER", "") # Để trống nếu ghi đè toàn bộ
# -----------------

# Biến toàn cục
last_known_json_hash = ""
gc_instance = None

def get_gspread_client():
    """Khởi tạo và trả về gspread client, tái sử dụng nếu đã có."""
    global gc_instance
    if gc_instance is None:
        try:
            print("Đang khởi tạo Google Sheets client...")
            gc_instance = gspread.service_account(filename=SERVICE_ACCOUNT_FILE_IN_CONTAINER)
            print("Google Sheets client đã được khởi tạo.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo Google Sheets client: {e}")
            gc_instance = None
    return gc_instance

def get_file_hash(filepath):
    """Tạo hash MD5 cho nội dung file để kiểm tra thay đổi."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None

def sheet_to_json_list(worksheet):
    """
    Chuyển đổi dữ liệu từ worksheet thành list of dictionaries.
    CỐ GẮNG ÉP TẤT CẢ GIÁ TRỊ THÀNH CHUỖI.
    """
    try:
        print("DEBUG: Đang cố gắng gọi worksheet.get_all_values()")
        # Lấy tất cả giá trị dưới dạng list of lists.
        # get_all_values() thường trả về tất cả dưới dạng chuỗi nếu ô được định dạng là Plain Text.
        # Nếu không, nó có thể trả về số cho các ô trông giống số.
        all_values = worksheet.get_all_values() # Bao gồm cả header
        
        if not all_values:
            print("DEBUG: Worksheet không có dữ liệu (hoặc get_all_values trả về rỗng).")
            return []

        headers = [str(header) for header in all_values[0]] # Dòng đầu tiên là header, ép kiểu sang string
        records = []
        for i in range(1, len(all_values)): # Bắt đầu từ dòng 1 (sau header)
            row_values = all_values[i]
            record = {}
            for col_idx, header_name in enumerate(headers):
                # Đảm bảo tất cả giá trị được ép kiểu thành chuỗi
                # Ngay cả khi gspread trả về số, chúng ta cũng chuyển nó thành chuỗi
                if col_idx < len(row_values):
                    record[header_name] = str(row_values[col_idx])
                else:
                    record[header_name] = "" # Nếu hàng đó thiếu cột (không nên xảy ra với get_all_values)
            records.append(record)
        
        print(f"DEBUG: sheet_to_json_list xử lý {len(records)} bản ghi. 5 bản ghi đầu: {records[:5] if records else 'Không có records'}")
        return records
    except Exception as e:
        import traceback
        print(f"Lỗi trong sheet_to_json_list: {e}")
        print(traceback.format_exc())
        return None

def initialize_json_from_sheet(force_overwrite=True):
    """
    Lấy dữ liệu từ Google Sheet và ghi vào file JSON.
    Nếu force_overwrite là True, sẽ luôn ghi đè file JSON hiện có.
    """
    global last_known_json_hash

    if not JSON_FILE_PATH_IN_CONTAINER:
        print("LỖI NGHIÊM TRỌNG: JSON_FILE_PATH_IN_CONTAINER không được định nghĩa hoặc rỗng!")
        return False
    print(f"DEBUG: Bắt đầu initialize_json_from_sheet với JSON_FILE_PATH_IN_CONTAINER = '{JSON_FILE_PATH_IN_CONTAINER}', force_overwrite={force_overwrite}")

    gc = get_gspread_client()
    if not gc:
        print("Không thể kết nối Google Sheet để khởi tạo JSON.")
        return False

    # Đảm bảo thư mục chứa file JSON tồn tại
    json_dir = os.path.dirname(JSON_FILE_PATH_IN_CONTAINER)
    if json_dir and not os.path.exists(json_dir): # Chỉ tạo nếu json_dir có giá trị (không phải thư mục hiện tại)
        print(f"DEBUG: Thư mục '{json_dir}' không tồn tại, đang tạo...")
        try:
            os.makedirs(json_dir, exist_ok=True)
            print(f"DEBUG: Đã tạo thư mục '{json_dir}'.")
        except Exception as e_mkdir:
            import traceback
            print(f"Lỗi khi tạo thư mục '{json_dir}': {e_mkdir}")
            print(traceback.format_exc())
            return False # Không thể tiếp tục nếu không tạo được thư mục

    # Logic kiểm tra file tồn tại và force_overwrite
    if not force_overwrite and os.path.exists(JSON_FILE_PATH_IN_CONTAINER):
        print(f"File JSON '{JSON_FILE_PATH_IN_CONTAINER}' đã tồn tại và không ép buộc ghi đè. Bỏ qua bước khởi tạo từ Sheet.")
        last_known_json_hash = get_file_hash(JSON_FILE_PATH_IN_CONTAINER)
        return True
    
    # Nếu force_overwrite=True HOẶC file không tồn tại, thì tiến hành lấy từ sheet
    action_reason = "do ép buộc ghi đè." if force_overwrite else "do file chưa tồn tại."
    print(f"Đang lấy dữ liệu từ Google Sheet để ghi vào '{JSON_FILE_PATH_IN_CONTAINER}' {action_reason}")
    
    try:
        sh = gc.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)
        # Hàm sheet_to_json_list đã được cập nhật để ép kiểu thành chuỗi
        data_from_sheet = sheet_to_json_list(sh)

        if data_from_sheet is not None: # data_from_sheet có thể là [] nếu sheet rỗng (chỉ có header)
            print(f"DEBUG: Ghi {len(data_from_sheet)} bản ghi (đã ép kiểu chuỗi) vào '{JSON_FILE_PATH_IN_CONTAINER}'")
            with open(JSON_FILE_PATH_IN_CONTAINER, 'w', encoding='utf-8') as f:
                json.dump(data_from_sheet, f, ensure_ascii=False, indent=4)
            print(f"Đã ghi dữ liệu từ Google Sheet vào '{JSON_FILE_PATH_IN_CONTAINER}'.")
            last_known_json_hash = get_file_hash(JSON_FILE_PATH_IN_CONTAINER)
            return True
        else: # sheet_to_json_list trả về None do lỗi
            print("Không thể lấy dữ liệu từ sheet (hàm sheet_to_json_list trả về None). Không ghi file JSON.")
            # Không nên tạo file rỗng ở đây nếu việc lấy dữ liệu thất bại hoàn toàn
            return False 

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Lỗi: Không tìm thấy Google Sheet '{GOOGLE_SHEET_NAME}'.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"Lỗi: Không tìm thấy Worksheet '{WORKSHEET_NAME}'.")
    except Exception as e:
        import traceback
        print(f"Lỗi không xác định khi khởi tạo JSON từ Sheet: {e}")
        print(traceback.format_exc())
    return False # Đảm bảo hàm luôn trả về boolean

def update_sheet_from_json():
    """
    Đọc file JSON và cập nhật Google Sheet.
    ĐẢM BẢO TẤT CẢ DỮ LIỆU GHI LÊN SHEET LÀ CHUỖI.
    """
    global last_known_json_hash
    current_file_hash = get_file_hash(JSON_FILE_PATH_IN_CONTAINER)

    if current_file_hash is None:
        print(f"Lỗi: Không tìm thấy file JSON tại {JSON_FILE_PATH_IN_CONTAINER} để cập nhật Sheet.")
        return

    if last_known_json_hash and current_file_hash == last_known_json_hash:
        # print(f"File JSON tại {JSON_FILE_PATH_IN_CONTAINER} không thay đổi. Bỏ qua cập nhật Sheet.")
        return

    print(f"Phát hiện thay đổi trong file {JSON_FILE_PATH_IN_CONTAINER}. Đang cập nhật Google Sheet...")

    gc = get_gspread_client()
    if not gc:
        print("Không thể kết nối Google Sheet để cập nhật.")
        return

    try:
        sh = gc.open(GOOGLE_SHEET_NAME).worksheet(WORKSHEET_NAME)

        with open(JSON_FILE_PATH_IN_CONTAINER, 'r', encoding='utf-8') as f:
            json_data = json.load(f) # Dữ liệu từ JSON sẽ giữ nguyên kiểu (string, number, boolean)

        if not isinstance(json_data, list):
            print("Định dạng JSON không phải là một list các object. Bỏ qua cập nhật.")
            return

        sh.clear() 

        if not json_data: 
            print("File JSON rỗng. Đã xóa nội dung trên Google Sheet.")
            last_known_json_hash = current_file_hash 
            return

        # Lấy header từ item đầu tiên nếu có, ép kiểu header thành chuỗi
        headers = [str(key) for key in json_data[0].keys()] if json_data else []
        if not headers: # Nếu json_data là list rỗng sau khi clear() hoặc ban đầu đã rỗng
             print("File JSON không có dữ liệu (hoặc không có keys). Google Sheet sẽ trống.")
             last_known_json_hash = current_file_hash
             return

        rows_to_insert = [headers] # Dòng header (đã là chuỗi)
        for item in json_data: # item là một dict từ JSON
            row = []
            for header_name in headers:
                # Lấy giá trị từ dict, nếu không có thì để trống
                # Quan trọng: ÉP KIỂU TẤT CẢ GIÁ TRỊ THÀNH CHUỖI trước khi gửi lên Sheet
                value = item.get(str(header_name), "") # header_name đã là chuỗi
                row.append(str(value)) # Ép kiểu giá trị thành chuỗi
            rows_to_insert.append(row)

        # value_input_option='USER_ENTERED' giúp Google Sheets diễn giải dữ liệu như người dùng nhập,
        # nhưng việc ép kiểu thành chuỗi trong Python là một lớp bảo vệ tốt.
        sh.update('A1', rows_to_insert, value_input_option='USER_ENTERED') 
        print(f"Đã cập nhật thành công sheet '{WORKSHEET_NAME}' từ file JSON ({len(json_data)} bản ghi, đã ép kiểu chuỗi).")
        last_known_json_hash = current_file_hash

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file JSON tại {JSON_FILE_PATH_IN_CONTAINER} khi đang cập nhật Sheet.")
    except json.JSONDecodeError:
        print(f"Lỗi: File JSON tại {JSON_FILE_PATH_IN_CONTAINER} không hợp lệ.")
    except gspread.exceptions.APIError as e:
        print(f"Lỗi API Google Sheets: {e}")
    except Exception as e:
        import traceback
        print(f"Đã xảy ra lỗi không xác định khi cập nhật Sheet: {e}")
        print(traceback.format_exc())

class JSONChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and os.path.abspath(event.src_path) == os.path.abspath(JSON_FILE_PATH_IN_CONTAINER):
            print(f"Sự kiện: File {event.src_path} được sửa đổi.")
            time.sleep(0.5) # Đợi file ghi xong hoàn toàn
            update_sheet_from_json()

    def on_created(self, event):
         if not event.is_directory and os.path.abspath(event.src_path) == os.path.abspath(JSON_FILE_PATH_IN_CONTAINER):
            print(f"Sự kiện: File {event.src_path} được tạo mới.")
            time.sleep(0.5)
            update_sheet_from_json()

if __name__ == "__main__":
    print("--- Khởi động JSON-Sheet Sync Service ---")
    print(f"File JSON mục tiêu: {JSON_FILE_PATH_IN_CONTAINER}")
    print(f"Service account: {SERVICE_ACCOUNT_FILE_IN_CONTAINER}")
    print(f"Google Sheet: {GOOGLE_SHEET_NAME}, Worksheet: {WORKSHEET_NAME}")
    if UNIQUE_ID_COLUMN_HEADER:
        print(f"Cột ID duy nhất (chưa dùng cho cập nhật chi tiết): '{UNIQUE_ID_COLUMN_HEADER}'")
    else:
        print("Sẽ ghi đè toàn bộ sheet khi JSON thay đổi.")

    # Bước 1: Khởi tạo file JSON từ Google Sheet.
    # force_overwrite=True sẽ luôn ghi đè file JSON hiện có bằng dữ liệu từ Sheet.
    # force_overwrite=False sẽ chỉ ghi nếu file JSON chưa tồn tại.
    FORCE_OVERWRITE_ON_STARTUP = True # Đặt thành True để luôn lấy từ Sheet khi khởi động
    print(f"Chế độ ghi đè khi khởi động (FORCE_OVERWRITE_ON_STARTUP): {FORCE_OVERWRITE_ON_STARTUP}")

    if not initialize_json_from_sheet(force_overwrite=FORCE_OVERWRITE_ON_STARTUP):
        if FORCE_OVERWRITE_ON_STARTUP:
            print("CẢNH BÁO: Khởi tạo JSON từ Sheet (ép buộc ghi đè) không thành công. Kiểm tra log lỗi.")
        else:
            print("Thông tin: Khởi tạo JSON từ Sheet (không ép buộc ghi đè) không thực hiện hoặc thất bại. Có thể file đã tồn tại.")
        # Quyết định có nên thoát hay không nếu khởi tạo thất bại
        # exit(1) # Bỏ comment nếu muốn thoát khi khởi tạo thất bại nghiêm trọng
    
    # Bước 2: Thiết lập theo dõi file JSON
    event_handler = JSONChangeHandler()
    observer = Observer()
    
    # Đảm bảo đường dẫn theo dõi (thư mục chứa file JSON) tồn tại
    watch_path = os.path.dirname(os.path.abspath(JSON_FILE_PATH_IN_CONTAINER))
    if not os.path.exists(watch_path):
        print(f"LỖI: Thư mục để theo dõi '{watch_path}' không tồn tại. Kiểm tra SYNC_JSON_FILE và volume mounts.")
        # Thử tạo thư mục nếu nó chưa tồn tại và là một phần của đường dẫn hợp lệ
        if json_dir: # json_dir được định nghĩa trong initialize_json_from_sheet
            try:
                print(f"Thử tạo thư mục theo dõi '{watch_path}'...")
                os.makedirs(watch_path, exist_ok=True)
                print(f"Đã tạo thư mục theo dõi '{watch_path}'.")
            except Exception as e_mkdir_watch:
                print(f"Lỗi khi tạo thư mục theo dõi '{watch_path}': {e_mkdir_watch}")
                exit(1)
        else: # Nếu watch_path là thư mục hiện tại và không tồn tại, đó là lỗi nghiêm trọng
            exit(1)


    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()
    print(f"Đang theo dõi thay đổi trong thư mục '{watch_path}' cho file '{os.path.basename(JSON_FILE_PATH_IN_CONTAINER)}'...")

    try:
        while True:
            if get_gspread_client() is None:
                print("Đang thử kết nối lại với Google Sheets...")
                time.sleep(10) 
            else:
                time.sleep(5)
    except KeyboardInterrupt:
        print("Đang dừng dịch vụ...")
    except Exception as e:
        import traceback
        print(f"Lỗi nghiêm trọng trong vòng lặp chính: {e}")
        print(traceback.format_exc())
    finally:
        observer.stop()
        observer.join()
        print("Dịch vụ đã dừng.")