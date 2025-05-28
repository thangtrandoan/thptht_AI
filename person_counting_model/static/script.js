// Biến toàn cục
let currentViewingStudentId = null;
let selectedAvatarFile = null;
let mediaStream = null; // Lưu trữ MediaStream để có thể tắt
let streamIntervalId = null; // Để lưu interval gửi frame
let isCameraOn = false; // Trạng thái bật/tắt camera
let activeSubmitButton = null; // Added
let originalActiveButtonText = ''; // Added

// --- WEBSOCKET VÀ CAMERA ---
const webcamFeed = document.getElementById('webcamFeed');
const hiddenFrameCanvas = document.getElementById('hiddenFrameCanvas'); // Canvas để lấy frame gửi đi
const hiddenFrameCtx = hiddenFrameCanvas.getContext('2d', { willReadFrequently: true });
const overlayCanvas = document.getElementById('overlayCanvas'); // Canvas để vẽ bounding box
const overlayCtx = overlayCanvas.getContext('2d');
const toggleCameraButton = document.getElementById('toggleCameraButton');

const socket = io();
const FPS_TO_SERVER = 10; // Gửi 10 FPS lên server

// Hàm bắt đầu và kết thúc loading cho một nút:
function startButtonLoading(button, loadingText = "Đang xử lý...") {
    if (activeSubmitButton && activeSubmitButton !== button) { // Nếu có nút khác đang load, khôi phục nó trước
        stopButtonLoading();
    }
    activeSubmitButton = button;
    originalActiveButtonText = button.innerHTML;
    button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${loadingText}`;
    button.disabled = true;
}

function stopButtonLoading() {
    if (activeSubmitButton) {
        activeSubmitButton.innerHTML = originalActiveButtonText;
        activeSubmitButton.disabled = false;
        activeSubmitButton = null;
        originalActiveButtonText = '';
    }
}


// Xử lý nút Bật/Tắt Camera
if (toggleCameraButton) {
    toggleCameraButton.addEventListener('click', () => {
        if (isCameraOn) {
            stopWebcam();
        } else {
            startWebcam();
        }
    });
}

function updateToggleButton(isOn) {
    if (toggleCameraButton) { // Ensure button exists
        if (isOn) {
            toggleCameraButton.innerHTML = '<i class="bi bi-camera-video-off-fill"></i> Tắt Camera';
            toggleCameraButton.classList.remove('btn-success');
            toggleCameraButton.classList.add('btn-danger');
        } else {
            toggleCameraButton.innerHTML = '<i class="bi bi-camera-video-fill"></i> Bật Camera';
            toggleCameraButton.classList.remove('btn-danger');
            toggleCameraButton.classList.add('btn-success');
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Xóa vẽ khi tắt cam
        }
    }
}

async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Trình duyệt của bạn không hỗ trợ API truy cập camera.");
        return;
    }
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 }, // Yêu cầu kích thước mong muốn
                height: { ideal: 480 },
                // facingMode: "user" // hoặc "environment" cho camera sau
            },
            audio: false
        });
        webcamFeed.srcObject = mediaStream;
        webcamFeed.onloadedmetadata = () => {
            console.log("Camera trình duyệt đã sẵn sàng.");
            // Đặt kích thước cho cả hai canvas theo video
            hiddenFrameCanvas.width = webcamFeed.videoWidth;
            hiddenFrameCanvas.height = webcamFeed.videoHeight;
            overlayCanvas.width = webcamFeed.videoWidth;
            overlayCanvas.height = webcamFeed.videoHeight;

            isCameraOn = true;
            updateToggleButton(true);
            startStreamingFrames();
        };
    } catch (err) {
        console.error("Lỗi truy cập camera trình duyệt:", err);
        alert("Không thể truy cập camera. Vui lòng kiểm tra quyền và thử lại.\n" + err.message);
        isCameraOn = false;
        updateToggleButton(false);
    }
}

function stopWebcam() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        webcamFeed.srcObject = null;
        mediaStream = null;
    }
    if (streamIntervalId) {
        clearInterval(streamIntervalId);
        streamIntervalId = null;
    }
    isCameraOn = false;
    updateToggleButton(false);
    console.log("Camera đã tắt.");
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Xóa vẽ khi tắt cam
}

function startStreamingFrames() {
    if (streamIntervalId) clearInterval(streamIntervalId);

    streamIntervalId = setInterval(() => {
        if (isCameraOn && webcamFeed.readyState >= webcamFeed.HAVE_CURRENT_DATA && webcamFeed.videoWidth > 0 && socket.connected) {
            hiddenFrameCtx.drawImage(webcamFeed, 0, 0, hiddenFrameCanvas.width, hiddenFrameCanvas.height);
            const dataURL = hiddenFrameCanvas.toDataURL('image/jpeg', 0.7);
            socket.emit('video_frame_to_server', dataURL);
        }
    }, 1000 / FPS_TO_SERVER);
    console.log("Bắt đầu gửi frame lên server...");
}

socket.on('tracking_update_to_client', (data) => {
    if (!isCameraOn || !overlayCanvas || !webcamFeed) return;

    const videoWidth = webcamFeed.videoWidth;
    const videoHeight = webcamFeed.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) return;

    if (overlayCanvas.width !== videoWidth || overlayCanvas.height !== videoHeight) {
        overlayCanvas.width = videoWidth;
        overlayCanvas.height = videoHeight;
    }

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (data.counting_line_x_on_frame !== undefined) {
        const lineX = data.counting_line_x_on_frame;
        overlayCtx.save();
        overlayCtx.beginPath();
        overlayCtx.moveTo(lineX, 0);
        overlayCtx.lineTo(lineX, videoHeight);
        overlayCtx.strokeStyle = 'yellow';
        overlayCtx.lineWidth = 2;
        overlayCtx.stroke();
        overlayCtx.restore();
    }

    if (data.tracked_objects_on_frame) {
        data.tracked_objects_on_frame.forEach(obj => {
            const [x1, y1, x2, y2] = obj.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;

            let color = 'rgba(255, 165, 0, 0.8)';
            let displayText = obj.name || obj.student_id || `ID: ${obj.id_track ? obj.id_track.substring(0,5) : 'N/A'}`;


            if (obj.student_id) {
                color = 'rgba(0, 255, 0, 0.8)';
                displayText = `${obj.name} (HS)`;
            } else if (obj.name === 'Stranger') {
                color = 'rgba(255, 0, 0, 0.8)';
                displayText = "Người lạ";
            } else if (obj.name === 'Processing...') {
                displayText = "Đang xử lý...";
            }

            overlayCtx.save();
            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = 2;
            overlayCtx.strokeRect(x1, y1, boxWidth, boxHeight);
            overlayCtx.restore();

            overlayCtx.save();
            const textX = x1 + 5;
            const textY = y1 - 7;
            overlayCtx.translate(textX, textY);
            overlayCtx.scale(-1, 1);
            overlayCtx.fillStyle = color;
            overlayCtx.font = 'bold 12px Arial';
            overlayCtx.textAlign = 'left';
            overlayCtx.fillText(displayText, 0, 0);
            overlayCtx.restore();
        });
    }
});

socket.on('connect', () => {
    console.log('Đã kết nối tới server WebSocket:', socket.id);
    if (isCameraOn && !streamIntervalId) {
        startStreamingFrames();
    }
});

socket.on('disconnect', (reason) => {
    console.log('Đã ngắt kết nối khỏi server WebSocket:', reason);
    if (streamIntervalId) {
        clearInterval(streamIntervalId);
        streamIntervalId = null;
    }
});

socket.on('connect_error', (err) => {
    console.error('Lỗi kết nối WebSocket:', err);
    if (streamIntervalId) {
        clearInterval(streamIntervalId);
        streamIntervalId = null;
    }
});

// Lắng nghe sự kiện WebSocket model_update_notification
socket.on('model_update_notification', (data) => {
    console.log('Thông báo cập nhật model từ server:', data);
    stopButtonLoading(); // Dừng/khôi phục nút đang loading (nếu có)

    if (data.status === 'success') {
        showToast(data.message, 'success');
        // Các hành động cần thiết sau khi model cập nhật thành công:
        if (data.action === 'update_embeddings' || data.action === 'save_cache') {
            // Cập nhật lại danh sách học sinh để hiển thị trạng thái mới hoặc ảnh mới (nếu có)
            loadStudentsList();
            // Nếu đang xem chi tiết học sinh vừa được cập nhật, làm mới ảnh
            if (currentViewingStudentId && currentViewingStudentId === data.student_id) {
                const studentImage = document.getElementById('studentImage');
                if (studentImage) {
                     const timestamp = new Date().getTime();
                     studentImage.src = `/api/student_image/${currentViewingStudentId}?t=${timestamp}`;
                }
                // Reset nút change avatar nếu nó đang trong trạng thái "Đã chọn"
                const avatarBtn = document.getElementById('changeAvatarBtn');
                if (avatarBtn) {
                    avatarBtn.textContent = "Thay ảnh đại diện";
                    avatarBtn.classList.replace('btn-success', 'btn-outline-primary');
                    const studentImageElement = document.getElementById('studentImage');
                    if (studentImageElement) studentImageElement.classList.remove('border', 'border-success', 'border-3');
                    const existingNotice = document.querySelector('.save-avatar-notice');
                    if (existingNotice) existingNotice.remove();
                }
            }
        }
    } else { // data.status === 'error'
        showToast(`Lỗi cập nhật model: ${data.message}`, 'danger');
    }
});

// Hàm xử lý cập nhật trạng thái chính từ WebSocket
function handleMainStatusUpdate(data) {
    console.log("Received main_status_updated:", data);

    const totalPeopleEl = document.getElementById('totalPeopleServer');
    const strangersEl = document.getElementById('strangersOnBusServer');
    const lastEventEl = document.getElementById('lastEventServer');
    const netCountEl = document.getElementById("netCountDisplay");
    const trackedList = document.getElementById("trackedObjectsList");

    if (totalPeopleEl) totalPeopleEl.textContent = data.total_people_on_bus || 0;
    if (strangersEl) strangersEl.textContent = data.strangers_on_bus_count || 0;

    if (lastEventEl && data.last_event_triggered) {
        const event = data.last_event_triggered;
        lastEventEl.innerText = `${event.name} ${event.action}` || '-';
    } else if (lastEventEl) {
        lastEventEl.innerText = '-'; // Fallback nếu không có thông tin sự kiện cụ thể
    }


    if (trackedList) {
        trackedList.innerHTML = ""; // Xóa danh sách cũ
        let currentStudentsOnBusCount = 0;
        if (data.students_status) { // data.students_status là một object {student_id: {name: ..., status: ...}}
            const studentsOnBus = Object.entries(data.students_status)
                .filter(([_, info]) => info.status === "in");

            currentStudentsOnBusCount = studentsOnBus.length;

            if (studentsOnBus.length === 0) {
                trackedList.innerHTML = "<li>Chưa có học sinh nào trên xe.</li>";
            } else {
                studentsOnBus.forEach(([id, info]) => {
                    const li = document.createElement("li");
                    li.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <span>${info.name || 'Không rõ tên'} (ID: ${id})</span>
                            <span class="badge bg-success">Trên xe</span>
                        </div>
                    `;
                    trackedList.appendChild(li);
                });
            }
        } else {
             trackedList.innerHTML = "<li>Dữ liệu trạng thái học sinh không có.</li>";
        }
        if (netCountEl) netCountEl.textContent = currentStudentsOnBusCount;
    }
}


// Lắng nghe sự kiện WebSocket main_status_updated từ server
socket.on('main_status_updated', (data) => {
    handleMainStatusUpdate(data); // Cập nhật các thông tin chính
    updateHistory(); // Cập nhật tab lịch sử
    loadStudentsList(); // Cập nhật tab danh sách học sinh (để badge trạng thái đúng)
});


// --- CẬP NHẬT THÔNG TIN TỪ SERVER (HTTP Polling - giữ lại làm fallback/initial load) ---
function updateServerTrackingInfo() {
    fetch('/student_management_info')
        .then(response => response.json())
        .then(data => {
            // Dùng hàm handleMainStatusUpdate để tránh lặp code
            // Tuy nhiên, /student_management_info trả về cấu trúc hơi khác (có original_data)
            // Nên ta sẽ điều chỉnh lại hoặc chỉ cập nhật các phần tử UI trực tiếp ở đây
            // như cách làm cũ, để WebSocket là nguồn chính cho cập nhật "live".
            
            const totalPeopleEl = document.getElementById('totalPeopleServer');
            const strangersEl = document.getElementById('strangersOnBusServer');
            const lastEventEl = document.getElementById('lastEventServer');
            const netCountEl = document.getElementById("netCountDisplay");

            if (totalPeopleEl) totalPeopleEl.textContent = data.total_people_on_bus || 0;
            if (strangersEl) strangersEl.textContent = data.strangers_on_bus_count || 0;
            if (lastEventEl) lastEventEl.innerText = data.last_event_message || '-'; // HTTP endpoint dùng last_event_message


            const trackedList = document.getElementById("trackedObjectsList");
            if (trackedList) {
                trackedList.innerHTML = ""; // Clear previous list
                const studentsOnBus = Object.entries(data.students_status || {})
                    .filter(([_, info]) => info.status === "in");
                
                if (netCountEl) netCountEl.textContent = studentsOnBus.length;

                if (studentsOnBus.length === 0) {
                    trackedList.innerHTML = "<li>Chưa có học sinh nào trên xe.</li>";
                } else {
                    studentsOnBus.forEach(([id, info]) => {
                        const li = document.createElement("li");
                        li.innerHTML = `
                            <div class="d-flex justify-content-between align-items-center">
                                <span>${info.name || 'Không rõ tên'} (ID: ${id})</span>
                                <span class="badge bg-success">Trên xe</span>
                            </div>
                        `;
                        trackedList.appendChild(li);
                    });
                }
            }
        })
        .catch(error => console.error('Lỗi khi lấy thông tin tracking từ server (HTTP):', error));
}

// --- QUẢN LÝ TAB (GIỮ NGUYÊN) ---
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-tabs a').forEach(tab => {
        tab.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelectorAll('.nav-tabs li').forEach(li => li.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            this.parentElement.classList.add('active');
            const targetId = this.getAttribute('href');
            const targetContent = document.querySelector(targetId);
            if (targetContent) targetContent.classList.add('active');

            if (targetId === '#history') updateHistory();
            else if (targetId === '#list') loadStudentsList();
        });
    });
});

// --- TAB 2: LỊCH SỬ (GIỮ NGUYÊN) ---
async function updateHistory() {
    const historyList = document.getElementById("historyList");
    if (!historyList) return; // Element might not be present on all pages/tabs
    try {
        const res = await fetch('/api/event_history');
        const history = await res.json();
        historyList.innerHTML = "";
        history
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .forEach(event => {
                const li = document.createElement("li");
                li.innerHTML = `
                    <div class="name">
                        <strong>${event.name}</strong><br>
                        <small>ID: ${event.id ? event.id.substring(0,8) : 'N/A'}</small>
                    </div>
                    <span class="status ${event.action === 'Lên xe' ? 'in' : 'out'}">${event.action}</span>
                    <span class="timestamp">${new Date(event.timestamp).toLocaleTimeString()}</span>
                `;
                historyList.appendChild(li);
            });
    } catch (err) {
        console.error("Lỗi khi tải lịch sử:", err);
        historyList.innerHTML = "<li>Không thể tải lịch sử.</li>";
    }
}

// --- TAB 3: DANH SÁCH HỌC SINH VÀ QUẢN LÝ ---
async function loadStudentsList() {
    try {
        const response = await fetch('/api/students');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const students = await response.json();
        const studentsListContainer = document.querySelector('#list .hoc-sinh');
        if (!studentsListContainer) return;

        let tableContainer = studentsListContainer.querySelector('.table-container');
        if (tableContainer) tableContainer.remove(); // Remove old table if exists
        
        tableContainer = document.createElement('div');
        tableContainer.className = 'table-container';
        const table = document.createElement('table');
        table.className = 'table table-striped table-hover';
        table.innerHTML = `
            <thead>
                <tr>
                    <th width="15%">Mã HS</th><th width="35%">Họ tên</th><th width="20%">Trạng thái</th>
                    <th width="30%" style="text-align: center;">Thao tác</th>
                </tr>
            </thead>
            <tbody id="studentsTableBody"></tbody>
        `;
        tableContainer.appendChild(table);
        studentsListContainer.appendChild(tableContainer);
        
        const tbody = document.getElementById('studentsTableBody');
        tbody.innerHTML = ''; // Clear previous content

        if (students.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center">Chưa có học sinh nào.</td></tr>';
        } else {
            students.forEach(student => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${student.id}</td><td>${student.name}</td>
                    <td><span class="badge ${student.status === 'in' ? 'bg-success' : 'bg-secondary'}">${student.status === 'in' ? 'Trên xe' : 'Ngoài xe'}</span></td>
                    <td class="text-center">
                        <div class="action-buttons d-flex justify-content-center gap-2">
                            <button class="btn btn-sm btn-info view-student-btn" data-studentid="${student.id}"><i class="bi bi-eye"></i> Xem</button>
                            <button class="btn btn-sm btn-danger delete-student-btn" data-studentid="${student.id}" data-studentname="${student.name}"><i class="bi bi-trash"></i></button>
                        </div>
                    </td>`;
                tbody.appendChild(tr);
            });
        }
        attachActionListeners();
    } catch (error) {
        console.error('Lỗi khi tải danh sách học sinh:', error);
        const tbody = document.getElementById('studentsTableBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Lỗi tải DS học sinh.</td></tr>';
    }
}

function attachActionListeners() {
    document.querySelectorAll('.view-student-btn').forEach(btn => {
        btn.removeEventListener('click', handleViewStudent); // Remove old before adding new
        btn.addEventListener('click', handleViewStudent);
    });
    document.querySelectorAll('.delete-student-btn').forEach(btn => {
        btn.removeEventListener('click', handleDeleteStudent);
        btn.addEventListener('click', handleDeleteStudent);
    });
}
function handleViewStudent(event) { viewStudentDetails(event.currentTarget.dataset.studentid); }
function handleDeleteStudent(event) { deleteStudent(event.currentTarget.dataset.studentid, event.currentTarget.dataset.studentname); }

// Xử lý form thêm học sinh
const studentForm = document.getElementById('addStudentForm');
if (studentForm) {
    studentForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const studentIdInput = document.getElementById('inputStudentId');
        const studentNameInput = document.getElementById('inputStudentName');
        const studentClassInput = document.getElementById('inputStudentClass');
        const studentAgeInput = document.getElementById('inputStudentAge');
        const studentAddressInput = document.getElementById('inputStudentAddress');
        const fatherNameInput = document.getElementById('inputFatherName');
        const fatherAgeInput = document.getElementById('inputFatherAge');
        const fatherPhoneInput = document.getElementById('inputFatherPhone');
        const motherNameInput = document.getElementById('inputMotherName');
        const motherAgeInput = document.getElementById('inputMotherAge');
        const motherPhoneInput = document.getElementById('inputMotherPhone');
        const imageFilesInput = document.getElementById('inputStudentImages');


        if (!studentIdInput.value || !studentNameInput.value || !studentClassInput.value ) {
            showToast('Vui lòng điền đầy đủ các trường Mã HS, Họ Tên và Lớp.', 'warning');
            return;
        }

        const imageFiles = imageFilesInput.files;
        if (!imageFiles || imageFiles.length === 0) {
            showToast('Vui lòng chọn ít nhất một ảnh cho học sinh.', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('student_id', studentIdInput.value);
        formData.append('name', studentNameInput.value);
        formData.append('class', studentClassInput.value);
        formData.append('age', studentAgeInput.value);
        formData.append('address', studentAddressInput.value);
        formData.append('father_name', fatherNameInput.value);
        formData.append('father_age', fatherAgeInput.value);
        formData.append('father_phone', fatherPhoneInput.value);
        formData.append('mother_name', motherNameInput.value);
        formData.append('mother_age', motherAgeInput.value);
        formData.append('mother_phone', motherPhoneInput.value);

        for (let i = 0; i < imageFiles.length; i++) {
            formData.append('images', imageFiles[i]);
        }

        const submitBtn = this.querySelector('button[type="submit"]');
        startButtonLoading(submitBtn, "Đang gửi yêu cầu..."); // Hiển thị loading

        try {
            const response = await fetch('/api/add_student', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (response.ok && result.success) {
                showToast(result.message, 'info'); // Ví dụ: "Yêu cầu đã tiếp nhận..."
                // Không stopButtonLoading() ở đây, đợi WebSocket
                const modal = bootstrap.Modal.getInstance(document.getElementById('addStudentModal'));
                if (modal) modal.hide();
                studentForm.reset();
                const previewContainer = document.getElementById('imagePreviewContainer');
                if (previewContainer) previewContainer.innerHTML = '';
                // loadStudentsList() sẽ được gọi bởi socket event 'model_update_notification'
            } else {
                stopButtonLoading();
                showToast(`Lỗi gửi yêu cầu: ${result.message || 'Không thể gửi yêu cầu.'}`, 'danger');
            }
        } catch (error) {
            console.error('Lỗi khi gửi form thêm học sinh:', error);
            stopButtonLoading();
            showToast('Có lỗi mạng xảy ra khi thêm học sinh. Vui lòng thử lại.', 'danger');
        }
    });
}

// Xem chi tiết học sinh (GIỮ NGUYÊN)
async function viewStudentDetails(studentId) { 
    currentViewingStudentId = studentId;
    try {
        const response = await fetch(`/api/students/${studentId}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const student = await response.json();
        if (student.error) { showToast(`Lỗi: ${student.error}`, 'danger'); return; }
        selectedAvatarFile = null; 
        const avatarFileInput = document.getElementById('avatarFileInput');
        if (avatarFileInput) avatarFileInput.value = '';

        document.getElementById('studentName').textContent = student.name || 'N/A';
        document.getElementById('studentClass').textContent = student.original_data?.class || 'N/A';
        document.getElementById('studentAge').textContent = student.original_data?.age || 'N/A';
        document.getElementById('studentAddress').textContent = student.original_data?.address || 'N/A';
        document.getElementById('fatherName').textContent = student.original_data?.father_name || 'N/A';
        document.getElementById('fatherAge').textContent = student.original_data?.father_age || 'N/A';
        document.getElementById('fatherPhone').textContent = student.original_data?.father_phone || 'N/A';
        document.getElementById('motherName').textContent = student.original_data?.mother_name || 'N/A';
        document.getElementById('motherAge').textContent = student.original_data?.mother_age || 'N/A';
        document.getElementById('motherPhone').textContent = student.original_data?.mother_phone || 'N/A';
        
        const timestamp = new Date().getTime();
        const studentImage = document.getElementById('studentImage');
        studentImage.src = `/api/student_image/${studentId}?t=${timestamp}`;
        studentImage.classList.remove('border', 'border-success', 'border-3');
        
        const avatarBtn = document.getElementById('changeAvatarBtn');
        if (avatarBtn) {
            avatarBtn.textContent = "Thay ảnh đại diện";
            avatarBtn.classList.remove('btn-success'); avatarBtn.classList.add('btn-outline-primary');
            const existingNotice = avatarBtn.parentNode.querySelector('.text-success.small.mt-1');
            if (existingNotice) existingNotice.remove();
        }
        
        const studentModal = new bootstrap.Modal(document.getElementById('studentModal'));
        studentModal.show();
    } catch (error) { console.error('Lỗi xem chi tiết:', error); showToast('Lỗi tải chi tiết HS.', 'danger'); }
}

// Sự kiện nút Sửa trong modal Xem chi tiết (GIỮ NGUYÊN)
const editStudentInfoButton = document.getElementById('editStudentInfoBtn');
if (editStudentInfoButton) {
    editStudentInfoButton.addEventListener('click', function () { 
        if (!currentViewingStudentId) return;
        document.getElementById('editStudentId').value = currentViewingStudentId;
        document.getElementById('editStudentName').value = document.getElementById('studentName').textContent;
        document.getElementById('editStudentClass').value = document.getElementById('studentClass').textContent;
        document.getElementById('editStudentAge').value = document.getElementById('studentAge').textContent;
        document.getElementById('editStudentAddress').value = document.getElementById('studentAddress').textContent;
        document.getElementById('editFatherName').value = document.getElementById('fatherName').textContent;
        document.getElementById('editFatherAge').value = document.getElementById('fatherAge').textContent;
        document.getElementById('editFatherPhone').value = document.getElementById('fatherPhone').textContent;
        document.getElementById('editMotherName').value = document.getElementById('motherName').textContent;
        document.getElementById('editMotherAge').value = document.getElementById('motherAge').textContent;
        document.getElementById('editMotherPhone').value = document.getElementById('motherPhone').textContent;
        
        const studentModalInstance = bootstrap.Modal.getInstance(document.getElementById('studentModal'));
        if (studentModalInstance) studentModalInstance.hide();
        
        const editModal = new bootstrap.Modal(document.getElementById('editStudentModal'));
        editModal.show();
    });
}

// Xử lý form sửa thông tin học sinh (GIỮ NGUYÊN)
const editStudentForm = document.getElementById('editStudentForm');
if (editStudentForm) {
    editStudentForm.addEventListener('submit', async function (e) { 
        e.preventDefault();
        const studentId = document.getElementById('editStudentId').value;
        const studentData = {
            name: document.getElementById('editStudentName').value, class: document.getElementById('editStudentClass').value,
            age: document.getElementById('editStudentAge').value, address: document.getElementById('editStudentAddress').value,
            father_name: document.getElementById('editFatherName').value, father_age: document.getElementById('editFatherAge').value,
            father_phone: document.getElementById('editFatherPhone').value, mother_name: document.getElementById('editMotherName').value,
            mother_age: document.getElementById('editMotherAge').value, mother_phone: document.getElementById('editMotherPhone').value
        };
        const submitBtn = this.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang lưu...'; submitBtn.disabled = true;
        try {
            const response = await fetch(`/api/students/${studentId}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(studentData) });
            const result = await response.json();
            if (response.ok && result.success) {
                showToast('Cập nhật thành công!', 'success');
                const editModalInstance = bootstrap.Modal.getInstance(document.getElementById('editStudentModal'));
                if (editModalInstance) editModalInstance.hide();
                loadStudentsList(); 
                viewStudentDetails(studentId); // Refresh detail view
            } else { showToast(`Lỗi: ${result.message || 'Không thể cập nhật'}`, 'danger'); }
        } catch (error) { console.error('Lỗi form cập nhật:', error); showToast('Lỗi cập nhật.', 'danger');
        } finally { submitBtn.innerHTML = originalBtnText; submitBtn.disabled = false; }
    });
}

// Xử lý thay đổi avatar
const changeAvatarButton = document.getElementById('changeAvatarBtn');
if (changeAvatarButton) {
    changeAvatarButton.addEventListener('click', function () { 
        const avatarFileInput = document.getElementById('avatarFileInput');
        if(avatarFileInput) avatarFileInput.click();
    });
}

const avatarFileInput = document.getElementById('avatarFileInput');
if (avatarFileInput) {
    avatarFileInput.addEventListener('change', function () { 
        if (this.files && this.files[0]) {
            selectedAvatarFile = this.files[0];
            const reader = new FileReader();
            reader.onload = function (e) {
                const studentImage = document.getElementById('studentImage');
                studentImage.src = e.target.result; 
                studentImage.classList.add('border', 'border-success', 'border-3');
                
                const avatarBtnElement = document.getElementById('changeAvatarBtn');
                avatarBtnElement.textContent = "Đã chọn ảnh mới ✓"; 
                avatarBtnElement.classList.replace('btn-outline-primary', 'btn-success');
                
                let saveNotice = avatarBtnElement.parentNode.querySelector('.save-avatar-notice');
                if (!saveNotice) {
                    saveNotice = document.createElement('div');
                    saveNotice.className = 'text-success small mt-1 save-avatar-notice';
                    avatarBtnElement.parentNode.insertBefore(saveNotice, avatarBtnElement.nextSibling);
                }
                saveNotice.innerHTML = 'Ảnh sẽ tự động lưu khi đóng modal.';
            };
            reader.readAsDataURL(selectedAvatarFile);
        }
    });
}

// Sự kiện khi đóng modal chi tiết học sinh - cập nhật ảnh
const studentModalElement = document.getElementById('studentModal');
if (studentModalElement) {
    studentModalElement.addEventListener('hidden.bs.modal', async function () {
        const changeAvatarButtonElement = document.getElementById('changeAvatarBtn');

        if (selectedAvatarFile && currentViewingStudentId) {
            const studentId = currentViewingStudentId;
            const formData = new FormData();
            formData.append('avatar', selectedAvatarFile);

            if (changeAvatarButtonElement) {
                startButtonLoading(changeAvatarButtonElement, "Đang cập nhật ảnh...");
            } else {
                showToast('Đang cập nhật ảnh và model...', 'info', false);
            }

            try {
                const response = await fetch(`/api/students/${studentId}/avatar`, {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                // Không stopButtonLoading() ở đây, đợi WebSocket
                if (response.ok && result.success) {
                    showToast(result.message, 'info'); // "Yêu cầu đã tiếp nhận..."
                    // loadStudentsList() và cập nhật ảnh sẽ do WebSocket 'model_update_notification' xử lý
                } else {
                    stopButtonLoading();
                    showToast(`Lỗi cập nhật avatar: ${result.message || 'Không thể gửi yêu cầu.'}`, 'danger');
                }
            } catch (error) {
                console.error('Lỗi khi gửi ảnh đại diện mới:', error);
                stopButtonLoading();
                showToast('Có lỗi mạng xảy ra khi cập nhật ảnh. Vui lòng thử lại.', 'danger');
            }
        }

        // Reset các trạng thái của avatar (chỉ nếu không có lỗi và nút không loading)
        selectedAvatarFile = null;
        const avatarFileInputElement = document.getElementById('avatarFileInput');
        if (avatarFileInputElement) avatarFileInputElement.value = '';

        if (changeAvatarButtonElement && !changeAvatarButtonElement.disabled) { // Chỉ reset nếu không loading
            changeAvatarButtonElement.textContent = "Thay ảnh đại diện";
            changeAvatarButtonElement.classList.replace('btn-success', 'btn-outline-primary');
        }
        const studentImageElement = document.getElementById('studentImage');
        if (studentImageElement) studentImageElement.classList.remove('border', 'border-success', 'border-3');
        const existingNotice = document.querySelector('.save-avatar-notice');
        if (existingNotice) existingNotice.remove();
    });
}

// Hàm xóa học sinh
async function deleteStudent(studentId, studentName) {
    if (!confirm(`Xóa học sinh ${studentName} (ID: ${studentId})? Model sẽ được cập nhật trong nền.`)) {
        return;
    }

    const deleteButton = document.querySelector(`.delete-student-btn[data-studentid="${studentId}"]`);
    if (deleteButton) {
        startButtonLoading(deleteButton, "Đang xóa...");
    } else {
        showToast(`Đang xử lý yêu cầu xóa HS ${studentName}...`, 'info', false);
    }

    try {
        const response = await fetch(`/api/students/${studentId}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok && result.success) {
            showToast(result.message, 'info'); // "Yêu cầu đã tiếp nhận..."
            // Không stopButtonLoading() ở đây, đợi WebSocket
            // loadStudentsList() sẽ được gọi bởi socket event
        } else {
            stopButtonLoading();
            showToast(`Lỗi khi gửi yêu cầu xóa: ${result.message || 'Không thể gửi yêu cầu.'}`, 'danger');
        }
    } catch (error) {
        console.error('Lỗi khi xóa học sinh:', error);
        stopButtonLoading();
        showToast('Có lỗi mạng xảy ra khi xóa học sinh. Vui lòng thử lại.', 'danger');
    }
}

// Hàm hiển thị Toast (GIỮ NGUYÊN)
function showToast(message, type = 'info', autoHide = true, delay = 3000) { 
    const toastContainer = document.createElement('div');
    toastContainer.className = 'position-fixed top-0 end-0 p-3'; toastContainer.style.zIndex = 1090;
    const toastId = 'toast-' + new Date().getTime();
    const toastHTML = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="${delay}">
            <div class="toast-header bg-${type} ${type === 'info' || type === 'success' || type === 'warning' || type === 'danger' ? 'text-white' : ''}">
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close ${type === 'info' || type === 'success' || type === 'warning' || type === 'danger' ? 'btn-close-white' : ''}" data-bs-dismiss="toast" aria-label="Close"></button>
            </div><div class="toast-body">${message}</div></div>`;
    toastContainer.innerHTML = toastHTML; document.body.appendChild(toastContainer);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: autoHide }); toast.show();
    if(autoHide) toastElement.addEventListener('hidden.bs.toast', function () { toastContainer.remove(); });
    else { // If not autoHide, ensure manual close button works and removes container
        const closeButton = toastElement.querySelector('.btn-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                toastContainer.remove();
            });
        }
    }
    return toast;
}

// --- KHỞI TẠO KHI TRANG TẢI XONG ---
document.addEventListener('DOMContentLoaded', function () {
    // Mặc định bật camera nếu có nút điều khiển
    if (document.getElementById('toggleCameraButton')) {
        startWebcam();
    }

    updateServerTrackingInfo(); // Initial load via HTTP
    updateHistory(); // Initial load
    loadStudentsList(); // Initial load
    
    // HTTP Polling interval for fallback/redundancy, 
    // but WebSocket 'main_status_updated' will provide more real-time updates for main stats.
    setInterval(updateServerTrackingInfo, 5000); // Reduced frequency as WebSocket is primary
    setInterval(updateHistory, 10000); // History can also be less frequent if WebSocket updates it on event

    // Xử lý image preview cho form thêm học sinh
    const imageInputAdd = document.getElementById('inputStudentImages');
    const previewContainerAdd = document.getElementById('imagePreviewContainer');
    if (imageInputAdd && previewContainerAdd) { 
        imageInputAdd.addEventListener('change', function () {
            previewContainerAdd.innerHTML = ''; // Clear old previews
            if (this.files && this.files.length > 0) {
                const fileCount = this.files.length;
                const previewSize = fileCount > 3 ? 80 : 100; // Adjust size based on count

                const imagesFlexContainer = document.createElement('div');
                imagesFlexContainer.className = 'd-flex flex-wrap gap-2 mt-2 justify-content-start';

                Array.from(this.files).forEach((file, index) => {
                    if (!file.type.match('image.*')) return; // Only process image files
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        const imgContainer = document.createElement('div');
                        imgContainer.className = 'position-relative d-flex flex-column align-items-center';
                        imgContainer.style.marginBottom = '10px';

                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'img-thumbnail';
                        img.style.width = `${previewSize}px`; 
                        img.style.height = `${previewSize}px`; 
                        img.style.objectFit = 'cover';

                        if (index === 0) { // First image is avatar
                            const avatarBadge = document.createElement('span');
                            avatarBadge.className = 'position-absolute top-0 start-0 translate-middle badge rounded-pill bg-primary';
                            avatarBadge.innerHTML = '<i class="bi bi-star-fill"></i>';
                            avatarBadge.title = 'Ảnh đại diện';
                            imgContainer.appendChild(avatarBadge);

                            const avatarLabel = document.createElement('div');
                            avatarLabel.className = 'text-center mt-1 small text-primary fw-bold';
                            avatarLabel.textContent = 'Ảnh đại diện';
                            imgContainer.appendChild(img); 
                            imgContainer.appendChild(avatarLabel);
                        } else {
                            imgContainer.appendChild(img);
                        }
                        imagesFlexContainer.appendChild(imgContainer);
                    };
                    reader.readAsDataURL(file);
                });
                previewContainerAdd.appendChild(imagesFlexContainer);

                // Add info text about avatar
                const infoText = document.createElement('div');
                infoText.className = 'alert alert-info mt-2 py-2 small';
                infoText.innerHTML = '<i class="bi bi-info-circle-fill me-1"></i> Ảnh đầu tiên sẽ là ảnh đại diện.';
                previewContainerAdd.appendChild(infoText);
            }
        });
    }
    // Reset form và preview khi modal thêm học sinh đóng
    const addStudentModalElement = document.getElementById('addStudentModal');
    if (addStudentModalElement) { 
        addStudentModalElement.addEventListener('hidden.bs.modal', function () {
            if(studentForm) studentForm.reset();
            if(previewContainerAdd) previewContainerAdd.innerHTML = '';
        });
    }
});