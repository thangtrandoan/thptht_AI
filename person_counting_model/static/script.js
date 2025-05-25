// Biến toàn cục để lưu ID học sinh đang xem chi tiết
let currentViewingStudentId = null;

function updateTrackingInfo() {
    fetch('/tracking_info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('netCountDisplay').textContent = data.net_count;

            const listElement = document.getElementById('trackedObjectsList');
            listElement.innerHTML = '';

            if (data.tracked_objects && data.tracked_objects.length > 0) {
                data.tracked_objects.forEach(obj => {
                    const listItem = document.createElement('li');
                    listItem.textContent = `ID: ${obj.id} (Box: ${obj.bbox.join(', ')})`;
                    listElement.appendChild(listItem);
                });
            } else {
                const listItem = document.createElement('li');
                listItem.textContent = 'Chưa có đối tượng nào đang được theo dõi.';
                listElement.appendChild(listItem);
            }
        })
        .catch(error => console.error('Lỗi khi lấy thông tin tracking:', error));
}

setInterval(updateTrackingInfo, 1500);
document.addEventListener('DOMContentLoaded', updateTrackingInfo);


// đổi tab bằng nav-tabs
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-tabs a').forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();

            document.querySelectorAll('.nav-tabs li').forEach(li => li.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            this.parentElement.classList.add('active');
            const targetId = this.getAttribute('href');
            document.querySelector(targetId).classList.add('active');
        });
    });
});


// tab1 người lên xuống xe
async function fetchStudentTrackingData() {
    try {
        const response = await fetch('/student_management_info');
        const data = await response.json();

        // Cập nhật danh sách học sinh đang trên xe (Tab 1)
        const trackedList = document.getElementById("trackedObjectsList");
        trackedList.innerHTML = ""; // Clear danh sách cũ

        const studentsOnBus = Object.entries(data.students_status)
            .filter(([_, info]) => info.status === "in");

        if (studentsOnBus.length === 0) {
            trackedList.innerHTML = "<li>Chưa có học sinh nào trên xe.</li>";
        } else {
            studentsOnBus.forEach(([id, info]) => {
                const li = document.createElement("li");
                li.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${info.name} (ID: ${id})</span>
                        <span class="badge bg-success">Trên xe</span>
                    </div>
                `;
                trackedList.appendChild(li);
            });
        }

        // Cập nhật số lượng học sinh trong thẻ hiển thị
        document.getElementById("netCountDisplay").textContent = studentsOnBus.length;

    } catch (error) {
        console.error("Lỗi khi lấy dữ liệu học sinh:", error);
    }
}

// Gọi hàm này mỗi 2 giây để cập nhật dữ liệu mới
setInterval(fetchStudentTrackingData, 2000);


// tab2 lịch sử lên xuống xe
async function updateHistory() {
    try {
        const res = await fetch('/api/history.json');  // fetch đúng từ Flask route
        const history = await res.json();
        const historyList = document.getElementById("historyList");
        historyList.innerHTML = "";

        history
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .forEach(event => {
                const li = document.createElement("li");
                li.innerHTML = `
                    <div class="name">
                        <strong>${event.name}</strong><br>
                        <small>ID: ${event.id}</small>
                    </div>
                    <span class="status ${event.action === 'Lên xe' ? 'in' : 'out'}">${event.action}</span>
                    <span class="timestamp">${new Date(event.timestamp).toLocaleTimeString()}</span>
                `;
                historyList.appendChild(li);

            });
    } catch (err) {
        console.error("Lỗi khi tải lịch sử:", err);
    }
}

// Gọi updateHistory khi trang load xong
document.addEventListener('DOMContentLoaded', function () {
    updateHistory(); // Lấy dữ liệu lần đầu    // Thêm xử lý hiển thị preview ảnh
    const imageInput = document.getElementById('inputStudentImages');
    const previewContainer = document.getElementById('imagePreviewContainer');
    
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            previewContainer.innerHTML = ''; // Xóa preview cũ
            
            if (this.files && this.files.length > 0) {
                // Xác định kích thước ảnh preview dựa trên số lượng ảnh
                const fileCount = this.files.length;
                const previewSize = fileCount > 3 ? 80 : 100; // Nếu > 3 ảnh, giảm kích thước xuống
                
                // Thêm container cho tất cả ảnh
                const imagesContainer = document.createElement('div');
                imagesContainer.className = 'd-flex flex-wrap gap-2 mt-2 justify-content-start';
                
                // Thêm nhãn "Ảnh đại diện" cho ảnh đầu tiên
                const firstImage = document.createElement('div');
                firstImage.className = 'position-relative';
                firstImage.style.marginBottom = '10px';
                
                Array.from(this.files).forEach((file, index) => {
                    if (!file.type.match('image.*')) return;
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const imgContainer = document.createElement('div');
                        imgContainer.className = 'position-relative';
                        imgContainer.style.marginBottom = '10px';
                        
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'img-thumbnail';
                        img.style.width = `${previewSize}px`;
                        img.style.height = `${previewSize}px`;
                        img.style.objectFit = 'cover';
                        
                        // Thêm badge nếu là ảnh đầu tiên (avatar)
                        if (index === 0) {
                            const avatarBadge = document.createElement('span');
                            avatarBadge.className = 'position-absolute top-0 start-0 translate-middle badge rounded-pill bg-primary';
                            avatarBadge.innerHTML = '<i class="bi bi-star-fill"></i>';
                            avatarBadge.title = 'Ảnh đại diện';
                            imgContainer.appendChild(avatarBadge);
                            
                            // Thêm label "Ảnh đại diện" dưới ảnh đầu tiên
                            const avatarLabel = document.createElement('div');
                            avatarLabel.className = 'text-center mt-1 small text-primary fw-bold';
                            avatarLabel.textContent = 'Ảnh đại diện';
                            imgContainer.appendChild(avatarLabel);
                        }
                        
                        imgContainer.appendChild(img);
                        imagesContainer.appendChild(imgContainer);
                    };
                    
                    reader.readAsDataURL(file);
                });
                
                previewContainer.appendChild(imagesContainer);
                
                // Thêm thông báo cho người dùng biết ảnh đầu tiên sẽ được sử dụng làm avatar
                const infoText = document.createElement('div');
                infoText.className = 'alert alert-info mt-2 py-2 small';
                infoText.innerHTML = '<i class="bi bi-info-circle-fill me-1"></i> Ảnh đầu tiên sẽ được sử dụng làm ảnh đại diện. Tải lên nhiều ảnh sẽ cải thiện độ chính xác nhận diện khuôn mặt.';
                previewContainer.appendChild(infoText);
            }
        });
    }
    
    // Xử lý form thêm học sinh
    const studentForm = document.getElementById('addStudentForm');
    if (studentForm) {
        studentForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            
            // Lấy thông tin học sinh từ form
            formData.append('student_id', document.getElementById('inputStudentId').value);
            formData.append('name', document.getElementById('inputStudentName').value);
            formData.append('class', document.getElementById('inputStudentClass').value);
            formData.append('age', document.getElementById('inputStudentAge').value);
            formData.append('address', document.getElementById('inputStudentAddress').value);
            
            // Thông tin phụ huynh
            formData.append('father_name', document.getElementById('inputFatherName').value);
            formData.append('father_age', document.getElementById('inputFatherAge').value);
            formData.append('father_phone', document.getElementById('inputFatherPhone').value);
            formData.append('mother_name', document.getElementById('inputMotherName').value);
            formData.append('mother_age', document.getElementById('inputMotherAge').value);
            formData.append('mother_phone', document.getElementById('inputMotherPhone').value);
              // Thêm ảnh vào formData
            const imageFiles = document.getElementById('inputStudentImages').files;
            const hasImages = imageFiles && imageFiles.length > 0;

            if (!hasImages) {
                alert('Vui lòng chọn ít nhất một ảnh cho học sinh.');
                return;
            }

            // Hiển thị thông báo đang tải
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang xử lý...';
            submitBtn.disabled = true;
            
            // Thêm tất cả ảnh vào formData
            for (let i = 0; i < imageFiles.length; i++) {
                formData.append('images', imageFiles[i]);
            }
            
            try {
                const response = await fetch('/api/add_student', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Hiện thông báo thành công với số lượng ảnh đã tải lên
                    alert(`Thêm học sinh thành công với ${imageFiles.length} ảnh!`);
                    
                    // Đóng modal
                    const modal = bootstrap.Modal.getInstance(document.getElementById('addStudentModal'));
                    modal.hide();
                    
                    // Cập nhật lại danh sách học sinh
                    loadStudentsList();
                    
                    // Reset form
                    studentForm.reset();
                    previewContainer.innerHTML = '';
                } else {
                    alert(`Lỗi: ${result.message || 'Không thể thêm học sinh'}`);
                }
            } catch (error) {
                console.error('Lỗi khi gửi form:', error);
                alert('Có lỗi xảy ra khi thêm học sinh. Vui lòng thử lại.');
            } finally {
                // Khôi phục nút submit
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
});

// Gọi lại updateHistory mỗi khi chuyển sang tab "Lịch sử"
document.querySelector('a[href="#history"]').addEventListener('click', function () {
    updateHistory();
});

// Cập nhật định kỳ mỗi 1 giây nếu muốn
setInterval(updateHistory, 1000);

// Tải danh sách học sinh và hiển thị
async function loadStudentsList() {
    try {
        const response = await fetch('/api/students');
        const students = await response.json();
        
        // Đổ dữ liệu vào tab 3 (danh sách học sinh)
        const studentsList = document.querySelector('#list .hoc-sinh');
          // Xóa danh sách cũ nếu có, giữ lại tiêu đề và nút thêm
        const title = studentsList.querySelector('h1');
        const addButton = studentsList.querySelector('button');
        studentsList.innerHTML = '';
        studentsList.appendChild(title);
        studentsList.appendChild(addButton);
        
        // Tạo container có thanh cuộn
        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-container';
          // Tạo bảng hiển thị danh sách học sinh
        const table = document.createElement('table');
        table.className = 'table table-striped';
        table.innerHTML = `
            <thead>
                <tr>
                    <th width="15%">Mã HS</th>
                    <th width="35%">Họ tên</th>
                    <th width="20%">Trạng thái</th>
                    <th width="30%" style="text-align: center; padding-right: 10px;">Thao tác</th>
                </tr>
            </thead>
            <tbody id="studentsTableBody"></tbody>
        `;
        
        tableContainer.appendChild(table);
        studentsList.appendChild(tableContainer);
        
        const tbody = document.getElementById('studentsTableBody');
        
        students.forEach(student => {
            const tr = document.createElement('tr');

            // Tạo ô ID
            const tdId = document.createElement('td');
            tdId.textContent = student.id;
            
            // Tạo ô tên
            const tdName = document.createElement('td');
            tdName.textContent = student.name;
            
            // Tạo ô trạng thái
            const tdStatus = document.createElement('td');
            const statusBadge = document.createElement('span');
            statusBadge.className = `badge ${student.status === 'in' ? 'bg-success' : 'bg-secondary'}`;
            statusBadge.textContent = student.status === 'in' ? 'Trên xe' : 'Không trên xe';
            tdStatus.appendChild(statusBadge);
            
            // Tạo ô thao tác
            const tdActions = document.createElement('td');
            tdActions.style.textAlign = 'right';     // Căn phải nội dung
            tdActions.style.padding = '5px';         // Thêm padding

            // Tạo div container để chứa các nút (để nằm ngang trên cùng một dòng)
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'action-buttons';
            buttonContainer.style.display = 'flex';
            buttonContainer.style.justifyContent = 'center'; // Căn phải
            buttonContainer.style.gap = '5px';                 // Khoảng cách giữa các nút
            
            // Nút xem
            const viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-sm btn-info';
            viewBtn.innerHTML = '<i class="bi bi-eye"></i> Xem';
            viewBtn.addEventListener('click', () => viewStudentDetails(student.id));
            buttonContainer.appendChild(viewBtn);
            
            // Nút xóa học sinh (nằm ngay bên cạnh nút xem)
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-sm btn-danger';
            deleteBtn.innerHTML = '<i class="bi bi-trash"></i>';
            deleteBtn.addEventListener('click', () => deleteStudent(student.id, student.name));
            buttonContainer.appendChild(deleteBtn);
            
            // Thêm container vào ô thao tác
            tdActions.appendChild(buttonContainer);
              // Thêm các ô vào hàng
            tr.appendChild(tdId);
            tr.appendChild(tdName);
            tr.appendChild(tdStatus);
            tr.appendChild(tdActions);
            
            // Thêm hàng vào bảng
            tbody.appendChild(tr);
        });
        
    } catch (error) {
        console.error('Lỗi khi tải danh sách học sinh:', error);
    }
}

// Xem chi tiết học sinh
async function viewStudentDetails(studentId) {
    try {
        const response = await fetch(`/api/students/${studentId}`);
        const student = await response.json();
        
        // Lưu ID học sinh hiện tại để sử dụng cho chức năng sửa
        currentViewingStudentId = studentId;
        
        // Đảm bảo reset tất cả trạng thái liên quan đến avatar
        selectedAvatarFile = null;
        document.getElementById('avatarFileInput').value = '';
        
        // Cập nhật thông tin trong modal
        document.getElementById('studentName').textContent = student.name || 'N/A';
        document.getElementById('studentClass').textContent = student.original_data?.class || 'N/A';
        document.getElementById('studentAge').textContent = student.original_data?.age || 'N/A';
        document.getElementById('studentAddress').textContent = student.original_data?.address || 'N/A';
        
        // Thông tin phụ huynh
        document.getElementById('fatherName').textContent = student.original_data?.father_name || 'N/A';
        document.getElementById('fatherAge').textContent = student.original_data?.father_age || 'N/A';
        document.getElementById('fatherPhone').textContent = student.original_data?.father_phone || 'N/A';
        
        document.getElementById('motherName').textContent = student.original_data?.mother_name || 'N/A';
        document.getElementById('motherAge').textContent = student.original_data?.mother_age || 'N/A';
        document.getElementById('motherPhone').textContent = student.original_data?.mother_phone || 'N/A';
        
        // Cập nhật ảnh với timestamp để tránh cache
        const timestamp = new Date().getTime();
        const studentImage = document.getElementById('studentImage');
        studentImage.src = `/api/student_image/${studentId}?t=${timestamp}`;
        
        // Reset visual cues
        studentImage.classList.remove('border', 'border-success', 'border-3');
        
        // Reset nút thay đổi avatar
        const avatarBtn = document.getElementById('changeAvatarBtn');
        avatarBtn.textContent = "Thay ảnh đại diện";
        avatarBtn.classList.remove('btn-success');
        avatarBtn.classList.add('btn-outline-primary');
        
        // Xóa thông báo tự động lưu nếu có
        const existingNotice = document.querySelector('#changeAvatarBtn + .text-success');
        if (existingNotice) {
            existingNotice.remove();
        }
        
        // Hiển thị modal
        const studentModal = new bootstrap.Modal(document.getElementById('studentModal'));
        studentModal.show();
        
    } catch (error) {
        console.error('Lỗi khi tải thông tin chi tiết học sinh:', error);
        alert('Không thể tải thông tin học sinh. Vui lòng thử lại sau.');
    }
}

// Tải danh sách học sinh khi trang tải xong
document.addEventListener('DOMContentLoaded', function() {
    loadStudentsList();
    
    // Sự kiện khi nhấn nút sửa thông tin
    document.getElementById('editStudentInfoBtn').addEventListener('click', function() {
        // Lấy thông tin từ modal xem chi tiết
        const studentId = currentViewingStudentId;
        
        // Điền thông tin vào form sửa
        document.getElementById('editStudentId').value = studentId;
        document.getElementById('editStudentName').value = document.getElementById('studentName').textContent;
        document.getElementById('editStudentClass').value = document.getElementById('studentClass').textContent;
        document.getElementById('editStudentAge').value = document.getElementById('studentAge').textContent;
        document.getElementById('editStudentAddress').value = document.getElementById('studentAddress').textContent;
        
        // Thông tin phụ huynh
        document.getElementById('editFatherName').value = document.getElementById('fatherName').textContent;
        document.getElementById('editFatherAge').value = document.getElementById('fatherAge').textContent;
        document.getElementById('editFatherPhone').value = document.getElementById('fatherPhone').textContent;
        
        document.getElementById('editMotherName').value = document.getElementById('motherName').textContent;
        document.getElementById('editMotherAge').value = document.getElementById('motherAge').textContent;
        document.getElementById('editMotherPhone').value = document.getElementById('motherPhone').textContent;
        
        // Đóng modal xem chi tiết
        const studentModal = bootstrap.Modal.getInstance(document.getElementById('studentModal'));
        studentModal.hide();
        
        // Mở modal sửa thông tin
        const editModal = new bootstrap.Modal(document.getElementById('editStudentModal'));
        editModal.show();
    });
    
    // Xử lý form sửa thông tin học sinh
    document.getElementById('editStudentForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const studentId = document.getElementById('editStudentId').value;
        
        // Tạo đối tượng dữ liệu để gửi đi
        const studentData = {
            name: document.getElementById('editStudentName').value,
            class: document.getElementById('editStudentClass').value,
            age: document.getElementById('editStudentAge').value,
            address: document.getElementById('editStudentAddress').value,
            father_name: document.getElementById('editFatherName').value,
            father_age: document.getElementById('editFatherAge').value,
            father_phone: document.getElementById('editFatherPhone').value,
            mother_name: document.getElementById('editMotherName').value,
            mother_age: document.getElementById('editMotherAge').value,
            mother_phone: document.getElementById('editMotherPhone').value
        };
        
        try {
            const response = await fetch(`/api/students/${studentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(studentData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                alert('Cập nhật thông tin học sinh thành công!');
                
                // Đóng modal
                const editModal = bootstrap.Modal.getInstance(document.getElementById('editStudentModal'));
                editModal.hide();
                
                // Cập nhật lại danh sách học sinh
                loadStudentsList();
                
                // Hiển thị lại thông tin chi tiết với dữ liệu mới
                viewStudentDetails(studentId);
            } else {
                alert(`Lỗi: ${result.message || 'Không thể cập nhật thông tin học sinh'}`);
            }
        } catch (error) {
            console.error('Lỗi khi gửi form cập nhật:', error);
            alert('Có lỗi xảy ra khi cập nhật thông tin học sinh. Vui lòng thử lại.');
        }
    });
      // Biến để lưu file ảnh đại diện đã chọn
    let selectedAvatarFile = null;
    
    // Xử lý sự kiện thay đổi ảnh đại diện
    document.getElementById('changeAvatarBtn').addEventListener('click', function() {
        // Mở hộp thoại chọn file
        document.getElementById('avatarFileInput').click();
    });
      // Xử lý khi chọn file ảnh mới - chỉ hiển thị preview
    document.getElementById('avatarFileInput').addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            selectedAvatarFile = file; // Lưu file để sử dụng khi đóng modal
            
            // Hiển thị preview ngay lập tức
            const reader = new FileReader();
            reader.onload = function(e) {
                const studentImage = document.getElementById('studentImage');
                studentImage.src = e.target.result;
                
                // Thêm hiệu ứng visual để người dùng biết đã chọn ảnh mới
                studentImage.classList.add('border', 'border-success', 'border-3');
                
                // Đổi text của nút "Thay ảnh đại diện" thành "Đã chọn ảnh mới"
                const avatarBtn = document.getElementById('changeAvatarBtn');
                avatarBtn.textContent = "Đã chọn ảnh mới ✓";
                avatarBtn.classList.remove('btn-outline-primary');
                avatarBtn.classList.add('btn-success');
                
                // Thêm thông báo tự động lưu
                const saveNotice = document.createElement('div');
                saveNotice.className = 'text-success small mt-1';
                saveNotice.innerHTML = 'Ảnh sẽ tự động lưu khi đóng';
                
                // Kiểm tra xem đã có thông báo chưa
                const existingNotice = document.querySelector('#changeAvatarBtn + .text-success');
                if (!existingNotice) {
                    // Thêm thông báo sau nút
                    avatarBtn.parentNode.insertBefore(saveNotice, avatarBtn.nextSibling);
                }
            };
            reader.readAsDataURL(file);
        }
    });    // Sự kiện khi đóng modal chi tiết học sinh - sẽ cập nhật ảnh nếu đã chọn
    document.getElementById('studentModal').addEventListener('hidden.bs.modal', async function() {
        // Kiểm tra xem có ảnh mới được chọn không
        if (selectedAvatarFile && currentViewingStudentId) {
            const studentId = currentViewingStudentId;
            const formData = new FormData();
            formData.append('avatar', selectedAvatarFile);
            
            try {
                // Hiển thị thông báo đang cập nhật
                const notification = document.createElement('div');
                notification.className = 'position-fixed top-0 end-0 p-3';
                notification.innerHTML = `
                    <div class="toast show" role="alert">
                        <div class="toast-header">
                            <strong class="me-auto">Thông báo</strong>
                            <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                        </div>
                        <div class="toast-body">
                            Đang cập nhật ảnh đại diện...
                        </div>
                    </div>
                `;
                document.body.appendChild(notification);
                
                const response = await fetch(`/api/students/${studentId}/avatar`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Xóa thông báo đang cập nhật
                document.body.removeChild(notification);
                
                if (response.ok) {
                    // Thông báo thành công
                    const successNotification = document.createElement('div');
                    successNotification.className = 'position-fixed top-0 end-0 p-3';
                    successNotification.innerHTML = `
                        <div class="toast show" role="alert">
                            <div class="toast-header bg-success text-white">
                                <strong class="me-auto">Thành công</strong>
                                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                            </div>
                            <div class="toast-body">
                                Cập nhật ảnh đại diện thành công!
                            </div>
                        </div>
                    `;
                    document.body.appendChild(successNotification);
                    
                    // Tự động xóa thông báo sau 3 giây
                    setTimeout(() => {
                        document.body.removeChild(successNotification);
                    }, 3000);
                    
                    // Reset biến lưu file đã chọn
                    selectedAvatarFile = null;
                    
                    // Cập nhật lại danh sách học sinh để hiển thị ảnh mới
                    setTimeout(() => {
                        // Cập nhật sau 500ms để đảm bảo server đã xử lý xong
                        loadStudentsList();
                    }, 500);
                } else {
                    alert(`Lỗi: ${result.message || 'Không thể cập nhật ảnh đại diện'}`);
                }
            } catch (error) {
                console.error('Lỗi khi gửi ảnh đại diện mới:', error);
                alert('Có lỗi xảy ra khi cập nhật ảnh đại diện. Vui lòng thử lại.');
            }
            
            // Reset biến lưu file ảnh
            selectedAvatarFile = null;
            // Reset input file để có thể chọn lại cùng một file nếu cần
            document.getElementById('avatarFileInput').value = '';
        }
    });
});

// Tải lại danh sách khi chuyển đến tab danh sách
document.querySelector('a[href="#list"]').addEventListener('click', loadStudentsList);

// Hàm xóa học sinh
async function deleteStudent(studentId, studentName) {
    // Hiển thị hộp thoại xác nhận
    if (!confirm(`Bạn có chắc chắn muốn xóa học sinh ${studentName} (ID: ${studentId}) không? Thao tác này sẽ xóa tất cả thông tin và ảnh của học sinh này.`)) {
        return; // Người dùng hủy xóa
    }
    
    try {
        // Hiển thị thông báo đang xóa
        const notification = document.createElement('div');
        notification.className = 'position-fixed top-0 end-0 p-3';
        notification.innerHTML = `
            <div class="toast show" role="alert">
                <div class="toast-header">
                    <strong class="me-auto">Thông báo</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    Đang xóa học sinh...
                </div>
            </div>
        `;
        document.body.appendChild(notification);
        
        const response = await fetch(`/api/students/${studentId}`, {
            method: 'DELETE'
        });
        
        // Xóa thông báo đang xóa
        document.body.removeChild(notification);
        
        if (response.ok) {
            // Thông báo thành công
            const successNotification = document.createElement('div');
            successNotification.className = 'position-fixed top-0 end-0 p-3';
            successNotification.innerHTML = `
                <div class="toast show" role="alert">
                    <div class="toast-header bg-success text-white">
                        <strong class="me-auto">Thành công</strong>
                        <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                    </div>
                    <div class="toast-body">
                        Đã xóa học sinh ${studentName} thành công!
                    </div>
                </div>
            `;
            document.body.appendChild(successNotification);
            
            // Tự động xóa thông báo sau 3 giây
            setTimeout(() => {
                document.body.removeChild(successNotification);
            }, 3000);
            
            // Cập nhật lại danh sách học sinh
            loadStudentsList();
        } else {
            const result = await response.json();
            alert(`Lỗi: ${result.message || 'Không thể xóa học sinh'}`);
        }
    } catch (error) {
        console.error('Lỗi khi xóa học sinh:', error);
        alert('Có lỗi xảy ra khi xóa học sinh. Vui lòng thử lại.');
    }
}
