document.getElementById('imageInput').addEventListener('change', function(event) {
    const files = event.target.files;
    const imageContainer = document.getElementById('imageContainer');
    
    imageContainer.innerHTML = ''; // Xóa nội dung cũ nếu có

    if (files) {
        Array.from(files).forEach(file => {
            const reader = new FileReader();
            
            // Khi FileReader đọc xong file ảnh
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                imageContainer.appendChild(img); // Thêm ảnh vào container
            }
            
            // Đọc file ảnh như URL
            reader.readAsDataURL(file);
        });
    }
});
