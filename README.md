# Sudoku – CNN, MobileNet, KNN

## Mục tiêu đề tài
Đề tài **“Nghiên cứu nhận dạng chữ số viết tay và ứng dụng giải Sudoku”** tập trung vào ba phương pháp phổ biến:  
- **CNN (Convolutional Neural Network)**  
- **MobileNet**  
- **KNN (K-Nearest Neighbors)**  

Nhóm phát triển một hệ thống giải Sudoku chữ số viết tay đơn giản và tiến hành so sánh hiệu quả giữa các mô hình.

---

## CNN (Convolutional Neural Network)
- Sử dụng tập dữ liệu **MNIST** (32x32 pixel, nhị phân).  
- Gồm **60.000 mẫu huấn luyện** và **10.000 mẫu kiểm thử**.  

<img width="566" height="465" alt="image" src="https://github.com/user-attachments/assets/fa350753-edaa-4e5f-a9fb-9e3f4b5ac1cf" />  
<br>
*Hình 1. Minh họa dữ liệu trong không gian 3D*  

<img width="950" height="968" alt="image" src="https://github.com/user-attachments/assets/0110cd2c-5e6c-41f0-aa76-93f06065b54a" />
<br>
*Hình 2. Cấu trúc mô hình CNN*

---

## MobileNet
Nhóm sử dụng **MobileNet V1** với 30 lớp:  
- Lớp 1: Convolution (stride = 2)  
- Lớp 2: Depthwise convolution  
- Lớp 3: Pointwise convolution  
- Lớp 4: Depthwise convolution (stride = 2)  
- Lớp 30: Softmax  

<img width="950" height="686" alt="image" src="https://github.com/user-attachments/assets/707a9698-fd8f-49b0-ae7d-47d896cb072d" />  
<br>
*Hình 3. Depthwise Separable Convolution*  

---

## KNN (K-Nearest Neighbors)
- Là thuật toán **lazy learning**, không xây dựng hàm ánh xạ tường minh.  
- Khi dự đoán, KNN tính toán khoảng cách từ điểm mới tới toàn bộ dữ liệu huấn luyện và chọn *k láng giềng gần nhất*.  

<img width="857" height="276" alt="image" src="https://github.com/user-attachments/assets/e06baa35-436d-49f1-97c5-d7fcbaf2849a" />  
<br>
*Hình 4. Hiệu suất KNN theo giá trị k (1–10)*  

---

## Sudoku
Trong đề tài, chỉ áp dụng cho Sudoku dạng chuẩn **9x9**.  

<img width="594" height="297" alt="Screenshot 2024-03-19 165339" src="https://github.com/user-attachments/assets/55362a7d-ab08-4279-b49a-802722cfe13d" />  
<br>
*Hình 5. Minh họa Sudoku*  

---

## Kết luận
- **CNN**: Độ chính xác cao (97–100%), ổn định, nhưng tốc độ chậm hơn (~0.1s/ảnh). Phù hợp với bài toán không yêu cầu thời gian thực.  
- **MobileNet**: Hiệu suất thấp hơn CNN một chút (82–100%) nhưng tốc độ nhanh hơn (~0.077s/ảnh). Thích hợp cho các ứng dụng yêu cầu xử lý nhanh.  
- **KNN**: Độ chính xác thấp nhất với dữ liệu mới, nhưng với dữ liệu huấn luyện thì cao (95–100%). Tốc độ dự đoán nhanh, phù hợp cho nhận dạng mẫu và tìm kiếm.  

---
