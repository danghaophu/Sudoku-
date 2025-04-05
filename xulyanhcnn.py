import cv2
import numpy as np
import tensorflow as tf
import time
import trogiup_xulianh


def tao_mat_luoi(hang_doc, hang_ngang):
    # kết hợp các đường dọc và ngang để tạo lưới
    grid = cv2.add(hang_ngang, hang_doc)
    # đạt ngưỡng và mở rộng lưới để bao phủ nhiều khu vực hơn
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # tìm danh sách vị trí của các dòng, đây là một mảng (rho, theta tính bằng radian)
    diem = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    dong = trogiup_xulianh.ke_dong(grid, diem)
    # trích xuất các dòng để chỉ còn lại các số
    mask = cv2.bitwise_not(dong)
    return mask


def duong_ke_luoi(anh, length=10):
    hang_ngang = trogiup_xulianh.tro_giup_duong_luoi(anh, 1, length)
    hang_doc = trogiup_xulianh.tro_giup_duong_luoi(anh, 0, length)
    return hang_doc, hang_ngang


def tim_vien(anh, nguyengoc):
    # tìm đường viền trên hình ảnh đạt ngưỡng
    vien, _ = cv2.findContours(anh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sắp xếp theo lớn nhất
    vien = sorted(vien, key=cv2.contourArea, reverse=True)
    da_giac = None

    # đây là hình sudoku cần định dạng
    for cnt in vien:
        dientich = cv2.contourArea(cnt)
        chuvi = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * chuvi, closed=True)
        so_goc = len(approx)

        if so_goc == 4 and dientich > 1000:
            da_giac = cnt
            break

    if da_giac is not None:
        # tìm 4 góc của sudoku
        goc_trentrai = trogiup_xulianh.tim_4_goc(da_giac, min, np.add)  # có giá trị (x + y) nhỏ nhất
        goc_trenphai = trogiup_xulianh.tim_4_goc(da_giac, max, np.subtract)  # có giá trị (x - y) lớn nhất
        goc_duoitrai = trogiup_xulianh.tim_4_goc(da_giac, min, np.subtract)  # có giá trị (x - y) nhỏ nhất
        goc_duoiphai = trogiup_xulianh.tim_4_goc(da_giac, max, np.add)  # có giá trị (x + y) lớn nhất

        # xét xem có phải là một hình vuông không
        if goc_duoiphai[1] - goc_trenphai[1] == 0:
            return []
        if not (0.95 < ((goc_trenphai[0] - goc_trentrai[0]) / (goc_duoiphai[1] - goc_trenphai[1])) < 1.05):
            return []

        cv2.drawContours(nguyengoc, [da_giac], 0, (0, 0, 255), 3)

        # vẽ các vòng tròn tương ứng 4 góc
        [trogiup_xulianh.ve_4_goc(x, nguyengoc) for x in [goc_trentrai, goc_trenphai, goc_duoiphai, goc_duoitrai]]

        return [goc_trentrai, goc_trenphai, goc_duoiphai, goc_duoitrai]

    return []


def be_cong_anh(goc, nguyengoc):
    # bẻ cong các điểm
    goc = np.array(goc, dtype='float32')
    goc_trentrai, goc_trenphai, goc_duoiphai, goc_duoitrai = goc

    # tìm chiều rộng cạnh tốt nhất, vì chúng ta sẽ uốn thành hình vuông, chiều cao = chiều dài
    width = int(max([
        np.linalg.norm(goc_trenphai - goc_duoiphai),
        np.linalg.norm(goc_trentrai - goc_duoitrai),
        np.linalg.norm(goc_duoiphai - goc_duoitrai),
        np.linalg.norm(goc_trentrai - goc_trenphai)
    ]))

    # tạo một mảng với hiển thị 4 góc
    anh_xa = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    ma_tran = cv2.getPerspectiveTransform(goc, anh_xa)

    return cv2.warpPerspective(nguyengoc, ma_tran, (width, width)), ma_tran


def chia_o_vuong(warped_img):
    o_vuong = []

    width = warped_img.shape[0] // 9

    # tìm mỗi hình vuông khi chúng có cùng cạnh
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Góc trên cùng bên trái của hộp giới hạn
            p2 = ((i + 1) * width, (j + 1) * width)  # Góc dưới cùng bên phải của hộp giới hạn
            o_vuong.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return o_vuong


def don_o_vuong(o_vuong):
    don_o_vuong = []
    i = 0

    for square in o_vuong:
        anh_moi, is_number = trogiup_xulianh.tro_giup_lam_moi(square)

        if is_number:
            don_o_vuong.append(anh_moi)
            i += 1

        else:
            don_o_vuong.append(0)

    return don_o_vuong


def find_and_process_contours(binary_images):
    processed_images = []
    
    for img in binary_images:
        # Tìm các contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Vẽ hình chữ nhật bao quanh contours
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Cắt ảnh
            cropped_img = img[y:y + h, x:x + w]
            # Resize ảnh về 32x32
            resized_img = cv2.resize(cropped_img, (28, 28))
            # Đưa ảnh vào mảng processed_images
            processed_images.append(resized_img)
    
    return processed_images



def nhandien_chuso(o_vuong_daxuli1, model):
    o_vuong_daxuli = find_and_process_contours(o_vuong_daxuli1)
    start_time = time.time()
    s = ""
    o_vuong_dadinhdang = []
    location_of_zeroes = set()

    # ảnh trắng
    anh_trang = np.zeros_like(cv2.resize(o_vuong_daxuli[0], (28, 28)))

    for i in range(len(o_vuong_daxuli)):
        if type(o_vuong_daxuli[i]) == int:
            location_of_zeroes.add(i)
            o_vuong_dadinhdang.append(anh_trang)
        else:
            anh = cv2.resize(o_vuong_daxuli[i], (28, 28))
            o_vuong_dadinhdang.append(anh)

    o_vuong_dadinhdang = np.array(o_vuong_dadinhdang)
    all_preds = list(map(np.argmax, model(tf.convert_to_tensor(o_vuong_dadinhdang))))
    for i in range(len(all_preds)):
        if i in location_of_zeroes:
            s += "0"
        else:
            s += str(all_preds[i] + 1)
    end_time = time.time()  # Kết thúc đếm thời gian
    elapsed_time = end_time - start_time  # Tính thời gian đã trôi qua

    return s, elapsed_time


def dien_so_len_anh(warped_img, solved_puzzle, o_vuong_daxuli):
    width = warped_img.shape[0] // 9

    img_w_text = np.zeros_like(warped_img)

    # tìm mỗi hình vuông khi chúng có cùng cạnh
    index = 0
    for j in range(9):
        for i in range(9):
            if type(o_vuong_daxuli[index]) == int:
                p1 = (i * width, j * width)  # Góc trên cùng bên trái của hộp giới hạn
                p2 = ((i + 1) * width, (j + 1) * width)  # Góc dưới cùng bên phải của hộp giới hạn

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text


def anh_khong_be_cong(img_src, img_dest, diem, time):
    diem = np.array(diem)

    height, width = img_src.shape[0], img_src.shape[1]
    diem_goc = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(diem_goc, diem)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, diem, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img
