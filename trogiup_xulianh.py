import operator
import cv2
import numpy as np


def tim_4_goc(da_giac, limit_fn, compare_fn):
    # limit_fn là hàm tối thiểu hoặc tối đa
    # compare_fn là hàm np.add hoặc np.subtract
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in da_giac]),
                          key=operator.itemgetter(1))

    return da_giac[section][0][0], da_giac[section][0][1]

def ve_4_goc(diem, nguyengoc):
    cv2.circle(nguyengoc, diem, 7, (0, 255, 0), cv2.FILLED)


def tro_giup_lam_moi(anh):
    if np.isclose(anh, 0).sum() / (anh.shape[0] * anh.shape[1]) >= 0.95:
        return np.zeros_like(anh), False

    height, width = anh.shape
    mid = width // 2
    if np.isclose(anh[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(anh), False

    # trung tâm ảnh
    vien, _ = cv2.findContours(anh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vien = sorted(vien, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(vien[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    anh_moi = np.zeros_like(anh)
    anh_moi[start_y:start_y + h, start_x:start_x + w] = anh[y:y + h, x:x + w]

    return anh_moi, True


def tro_giup_duong_luoi(anh, shape_location, length=10):
    anh_phu = anh.copy()
    # nếu hàng ngang thì shape_location 1, còn hàng dọc là 0
    row_or_col = anh_phu.shape[shape_location]
    # tìm khoảng cách các dòng vừa đặt
    size = row_or_col // length

    # tìm vị trí trung tâm
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # làm mờ và mở rộng các dòng
    anh_phu = cv2.erode(anh_phu, kernel)
    anh_phu = cv2.dilate(anh_phu, kernel)

    return anh_phu

def ke_dong(anh, dong):
    anh_phu = anh.copy()
    dong = np.squeeze(dong)

    for rho, theta in dong:
        # tìm và vẽ các đường thẳng kéo dài
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(anh_phu, (x1, y1), (x2, y2), (255, 255, 255), 4)
    return anh_phu

