import cv2


def tien_xulianh(anh):
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)

    # làm mờ
    blur = cv2.GaussianBlur(anh_xam, (5,5), 0)

    # đạt ngưỡng
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # đảo ngược để các đường lưới và văn bản có màu trắng
    inverted = cv2.bitwise_not(thresh, 0)

    # lấy trung tâm
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # biến đổi làm hình mượt hơn
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # giãn ra để tăng kích thước đường viền
    result = cv2.dilate(morph, kernel, iterations=1)
    return result
