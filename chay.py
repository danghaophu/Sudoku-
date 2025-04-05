
import sys
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import cv2
import time as t
import tien_xulianh
import xulianh
import sudoku
import os
import tensorflow
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    ImageUpdate2 = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)
        my_model = tensorflow.keras.models.load_model('mobilenet_mnist.h5')
        frame_rate = 30
        prev = 0

        seen = dict()
        while self.ThreadActive:
            success, anh = cap.read()
            time_elapsed = t.time() - prev
            index = 0
            if time_elapsed > 1. / frame_rate:
                prev = t.time()

                anh_ketqua = anh.copy()
                anh_chuagoc = anh.copy()
                anh_daxuli = tien_xulianh.tien_xulianh(anh)
                goc = xulianh.tim_vien(anh_daxuli, anh_chuagoc)
                if goc:
                    warped, ma_tran = xulianh.be_cong_anh(goc, anh)
                    warped_processed = tien_xulianh.tien_xulianh(warped)
                    # self.cop(warped_processed)

                    duongke_doc, duongke_ngang = xulianh.duong_ke_luoi(warped_processed)
                    try:
                        mask = xulianh.tao_mat_luoi(duongke_doc, duongke_ngang)

                        so = cv2.bitwise_and(warped_processed, mask)
                        self.cop(so)
                        new_size = (252, 252)
                        so_resized = cv2.resize(so, new_size)
                        o_vuong = xulianh.chia_o_vuong(so_resized)
                        o_vuong_daxuli = xulianh.don_o_vuong(o_vuong)
                        o_vuong_phong_doan = xulianh.nhandien_chuso(o_vuong_daxuli,my_model)
                        print(o_vuong_phong_doan)
                        # neu sudoku sai thi van tiep tuc          
                        if o_vuong_phong_doan in seen and seen[o_vuong_phong_doan] is False:
                            # print("sai")
                            
                            continue
                        

                                # neu sudoku dung thi hien dap an
                        if o_vuong_phong_doan in seen:
                            print("dung1")
                            xulianh.dien_so_len_anh(warped, seen[o_vuong_phong_doan][0], o_vuong_daxuli)
                            anh_ketqua = xulianh.anh_khong_be_cong(warped, anh_ketqua, goc, seen[o_vuong_phong_doan][1])                              
                            #luu ket qua
                            index += 1

                            cv2.imwrite(f"anhketqua_{index}.jpg", anh_ketqua)
                            
                        # try:
                        else:
                            if len(o_vuong_phong_doan)<=81:
                                solved_puzzle, time = sudoku.dieu_kien_giai(o_vuong_phong_doan)
                                print(solved_puzzle)
                                if solved_puzzle is not None:
                                    # print("dung1")
                                    
                                    xulianh.dien_so_len_anh(warped, solved_puzzle, o_vuong_daxuli)
                                    anh_ketqua = xulianh.anh_khong_be_cong(warped, anh_ketqua, goc, time)
                                    seen[o_vuong_phong_doan] = [solved_puzzle, 0]
                                #luu anh ketqua
                                    cv2.imwrite("anh_ket_qua.jpg", anh_ketqua)

                                    index += 1

                                    cv2.imwrite(f"anhketqua_{index}.jpg", anh_ketqua)

                            else:
                                # print("sudoku sai")
                                # print(o_vuong_phong_doan)
                                seen[o_vuong_phong_doan] = False
                        # except Exception as e:
                        #     print("Đã xảy ra lỗi khi thực hiện hàm sudoku.dieu_kien_giai():", e)
                    except Exception as e:
                        print("Đã xảy ra lỗi khi thực hiện hàm xulianh.tao_mat_luoi():", e)
                Image = cv2.cvtColor(anh_ketqua, cv2.COLOR_BGR2RGB)        
                anh_giaisudoku = QImage(Image.data, Image.shape[1], Image.shape[0], Image.strides[0], QImage.Format_RGB888)
                Pic = anh_giaisudoku.scaled(960, 720, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def cop(self,anh) :
          # Tạo một đối tượng của lớp xulianh
        anhcop = anh.copy()  # Sử dụng dữ liệu ảnh được truyền từ phương thức run()
        Image2 = cv2.cvtColor(anhcop, cv2.COLOR_BGR2RGB)        
        anh_giaisudoku2 = QImage(Image2.data, Image2.shape[1], Image2.shape[0], Image2.strides[0], QImage.Format_RGB888)
        Pic2 = anh_giaisudoku2.scaled(960, 720, Qt.KeepAspectRatio)
        self.ImageUpdate2.emit(Pic2)
    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Tạo layout chính để chứa cả hai cửa sổ nhỏ
        main_layout = QHBoxLayout()

        # Tạo cửa sổ nhỏ thứ nhất
        feed1_layout = QVBoxLayout()
        self.FeedLabel1 = QLabel()
        feed1_layout.addWidget(self.FeedLabel1)

        # Tạo cửa sổ nhỏ thứ hai
        feed2_layout = QVBoxLayout()
        self.FeedLabel2 = QLabel()
        feed2_layout.addWidget(self.FeedLabel2)
        
        # Thêm cả hai cửa sổ nhỏ vào layout chính
        main_layout.addLayout(feed1_layout)
        main_layout.addLayout(feed2_layout)

        # Thêm layout chính vào cửa sổ ứng dụng
        self.setLayout(main_layout)

        # Tạo Worker1 để xử lý việc quay video
        self.Worker1 = Worker1()
        # Kết nối cả hai QLabel với cùng một Worker1
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot1)

        self.Worker1.ImageUpdate2.connect(self.ImageUpdateSlot2)


        # Bắt đầu quay video
        self.Worker1.start()

    def ImageUpdateSlot1(self, Image):
        # Cập nhật ảnh vào cửa sổ nhỏ thứ nhất
        self.FeedLabel1.setPixmap(QPixmap.fromImage(Image))

    def ImageUpdateSlot2(self, Image):
        # Cập nhật ảnh vào cửa sổ nhỏ thứ hai

        self.FeedLabel2.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        # Ngừng quay video và đóng ứng dụng
        self.Worker1.stop()
        QApplication.quit()
    
if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())



