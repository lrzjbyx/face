# -- coding: utf-8 --
# @Time : 2022/3/17 15:53
# @Author : liufeng
# @Email : jackmca@163.com
# @File : main.py



import math
import time
import cv2
import os
import openface
import numpy as np
import itertools as it


# 人脸识别
class FaceLocation(object):
    align = openface.AlignDlib(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'),
                                            "shape_predictor_68_face_landmarks.dat"))

    @staticmethod
    def fitLine(points):
        dist_func_names = it.cycle('DIST_L2 DIST_L1 DIST_L12 DIST_FAIR DIST_WELSCH DIST_HUBER'.split())
        vx, vy, cx, cy = cv2.fitLine(points, getattr(cv2, next(dist_func_names)), 0, 0.01, 0.01)
        angle = np.arctan(vy / vx) * 57.29577
        return angle[0]


    def __init__(self, image):
        # 扩充边界
        zero = np.zeros((image.shape[0]+20,image.shape[1]+20,image.shape[2]),np.uint8)
        zero[:] = 255
        zero[10:-10,10:-10,0:3] = image
        image = zero

        self._image = image.copy()
        self.image = image

        self.height = self._image.shape[0]
        self.width = self._image.shape[1]

        # 免冠照的位置
        self.confidence_image = None
        self.confidence_x = 0
        self.confidence_y = 0
        self.confidence_w = 0
        self.confidence_h = 0

        # 裁剪位置
        self.cutting_image = None
        self.cutting_x1 = 0
        self.cutting_y1 = 0
        self.cutting_x2 = 0
        self.cutting_y2 = 0

        # 脸部位置
        self.face_image = None
        self.face_x = 0
        self.face_y = 0
        self.face_w = 0
        self.face_h = 0

        #
        self.key_points = []

    # 投影确定表格位置
    def table_position(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        blur_not = cv2.bitwise_not(blur)
        AdaptiveThreshold = cv2.adaptiveThreshold(blur_not, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,
                                                  -2)

        horizontal = AdaptiveThreshold.copy()
        vertical = AdaptiveThreshold.copy()
        scale = 20

        horizontalSize = int(horizontal.shape[1] / scale)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        verticalsize = int(vertical.shape[1] / scale)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        kernel = np.ones((7, 7), np.uint8)
        dilation = cv2.dilate(horizontal + vertical, kernel, )

        ret, mask = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY_INV)

        return mask

    # 分离文字和表格
    def separate_table_script(self, image):
        # cv2.imwrite("table_canva2222s.png", image)
        img = image.copy()
        # 扩大
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        # 表格分片
        table_canvas = img.copy()

        # 文字分片
        script_canvas = np.zeros(img.shape, np.uint8)
        script_canvas[:, :, :] = 255

        # mser创建
        mser = cv2.MSER_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 文字区域
        regions = mser.detectRegions(gray)

        # 坐标点
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
        # 绘制文字轮廓
        for hull in hulls:
            cv2.fillPoly(script_canvas, [np.array([list(p[0]) for p in hull], np.uint64)], (0, 0, 0))
            cv2.fillPoly(table_canvas, [np.array([list(p[0]) for p in hull], np.uint64)], (255, 255, 255))

        # 恢复大小
        table_canvas = cv2.resize(table_canvas, (image.shape[1], image.shape[0]))
        script_canvas = cv2.resize(script_canvas, (image.shape[1], image.shape[0]))

        # cv2.imwrite("table_canvas.png",table_canvas)
        # cv2.imwrite("script_canvas.png",script_canvas)

        return (table_canvas, script_canvas)

    # 最大外接矩形
    def bounding_box(self, image):
        image = image.copy()
        (h, w) = image.shape
        # black_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, threshold = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("thresholdssss.png",image)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if 2000 < cv2.contourArea(cnt) < w * h]
        cutting_w = 0
        cutting_h = 0
        cutting_x = 0
        cutting_y = 0

        face_x = self.face_x - self.cutting_x1
        face_y = self.face_y - self.cutting_y1

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 将轮廓信息转换成(x, y)坐标，并加上矩形的高度和宽度
            if w > cutting_w and h > cutting_h and x != 0 and y != 0 and x < face_x and face_y > y:
                cutting_w = w
                cutting_h = h
                cutting_x = x
                cutting_y = y

        # cv2.rectangle(image, (cutting_x, cutting_y), (cutting_x + cutting_w, cutting_y + cutting_h), (255, 255, 0),
        #               5)  # 画出矩形  cutting_y = y

        return cutting_x, cutting_y, cutting_h, cutting_w

    # Key point inspection
    def face_key_point_inspection(self):

        # 左右眼睛眉毛
        left_eye_brow = np.mean(np.array(self.key_points[17:22]).T, axis=1)
        right_eye_brow = np.mean(np.array(self.key_points[22:27]).T, axis=1)

        # 左右眼睛
        left_eye = np.mean(np.array(self.key_points[36:42]).T, axis=1)
        right_eye = np.mean(np.array(self.key_points[42:48]).T, axis=1)

        # 嘴巴位置
        mouth = np.array(self.key_points[48:68])
        mouth_angle = FaceLocation.fitLine(mouth)

        # 鼻梁位置
        nose = np.array(self.key_points[27:31])
        nose_angle = FaceLocation.fitLine(nose)


        #左右眉毛斜率
        eye_brow_k = 100
        if not left_eye_brow[1] - right_eye_brow[1] == 0:
            eye_brow_k = math.fabs(np.arctan(-(left_eye_brow[0] - right_eye_brow[0]) / (left_eye_brow[1] - right_eye_brow[1])) * 57.29577)

        # 左右眼睛斜率
        eye_k = 100
        if not left_eye[1] - right_eye[1] == 0:
            eye_k = math.fabs(np.arctan(-(left_eye[0] - right_eye[0]) / (left_eye[1] - right_eye[1])) * 57.29577)


        # 眼睛和眉毛斜率不能差别不能太大
        if abs(eye_brow_k-eye_k) > 5 and (eye_k == 100 or eye_brow_k == 100 ):
            return False

        # 鼻梁和嘴巴基本保持垂直
        if abs(nose_angle) + abs(mouth_angle) <80:
            return False

        return True

    def face_location(self):
        face_rect = FaceLocation.align.getLargestFaceBoundingBox(self.image)

        if not face_rect is None:
            bb = FaceLocation.align.findLandmarks(self.image, face_rect)
            self.key_points = bb

            if not self.face_key_point_inspection():
                return False

            self.face_x = face_rect.left()
            self.face_y = face_rect.top()
            self.face_w = face_rect.width()
            self.face_h = face_rect.height()

            # ###################人脸矩形#####################
            # canvas2 = self.image.copy()
            # # cv2.rectangle(canvas,(self.face_x,self.face_y),(self.face_x+self.face_w,self.face_y+self.face_h),(0,255,0),2)
            # cv2.rectangle(canvas2, (self.face_x, self.face_y), (self.face_x + self.face_w, self.face_y + self.face_h),
            #               (0, 255, 0), 2)
            # cv2.imwrite("canvas_face_rect.png", canvas2)
            # ########################################


            self.face_image = self.image[self.face_y:self.face_y + self.face_h, self.face_x:self.face_x + self.face_w]



            cv2.circle(self.image, center=(self.face_x + int(self.face_w / 2), self.face_y),
                       radius=int(self.face_w / 2), color=(0, 0, 255), thickness=-1)
            cv2.rectangle(self.image, (self.face_x, self.face_y),
                          (self.face_x + self.face_w, self.face_y + self.face_h), (0, 0, 255), -1)
            cv2.rectangle(self.image, (self.face_x - int(self.face_w * 0.2), self.face_y + self.face_h), (
                self.face_x + self.face_w + int(self.face_w * 0.2), self.face_y + self.face_h + int(self.face_h / 2)),
                          (0, 0, 255), -1)

            self.cutting_x1 = self.face_x - self.face_w
            self.cutting_y1 = self.face_y - int(self.face_h * 1.65)
            self.cutting_x2 = self.face_x + self.face_w * 2
            self.cutting_y2 = self.face_y + self.face_h + int(self.face_h * 1.65)

            if self.cutting_x1 < 0 and self.cutting_x2 > self.width:
                self.cutting_x1 = 0
                self.cutting_x2 = self.width
            elif self.cutting_y2 > self.height and self.cutting_y1 < 0:
                self.cutting_y1 = 0
                self.cutting_y2 = self.height
            elif self.cutting_y1 < 0:
                self.cutting_y1 = 0
            elif self.cutting_y2 > self.height:
                self.cutting_y2 = self.height
            elif self.cutting_x1 < 0:
                self.cutting_x1 = 0
            elif self.cutting_x2 > self.width:
                self.cutting_x2 = self.width

            # 防溢出处理
            if self.cutting_y1< 0 or self.cutting_x1< 0:
                self.cutting_y1 = 0
            if self.cutting_y2 > self.height:
                self.cutting_y2 = self.height
            if self.cutting_x2 > self.width:
                self.cutting_x2 = self.width

            self.cutting_image = self.image[self.cutting_y1:self.cutting_y2, self.cutting_x1:self.cutting_x2]
            # cv2.imwrite("self.cutting_image.png", self.cutting_image)

            ##
            gray = cv2.cvtColor(self.cutting_image, cv2.COLOR_BGR2GRAY)

            ret, threshold = cv2.threshold(gray, int(np.mean(gray))+20, 255, cv2.THRESH_BINARY)

            (table_canvas, script_canvas) = self.separate_table_script(cv2.merge([threshold, threshold, threshold]))

            mask = self.table_position(table_canvas)

            mask_points = np.where(mask == 0)

            for p in range(len(mask_points[0])):
                threshold[mask_points[0][p]][mask_points[1][p]] = 255



            (x, y, h, w) = self.bounding_box(threshold)

            if not (h == 0 or w == 0):
                self.confidence_x = x + self.cutting_x1
                self.confidence_y = y + self.cutting_y1
                self.confidence_w = w
                self.confidence_h = h

            else:
                self.confidence_x = self.face_x - int(self.face_w * 0.2)
                self.confidence_y = self.face_y - int(self.face_h * 0.35)
                self.confidence_w = self.face_w + int(self.face_w * 0.2) * 2
                self.confidence_h = self.face_h + int(self.face_h * 0.35) * 2

            self.confidence_image = self._image[self.confidence_y:self.confidence_y + self.confidence_h,
                                    self.confidence_x:self.confidence_x + self.confidence_w]

            # cv2.rectangle(self._image, (self.confidence_x, self.confidence_y),
            #               (self.confidence_x + self.confidence_w, self.confidence_y + +self.confidence_h),
            #               (0, 255, 255), 20)

            return True

    def run(self):
        if self.face_location():
            return (True, self._image[10:-10,10:-10], self.confidence_image)
        else:
            return (False, self._image[10:-10,10:-10], self.confidence_image)


if __name__ == "__main__":

    rootdir = "resource/"
    # rootdir = "resource/0.4-1-5-003.jpg"
    list_paths = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list_paths)):
        path = os.path.join(rootdir, list_paths[i])
        start = time.time()
        try:
            print(path)
            img = cv2.imread(path)
            face = FaceLocation(img)
            ret, image, confidence_image = face.run()
            if ret:
                # 原图
                cv2.imwrite("head_picture/canvas" + os.path.basename(path), image)
                # 免冠照
                cv2.imwrite("head_picture/canvas--" + os.path.basename(path), confidence_image)
        except:
            continue
        end = time.time()
        print("running spend time :{0}s".format(str(round(end - start, 2))))


