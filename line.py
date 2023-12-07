import cv2
import numpy as np


def line_detect(image_path, threshold=200):
    """检测图像中的线段
    :param threshold: 线段点数阈值，小于此值的不检测

    https://blog.csdn.net/ftimes/article/details/106816736
    简单场景下 HoughLines 够用，实际场景可能存在严重的误检
    HoughLinesP 在复杂场景比 HoughLines 识别的更少直线
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘检测,50和150是两个用于确定边缘的强度阈值，apertureSize表示梯度计算的卷积核大小
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow('edges', edges)
    # 直线段检测（标准霍夫变换）,edges是二值图像；1表示精度，即每个像素点都有可能成为线段的端点；np.pi/180 是角度搜索精度，表示搜索所有可能的角度
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("detected lines", image)
    cv2.waitKey(0)  # Wait for a key press and close the window. Esc
    cv2.destroyAllWindows()


def line_detect_p(image_path, threshold=200, minLineLength=300, maxLineGap=100):
    """使用概率霍夫变换（HoughLinesP）检测图像中的线段"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("detected lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_detect_padding(image_path, padding):
    """检测图像中的线段，距离图像边缘小于 padding 的线段不检测"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # 图片的高，图片的宽，色彩通道
    cropped = image[
        padding : height - padding, padding : width - padding
    ]  # 四周都裁剪掉 padding
    # cropped = cv2.resize(image, (width, height // 2)) # 把图像高度减半，裁剪
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("detected lines", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_detect_direction(image_path, horizontal=True, vertical=True, angel_pi=8):
    """
    检测图像中的 水平线 或 竖直线
    :param image_path: 图像路径
    :param horizontal: 是否检测水平线
    :param vertical: 是否检测竖直线
    :param angel_pi: 线段与水平线/铅锤线之间的夹角阈值，单位为弧度，默认是 8，意味着阈值是pi/8个弧度(22.5度)

    铅锤线的弧度为pi
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            if (
                horizontal
                and (abs(theta) > ((np.pi / 2) - (np.pi / angel_pi)))
                and (abs(theta) < ((np.pi / 2) + (np.pi / angel_pi)))
            ):  # 只画水平线
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif (vertical and (abs(theta) < (np.pi / angel_pi))) or (
                vertical and (abs(theta) > (np.pi - (np.pi / angel_pi)))
            ):  # 只画竖直线
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("detected lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "/tmp/test.jpg"
    line_detect(image_path)
    line_detect_p(image_path)
    line_detect_padding(image_path, 50)
    line_detect_direction(image_path, True, False)
    line_detect_direction(image_path, False, True)
