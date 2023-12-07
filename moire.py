import cv2
import numpy as np


def reduce_moire(filepath):
    # 减少摩尔纹  不是完全消除摩尔纹，但可以显著减少其影响
    image = cv2.imread(filepath)
    f = np.fft.fft2(image)  # 傅立叶变换
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 计算振幅谱
    blurred = cv2.GaussianBlur(magnitude_spectrum, (21, 21), 0)  # 高斯滤波器
    # 傅立叶变换反转
    back_shifted = np.fft.ifftshift(blurred)
    back_to_real_domain = np.fft.ifft2(back_shifted)
    image_denoised = np.real(back_to_real_domain)
    cv2.imshow("image_denoised", image_denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reduce_moire2(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用高斯滤波器去除噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 计算幅度谱
    f = np.fft.fft2(blurred)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 获取幅度谱的尺寸
    rows, cols = magnitude_spectrum.shape

    # 创建一个掩模，用于跳过全黑的区域
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # 遍历幅度谱，将全黑的区域标记为1
    for i in range(rows):
        for j in range(cols):
            if magnitude_spectrum[i, j] != 0:
                mask[i, j] = 1

    # 对幅度谱进行掩模处理
    masked_spectrum = np.ma.masked_array(magnitude_spectrum, mask=mask)

    # 计算掩模处理后的幅度谱的平均值
    avg_magnitude = np.mean(masked_spectrum)

    # 创建一个新的掩模，用于标记全黑的区域
    mask2 = np.zeros((rows, cols), dtype=np.uint8)

    # 将平均值填充到掩模中
    for i in range(rows):
        for j in range(cols):
            if magnitude_spectrum[i, j] == 0:
                mask2[i, j] = 1
            else:
                mask2[i, j] = 0 if magnitude_spectrum[i, j] < avg_magnitude else 1

    # 对幅度谱进行掩模处理
    masked_spectrum2 = np.ma.masked_array(magnitude_spectrum, mask=mask2)

    # 计算掩模处理后的幅度谱的平均值
    avg_magnitude2 = np.mean(masked_spectrum2)

    # 创建一个新的幅度谱，用于存储去除摩尔纹后的图像
    cleaned_spectrum = np.zeros((rows, cols), dtype=np.uint8)

    # 将掩模处理后的幅度谱的平均值填充到新的幅度谱中
    for i in range(rows):
        for j in range(cols):
            if mask2[i, j] == 0:
                cleaned_spectrum[i, j] = 255
            else:
                cleaned_spectrum[i, j] = magnitude_spectrum[i, j]

    # 对新的幅度谱进行逆傅里叶变换
    cleaned_image = np.fft.ifft2(np.fft.ifftshift(cleaned_spectrum))

    # 转换回uint8类型
    cleaned_image = np.uint8(np.abs(cleaned_image))
    cv2.imshow("cleaned_image", cleaned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reduce_moire3(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # 使用高斯滤波器去除噪声，能去除一点
    # 使用阈值处理去除摩尔纹
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 对阈值处理后的图像进行二值化
    _, binary = cv2.threshold(thresholded, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    reduce_moire("/tmp/me.jpeg")  # 结果几乎全黑的
    reduce_moire2("/tmp/me.jpeg")  # 结果全黑的
    reduce_moire3("/tmp/me.jpeg")  # 消了一点
