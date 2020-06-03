import cv2
import numpy as np
from darkflow.net.build import TFNet        #fortraffic
import tensorflow as tf                     #fortraffic


##for traffic begin
config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False
with tf.Session(config=config) as sess:
    options = {
        'model': './cfg/yolo.cfg',
        'load': './yolov2.weights',
        'threshold': 0.4
    }
    tfnet=TFNet(options)
##for traffic end

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def process_image(image):
    grayscaled=grayscale(image)
    kernelSize = 7
    gaussianBlur = gaussian_blur(grayscaled,kernelSize)
    minThreshold = 195
    maxThreshold = 250
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
    lowerLeftPoint = [330, 530]
    upperLeftPoint = [470, 400]
    upperRightPoint = [580, 400]
    lowerRightPoint = [790, 530]
    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)
    rho = 1
    theta = np.pi / 180
    threshold = 38
    min_line_len = 10
    max_line_gap = 1000
    houged = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    colored_image = weighted_img(houged, image)
    # traffic
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(img)
    print(results)
    for (i, result) in enumerate(results):
        x = result['topleft']['x']
        w = result['bottomright']['x'] - result['topleft']['x']
        y = result['topleft']['y']
        h = result['bottomright']['y'] - result['topleft']['y']
        cv2.rectangle(colored_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_position = (x + int(w / 2)), abs(y - 10)
        cv2.putText(colored_image, result['label'], label_position, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
    #traffic ends
    return colored_image

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img, low_threshold,high_threshold)

def region_of_interest(img,vertices):
    mask=np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else :
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img, lines, color=(0,0,255), thickness=2):
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    all_right_grad = []
    all_right_y=[]
    all_right_x=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1,y2),ymin_global)
            if(gradient>0):
                all_left_grad+=[gradient]
                all_left_y+=[y1,y2]
                all_left_x+=[x1,x2]
            else :
                all_right_grad+=[gradient]
                all_right_y+=[y1,y2]
                all_right_x+=[x1,x2]
            left_mean_grad=np.mean(all_left_grad)
            left_y_mean=np.mean(all_left_y)
            left_x_mean=np.mean(all_left_x)
            left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
            right_mean_grad = np.mean(all_right_grad)
            right_y_mean = np.mean(all_right_y)
            right_x_mean = np.mean(all_right_x)
            right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
            if((len(all_left_grad)> 0) and (len(all_right_grad) > 0)):
                upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
                lower_left_x = int((ymax_global - left_intercept)/left_mean_grad)
                upper_right_x = int((ymin_global - right_intercept)/right_mean_grad)
                lower_right_x = int((ymax_global - right_intercept)/right_mean_grad)
                cv2.line(img, (upper_left_x, ymin_global),(lower_left_x, ymax_global), color, thickness)
                cv2.line(img,(upper_right_x,ymin_global), (lower_right_x, ymax_global),color,thickness)


def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3),dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, beta =1.0, gamma = 0.0) :
    return cv2.addWeighted(initial_img,alpha, img, beta ,gamma)

cap = cv2.VideoCapture('Testingt1.mp4')
if cap.isOpened()==False :
    print('Error file not found')
while cap.isOpened():
    ret,frame = cap.read()
    framenew = process_image(frame)
    if ret==True :
        cv2.imshow('frame',framenew)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()