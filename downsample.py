import cv2
import pydicom

#dcm = pydicom.read_file(r'D:\python\EDSR-Tensorflow-master\Experiment2\x2out_dcm\x2out1.DCM')
img = cv2.imread(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\in_1.jpg',cv2.IMREAD_GRAYSCALE)
#img = dcm.pixel_array
x,y = img.shape
out = cv2.resize(img,(x*2,y*2),interpolation=cv2.INTER_CUBIC)
#dcm.Rows,dcm.Columns = out.shape
#dcm.PixelData = out.tobytes()
#dcm.save_as("downcubic_x2_1.DCM")
cv2.imwrite(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\cubicup_1.jpg',out,[int(cv2.IMWRITE_JPEG_QUALITY),100])
