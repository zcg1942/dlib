

	/*dlib::deserialize("D:\\program\Anaconda2\\Tools\dlib-18.17\\shape_predictor_68_face_landmarks.dat") >> pose_model;
	cv::Mat img = imread("D:\\program\Anaconda2\\Tools\dlib-18.17\\zkaj.jpg", 1);*/
#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  

	using namespace dlib;
	using namespace std;

	int main()
	{
		try
		{
			// Load face detection and pose estimation models.  
			frontal_face_detector detector = get_frontal_face_detector();//正脸检测器，得到人脸的边界框
			//检测器使用经典的HOG特征和线性分类器，图像金字塔和滑窗完成
			shape_predictor pose_model;//shape_predictor用于预测脸上标志的位置
			deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;//读取已经训练好的模型，在 iBUG 300-W人脸数据集（禁止商用）上训练得到 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.

			// Grab and process frames until the main window is closed by the user.  

			// Grab a frame  
			cv::Mat temp;
			//cap >> temp;  
			temp = cv::imread("timg.jpg", 1);
			//cv::VideoCapture cap(0);也可以从摄像头获取当前画面

			cv_image<bgr_pixel> cimg(temp);//拷贝赋值 dlib的图像格式是array2d
			// Detect faces   
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.  
			std::vector<full_object_detection> shapes;//用于存放特征点的容器
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));//68个点组成人脸pose

			if (!shapes.empty())
			{
				for (int j = 0; j < shapes.size(); j++)//遍历每个检测到的人脸
				{
					for (int i = 0; i < 68; i++)//遍历每个人脸的68个特征点
					{
						circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);//半径是3，颜色是红色。-1表示圆被填充
						putText(temp, to_string(i), cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 4);//线条宽度是1，字形是4邻域。to_string将int转为string，这是c++11新添加的
					}
				}
			}

			//Display it all on the screen  
			imshow("Dlib特征点", temp);
			imwrite("Dlib特征点1.jpg", temp);

			cv::waitKey(0);
		}
		catch (serialization_error& e)
		{
			cout << "You need dlib's default face landmarking model file to run this example." << endl;
			cout << "You can get it from the following URL: " << endl;
			cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			cout << endl << e.what() << endl;
		}
	}