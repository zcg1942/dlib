

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
			frontal_face_detector detector = get_frontal_face_detector();//������������õ������ı߽��
			//�����ʹ�þ����HOG���������Է�������ͼ��������ͻ������
			shape_predictor pose_model;//shape_predictor����Ԥ�����ϱ�־��λ��
			deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;//��ȡ�Ѿ�ѵ���õ�ģ�ͣ��� iBUG 300-W�������ݼ�����ֹ���ã���ѵ���õ� http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.

			// Grab and process frames until the main window is closed by the user.  

			// Grab a frame  
			cv::Mat temp;
			//cap >> temp;  
			temp = cv::imread("timg.jpg", 1);
			//cv::VideoCapture cap(0);Ҳ���Դ�����ͷ��ȡ��ǰ����

			cv_image<bgr_pixel> cimg(temp);//������ֵ dlib��ͼ���ʽ��array2d
			// Detect faces   
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.  
			std::vector<full_object_detection> shapes;//���ڴ�������������
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));//68�����������pose

			if (!shapes.empty())
			{
				for (int j = 0; j < shapes.size(); j++)//����ÿ����⵽������
				{
					for (int i = 0; i < 68; i++)//����ÿ��������68��������
					{
						circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);//�뾶��3����ɫ�Ǻ�ɫ��-1��ʾԲ�����
						putText(temp, to_string(i), cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 4);//���������1��������4����to_string��intתΪstring������c++11����ӵ�
					}
				}
			}

			//Display it all on the screen  
			imshow("Dlib������", temp);
			imwrite("Dlib������1.jpg", temp);

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