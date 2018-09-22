#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <math.h>
#include <ctime>

using namespace cv;
using namespace std;

Mat3b src, hsv;

Mat train_data, x;
Mat labels (165,1,CV_32F);
int thresh = 100;
int max_thresh = 255;
RNG rng(12345); 

/// Function header
Mat CtC_features(Mat);
double euclidean_distance(Point2f a, Point2f b);
double angle(Point2f a, Point2f b);
Mat pre_process(Mat3b);
void model(int, void*);
void make_train_data(int, void*);

/** @function main */
int main()
{
	/// Load source image and convert it to gray
	
	//imshow("BGR", src);
	//Mat p = pre_process(src);
	//Mat temp = CtC_features(p);

	make_train_data(0, 0);
	cout << "Training data created. \n";
	cout << train_data.rows << " " << train_data.cols << endl;
	cout << labels.rows << " " << labels.cols << endl;
	x = train_data.row(0);
	model(0, 0); 
	//pre_process(0, 0);
	//CtC_features(0, 0);
	//waitKey(0);
	_getch();
	return(0);
}

/** @function thresh_callback */
Mat CtC_features(Mat canny)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int start = clock();
	
	/// Find contours
	findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	///Get the mass centers
	Point2f centroid;
	double xmid = 0, ymid = 0;
	long long n_points = 0;
	vector<Point2f> mc(contours.size());
	for (int i = 0; i<contours.size(); i++){
		n_points += contours[i].size();
		for (int j = 0; j < contours[i].size(); j++) {
			Point2f point = contours[i][j];
			xmid += point.x;
			ymid += point.y;
		}
	}
	centroid.x = xmid / n_points;
	centroid.y = ymid / n_points;
	//cout << centroid.x << " " << centroid.y<<endl;

	vector<pair<double, double>> feature_v;
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < contours[i].size(); j++) {
			Point2f curr = contours[i][j];
			double dist = euclidean_distance(centroid, curr);
			double ang = angle(centroid, curr) ;
			if (centroid.x > curr.x) ang += 3.14159;  //PI
			//cout << dist << " " << ang << endl;
			feature_v.push_back(make_pair(ang, dist));
		}
	}
	sort(feature_v.begin(), feature_v.end());

	int degree = 10;
	vector<double> temp, feature(360 / degree, 0);
	double interval = double((double(degree) / double(360)) * 2 * 3.14159); //5 degrees interval
	//cout << interval << endl;
	double ang = - 1.57079;
	int j = 0;
	for (int i = 0; i < feature_v.size(); i++) {
		while (feature_v[i].first > ang) {
			//cout << ang << endl;
			ang += interval;
			if (temp.empty()) temp.push_back(0);
			feature[j++] = *max_element(temp.begin(), temp.end());
			temp.clear();
		}
		temp.push_back(feature_v[i].second);
	}

	double maxf = *max_element(feature.begin(), feature.end());
	for (int i = 0; i < 360 / degree; i++)
		feature[i] /= maxf;

	Mat f = Mat(feature).reshape(0, 1);
	f.convertTo(f, CV_32F);

	int end = clock();
	//cout<< "Execution time : " << (end - start) / double(CLOCKS_PER_SEC) * 1000 << " ms"<< endl;
	for (int i = 0; i < feature.size(); i++)
		cout << feature[i] << endl;
	//cout << feature.size() << endl;  

	/// Draw contours
	/*Mat drawing = Mat::zeros(canny.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing); */

	return f;
}

Mat pre_process(Mat src) {
	//imshow("BGR", src);

	Mat1b mask1, mask2, mask3, mask;
	Mat grayImg, blurred, otsu, canny, hsv;

	//GaussianBlur(src, src2, Size(3, 3), 0, 0);

	cvtColor(src, hsv, COLOR_BGR2HSV);

	inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
	inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);

	mask = (mask1 | mask2);
	mask = ~mask;
	//imshow("Mask", mask);

	if (src.channels() == 3)
		cvtColor(src, grayImg, CV_BGR2GRAY);
	else if (src.channels() == 4)
		cvtColor(src, grayImg, CV_BGRA2GRAY);
	else grayImg = src;

	GaussianBlur(grayImg, blurred, Size(7, 7), 0, 0);
	//equalizeHist(blurred, blurred);
	double o = threshold(blurred, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	otsu = ~otsu;
	//imshow("Out", otsu);
	multiply(mask, otsu, otsu);

	Canny(otsu, canny, o, o * 1 / 2, 3, 1);
	//imshow("Canny", canny);
	return canny;
}

void model(int, void*) {
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::POLY);
	svm->setGamma(3);
	svm->setDegree(3);
	//svm->setC(0.025);
	cout << "Training started\n";
	svm->train(train_data, ml::ROW_SAMPLE, labels);
	cout << "Training complete\n";
	svm->save("SVM_5_new.xml");
	cout << svm->predict(x)<<endl;
}

void make_train_data(int, void*) {
	std::stringstream inp_path;
	vector<double> label;

	int l = 1;
	for (int i = 1; i <= 13; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/30/"<<i<<".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 17; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/40/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 16; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/50/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 19; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/60/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 15; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/70/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 18; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/80/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 19; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/90/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 19; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/100/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	l++;
	for (int i = 1; i <= 29; i++) {
		//cout << i << endl;
		inp_path.str(string());
		inp_path << "C:/Users/L3IN/Downloads/image data set/out/FP/" << i << ".bmp";
		Mat img = imread(inp_path.str());
		Mat processed_img = pre_process(img);
		Mat feature = CtC_features(processed_img);
		train_data.push_back(feature);
		label.push_back(l);
	}
	labels = Mat(label).reshape(0, 165);
	labels.convertTo(labels, CV_32S);
}

double euclidean_distance(Point2f a, Point2f b) {
	return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

double angle(Point2f a, Point2f b) {
	return atan((a.y - b.y) / (a.x - b.x));
}