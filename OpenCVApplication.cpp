// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <conio.h>
#include <opencv2/core/utils/logger.hpp>




wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}
void borderTracing(Mat_<uchar> src) {

	std::vector <int> chainCode;

	bool done = false;

	int height = src.rows;
	int width = src.cols;

	
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Point borderPoints[10000];

	int ctr = 0; 
	Point point;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src(i, j) == 0) {
				point = Point(j, i);
				borderPoints[ctr] = point;
				ctr++;
				done = true;
				break;
			}
		}

		if (done) {
			break;
		}
	}


	int dir = 7;
	while (!(borderPoints[0] == borderPoints[ctr - 2] && borderPoints[1] == borderPoints[ctr - 1]) || (ctr <= 2)) {
		if (dir % 2 == 0) { 
			dir = (dir + 7) % 8;
		}
		else { 
			dir = (dir + 6) % 8;
		}

		for (int k = dir; k < dir + 8; k++) {
			uchar neighbour = src(point.y + di[k % 8], point.x + dj[k % 8]);
			if (neighbour == 0) {
				borderPoints[ctr] = Point(point.x + dj[k % 8], point.y + di[k % 8]);
				ctr++;
				point = Point(point.x + dj[k % 8], point.y + di[k % 8]);
				chainCode.push_back(dir);
				dir = k % 8;
				break;
			}
		}
	}

	Mat_<uchar> dst(height, width);
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst(i, j) = 0;
		}
	}

	for (int i = 0; i < ctr; i++) {
		dst(borderPoints[i].y, borderPoints[i].x) = 255;
	}

	imshow("Initial image", src);
	imshow("Border", dst);
	waitKey(0);

	printf("AC\n");
	for (int i = 0; i < chainCode.size(); i++) {
		printf("%d ", chainCode[i]);
	}

	printf("\nDC\n");
	for (int i = 1; i < chainCode.size(); i++) {
		int deriv = (chainCode[i] - chainCode[i - 1] + 8) % 8;
		printf("%d ", deriv);
	}

	_getch();
}

void contourColouring()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		FILE* file_pointer;
		file_pointer = fopen("reconstruct.txt", "r");
		int dx[8] = { 0,-1,-1,-1,0,1,1,1 };
		int dy[8] = { 1,1,0,-1,-1,-1,0,1 };

		int x, y, size;
		int directions[10000];
		fscanf(file_pointer, "%d %d %d", &x, &y, &size);

		for (int i = 0; i < size; i++) {
			fscanf(file_pointer, "%d", &directions[i]);
		}


		Point firstPoint;
		firstPoint = Point(y, x);
		src.at<uchar>(firstPoint) = 255;


		for (int i = 0; i < size; i++) {
			y = y + dy[directions[i]];
			x = x + dx[directions[i]];
			firstPoint = Point(y, x);
			src.at<uchar>(firstPoint) = 0;
		}

		imshow("Excellent", src);
	}

}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void addGrey() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int plus = 50;
				uchar neg;
				if (val + plus <= 255) {
					neg = val + plus;

				}
				else {
					neg = 255;

				}
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grayer image", dst);
		waitKey();
	}

}



void colorImage() {

	int height = 256;
	int width = 256;

	Mat dst = Mat(height, width, CV_8UC3, Scalar(0, 255, 255));


	int i, j;
	for (i = 0; i <= height / 2; i++)
		for (j = 0; j < width / 2; j++)
		{
			dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

		}
	for (i = height / 2; i < height; i++)
		for (j = width / 2; j < width; j++)
			dst.at<Vec3b>(i, j) = Vec3b(0, 0, 255);

	for (i = height / 2; i < height; i++)
		for (j = 0; j < width / 2; j++)
			dst.at<Vec3b>(i, j) = Vec3b(0, 255, 0);


	imshow("help me", dst);
	waitKey();


}
void inverseMatrix() {

	float vals[9] = { 1.1, 2, 3, 4, 5, 6, 7, 8, 9 };
	Mat M(3, 3, CV_32FC1, vals); //4 parameter constructor

	std::cout << M.inv() << std::endl;
	getchar();
	getchar();

}

void colorToGray() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat grey(height, width, CV_8UC1);
		

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b p = src.at<Vec3b>(i, j);

				grey.at <uchar>(i, j) = uchar((p[0]+p[1]+p[2])/3);
				

			}


		imshow("input image", src);
		imshow("grey transformation", grey);

		waitKey();
	}
}

void RGBChannel() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat Red(height, width, CV_8UC3);
		Mat Green(height, width, CV_8UC3);
		Mat Blue(height, width, CV_8UC3);


		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b p = src.at<Vec3b>(i, j);

				Red.at <Vec3b>(i, j) = Vec3b(0, 0, p[2]);
				Green.at <Vec3b>(i, j) = Vec3b(0, p[1], 0);
				Blue.at <Vec3b>(i, j) = Vec3b(p[0], 0, 0);

			}


		imshow("input image", src);
		imshow("RChannel", Red);
		imshow("GChannel", Green);
		imshow("BChannel", Blue);

		waitKey();
	}
}

void greyscaleToBW()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		int t = 127;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				///Vec3b p = src.at<Vec3b>(i, j);
				if(src.at <uchar>(i, j) <= t)
					dst.at <uchar>(i, j) = 0;
				else
					dst.at <uchar>(i, j) = 255;


			}


		imshow("input image", src);
		imshow("BW",dst);

		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}


void myHSV() {	

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		float v, s, h;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				float r =(float)p[2] / 255;
				float g = (float)p[1] / 255;
				float b = (float)p[0] / 255;

				float M = max(r, max(g, b));
				float m = min(r, min(g, b));
				float C = M - m;

				///Value
				 v = M;
				 v = v * 255;
				 V.at<uchar>(i, j) = v;


				///Saturation
				if(v != 0)
					s = C / M;
				else // grayscale
					s = 0;

				s = s * 255;
				S.at<uchar>(i, j) = s;


				///Hue

				if(C != 0) {
					if (M == r) h = 60 * (g - b) / C;
					if (M == g) h = 120 + 60 * (b - r) / C;
					if (M == b) h = 240 + 60 * (r - g) / C;
				}
				else // grayscale
					h = 0;
				if(h < 0)
					h = h + 360;

				h = h * 255 / 360;
				H.at<uchar>(i, j) = h;

			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}

}


void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		dst = src.clone();

		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void myHist() {
	int h[256] = { 0 };

	float norm[256] = { 0 };
 	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				h[src.at<uchar>(i, j)]++;
 
			} 
		int M = height * width;
		for (int i = 0; i <= M; i++)
			norm[i] =(float) h[i] / M;

		imshow("input image", src);
		showHistogram("Histogram", h, 256, 500);
		waitKey();
	}

}

void myMultiLevelConspiracy() {

	int h[256] = { 0 };
	float th = 0.0003;
	float norm[256] = { 0 };
 	char fname[MAX_PATH];

	int maxims[255] = { 0 };



	int index = 1;

	while (openFileDlg(fname))
	{



		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int wh = 5;


		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {

				h[src.at<uchar>(i, j)]++;

			}
		float M = height * width;
		for (int i = 0; i < 256; i++)
			norm[i] = (float)h[i] / M;

		///for(int i = wh ; i<=255-wh; i++)
			//for(int j = 0; j<=M; j++)
		for (int i = wh; i <= 255-wh; i++)
		{
			float maxi = norm[i];
			float sum = 0;
			for (int j = i - wh; j <= i + wh; j++)
			{
				sum = sum + norm[j];
				if (norm[j] > maxi)
					maxi = norm[j];

			}
			float average = sum / (2 * wh + 1);
			if (maxi > average + th) {
				if (maxi == norm[i]) {
					maxims[index] = i;
					index++;
				}
				

			}
				
		}

		maxims[index] = 255;
		Mat dst(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at <uchar>(i, j);
				for(int g = 0; g< index; g++)
					if (val >= maxims[g] && val <= maxims[g + 1])
					{
						if ((maxims[g + 1] - val) < (val - maxims[g]))
							dst.at<uchar>(i, j) = maxims[g + 1];
						else
							dst.at<uchar>(i, j) = maxims[g];

					}

			}


		imshow("input image", src);
		imshow("Multilevel Thresholding", dst);
		waitKey();
	}

}
void myFloyd_Steinberg() {


	int h[256] = { 0 };
	float th = 0.0003;
	float norm[256] = { 0 };
	char fname[MAX_PATH];

	int maxims[255] = { 0 };

	int index = 1;

	while (openFileDlg(fname))
	{



		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int wh = 5;


		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {

				h[src.at<uchar>(i, j)]++;

			}
		float M = height * width;
		for (int i = 0; i < 256; i++)
			norm[i] = (float)h[i] / M;

		///for(int i = wh ; i<=255-wh; i++)
			//for(int j = 0; j<=M; j++)
		for (int i = wh; i <= 255 - wh; i++)
		{
			float maxi = norm[i];
			float sum = 0;
			for (int j = i - wh; j <= i + wh; j++)
			{
				sum = sum + norm[j];
				if (norm[j] > maxi)
					maxi = norm[j];

			}
			float average = sum / (2 * wh + 1);
			if (maxi > average + th) {
				if (maxi == norm[i]) {
					maxims[index] = i;
					index++;
					
				}


			}

		}

		maxims[index] = 255;
		Mat dst(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at <uchar>(i, j);
				for (int g = 0; g < index; g++)
					if (val >= maxims[g] && val <= maxims[g + 1])
					{
						if ((maxims[g + 1] - val) < (val - maxims[g]))
							dst.at<uchar>(i, j) = maxims[g + 1];
						else
							dst.at<uchar>(i, j) = maxims[g];

					}

			}





		for (int i = 1; i < height-1; i++)
			for (int j = 1; j < width-1; j++) {
				uchar val = src.at <uchar>(i, j);
				uchar dstVal = dst.at<uchar>(i, j);
				int err = 0;
				err = val - dstVal;

				dst.at<uchar>(i, j + 1) = min(max(0, dst.at<uchar>(i, j + 1) + 7.0 / 16.0 * err), 255);

				dst.at<uchar>(i+1, j + 1) = min(max(0, dst.at<uchar>(i+1, j + 1) + 1.0 / 16.0 * err), 255);

				dst.at<uchar>(i+1, j ) = min(max(0, dst.at<uchar>(i+1, j) + 5.0 / 16.0 * err), 255);

				dst.at<uchar>(i-1, j -1) = min(max(0, dst.at<uchar>(i-1, j + 1) + 3.0 / 16.0 * err), 255);

			}


		imshow("input image", src);
		imshow("Dithering", dst);
		waitKey();
	}


}

void dilation() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst;
		dst = src.clone();

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int i = 1; i < height-1; i++)
			for (int j = 1; j < width-1; j++) {
				
				///Vec3b p = src.at<Vec3b>(i, j);
					if(src.at<uchar>(i,j)== 0)
					{
						for (int k = 0; k < 4; k++) {
							dst.at<uchar>(i + dx[k], j + dy[k]) = 0;
						}

				}
				
			}


		imshow("input image", src);
		imshow("agiutor", dst);

		waitKey();
	}
}

void dilationNtimes()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst;
		dst = src.clone();
		int n = 7;

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int count = 1; count <= n; count++) {
			for (int i = 1; i < height - 1; i++)
				for (int j = 1; j < width - 1; j++) {

					///Vec3b p = src.at<Vec3b>(i, j);
					if (src.at<uchar>(i, j) == 0)
					{
						for (int k = 0; k < 4; k++) {
							dst.at<uchar>(i + dx[k], j + dy[k]) = 0;
						}

					}

				}
			src = dst.clone();
		}

		imshow("input image", src);
		imshow("agiutor", dst);

		waitKey();
	}

}

void notDilation() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();
		dst.setTo(255);

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {
				
				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0) {
					bool flag = 0;

					for (int k = 0; k < 4; k++)
					{
						if (src.at<uchar>(i + dx[k], j + dy[k]) != 0) {

							flag = 1;
							break;
						}
							
					}
					if (flag == 0)
							dst.at<uchar>(i , j ) = 0;

				}


			}


		imshow("input image", src);
		imshow("agiutor", dst);

		waitKey();
	}

}

void notDilationNtimes() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();
		dst.setTo(255);
		Mat showy = src.clone();

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		int n = 20;

		for (int count = 1; count <= n; count++) {

			for (int i = 1; i < height - 1; i++)
				for (int j = 1; j < width - 1; j++) {

					///Vec3b p = src.at<Vec3b>(i, j);
					if (src.at<uchar>(i, j) == 0) {
						bool flag = 0;

						for (int k = 0; k < 4; k++)
						{
							if (src.at<uchar>(i + dx[k], j + dy[k]) != 0) {

								flag = 1;
								break;
							}

						}
						if (flag == 0)
							dst.at<uchar>(i, j) = 0;

					}


				}
			src = dst.clone();
			dst.setTo(255);
		}

		imshow("input image", showy);
		imshow("agiutor", dst);

		waitKey();
	}
}

void closing() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst;
		dst = src.clone();

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {

				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 4; k++) {
						dst.at<uchar>(i + dx[k], j + dy[k]) = 0;
					}

				}

			}

		src = dst.clone();
		dst.setTo(255);


		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {

				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0) {
					bool flag = 0;

					for (int k = 0; k < 4; k++)
					{
						if (src.at<uchar>(i + dx[k], j + dy[k]) != 0) {

							flag = 1;
							break;
						}

					}
					if (flag == 0)
						dst.at<uchar>(i, j) = 0;

				}


			}

		imshow("input image", src);
		imshow("agiutor", dst);

		waitKey();
	}
}

void open() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();
		dst.setTo(255);

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {

				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0) {
					bool flag = 0;

					for (int k = 0; k < 4; k++)
					{
						if (src.at<uchar>(i + dx[k], j + dy[k]) != 0) {

							flag = 1;
							break;
						}

					}
					if (flag == 0)
						dst.at<uchar>(i, j) = 0;

				}



			}

		src = dst.clone();
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {

				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 4; k++) {
						dst.at<uchar>(i + dx[k], j + dy[k]) = 0;
					}

				}

			}


		imshow("input image", src);
		imshow("agiutor", dst);

		waitKey();
	}


}

void boundaryExtraction() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat showy = src.clone();
		Mat dst = src.clone();
		dst.setTo(255);

		int dx[4] = { -1,1,0,0 };
		int dy[4] = { 0,0,1,-1 };

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {

				///Vec3b p = src.at<Vec3b>(i, j);
				if (src.at<uchar>(i, j) == 0) {
					bool flag = 0;

					for (int k = 0; k < 4; k++)
					{
						if (src.at<uchar>(i + dx[k], j + dy[k]) != 0) {

							flag = 1;
							break;
						}

					}
					if (flag == 0)
						dst.at<uchar>(i, j) = 0;

				}


			}
			
		src = dst.clone();
		dst = showy.clone();
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++) {
				if (src.at<uchar>(i, j) == 0 && showy.at<uchar>(i, j) == 0)
			
					dst.at<uchar>(i,j) = 255;
			}


		imshow("input image", showy);
		imshow("agiutor", dst);

		waitKey();
	}
}

void saltyPeppers(int w)
{
	int dx[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dy[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				std::vector<uchar> v;
				for (int k = 0; k < 8; k++) {
					int newX = i + dx[k];
					int newY = j + dy[k];

					v.push_back(src.at<uchar>(newX, newY));
				}

				std::sort(v.begin(), v.end());

				dst.at<uchar>(i, j) = v[4];
			}
		}

		imshow("dst", dst);
		waitKey();
	}
}

void meanFilter3() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst(height, width);

		int H[3][3] = {
			{1,1,1},
			{1,1,1},
			{1,1,1}
		};

		dst = src.clone();

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {

				int value = 0;

				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						value += H[k][l] * src(i - 1 + k, j - 1 + l);
					}
				}

				value /= 9;
				dst.at<uchar>(i, j) = value;
			}
		}

		imshow("source image", src);
		imshow("destionation image", dst);
		waitKey();
	}
}

void meanFilter5() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst(height, width);

		int H[5][5] = {
			{1,1,1},
			{1,1,1},
			{1,1,1},
			{1,1,1},
			{1,1,1}
		};

		dst = src.clone();

		for (int i = 2; i < height - 2; i++) {
			for (int j = 2; j < width - 2; j++) {

				int value = 0;

				for (int k = 0; k < 5; k++) {
					for (int l = 0; l < 5; l++) {
						value += H[k][l] * src(i - 2 + k, j - 2 + l);
					}
				}

				value /= 25;
				dst.at<uchar>(i, j) = value;
			}
		}

		imshow("source image", src);
		imshow("destionation image", dst);
		waitKey();
	}
}

void laplace() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst(height, width);

		int H[3][3] = {
			{-1,-1,-1},
			{-1, 8,-1},
			{-1,-1,-1}
		};

		dst = src.clone();

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {

				int value = 0;

				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						value += H[k][l] * src(i - 1 + k, j - 1 + l);
					}
				}

				if (value >= 0 && value <= 255) {
					dst.at<uchar>(i, j) = value;
				}
				else
					if (value < 0) {
						dst.at<uchar>(i, j) = 0;
					}
					else
						if (value > 255) {
							dst.at<uchar>(i, j) = 255;
						}
			}
		}

		imshow("source image", src);
		imshow("destionation image", dst);
		waitKey();
	}
}


void gaussian(float sigma) {

	char fname[MAX_PATH];
	int w = ceil(6.0 * sigma);
	if (w % 2 == 0) {
		w++;
	}
	float mat[5][5];
	float sum = 0.0f;

	int x0 = w / 2;
	int y0 = w / 2;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			mat[i][j] = 1 / (2 * CV_PI * sigma * sigma) * exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * sigma * sigma));
			sum += mat[i][j];
		}
	}

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		//Mat dst(src.rows, src.cols, CV_8UC1);
		Mat dst = src.clone();
		double t = (double)getTickCount();

		for (int i = 2; i < src.rows - 2; i++) {
			for (int j = 2; j < src.cols - 2; j++) {
				int value = 0;
				for (int k = 0; k < 5; k++) {
					for (int l = 0; l < 5; l++) {
						value += mat[k][l] * src.at<uchar>(i - 2 + k, j - 2 + l);
					}
				}
				dst.at<uchar>(i, j) = value / sum;
			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f\n", t * 1000);

		imshow("dst", dst);
		waitKey();
	}
}

void highPass() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_ <uchar> src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_ <uchar> dst(height, width);

		int H[3][3] = {
			{0, -1, 0},
			{-1, 5, -1},
			{0, -1, 0}
		};

		dst = src.clone();

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {

				int value = 0;

				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						value += H[k][l] * src(i - 1 + k, j - 1 + l);
					}
				}

				if (value >= 0 && value <= 255) {
					dst.at<uchar>(i, j) = value;
				}
				else
					if (value < 0) {
						dst.at<uchar>(i, j) = 0;
					}
					else
						if (value > 255) {
							dst.at<uchar>(i, j) = 255;
						}
			}
		}


		imshow("source image", src);
		imshow("destionation image", dst);
		waitKey();
	}
}

void centering_transform(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 1; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

void fourierSpectrum() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat srcf;

		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, phi;
		magnitude(channels[0], channels[1], mag);


		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				mag.at<float>(i, j) = log(mag.at<float>(i, j)) + 1;
			}
		}

		Mat dst;
		normalize(mag, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("source", src);
		imshow("final image", dst);
		waitKey();
	}
}

void idealLPF() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat srcf;

		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, dst, dstf;
		magnitude(channels[0], channels[1], mag);
		float R = 10.0f;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float value = (i - height / 2) * (i - height / 2) + (j - width / 2) * (j - width / 2);
				if (value > R * R) {
					channels[0].at<float>(i, j) = 0.0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}

		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("source", src);
		imshow("final image", dst);
		waitKey();
	}

}

void idealHPF() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat srcf;

		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, dst, dstf;
		magnitude(channels[0], channels[1], mag);
		float R = 10.0f;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float value = (i - height / 2) * (i - height / 2) + (j - width / 2) * (j - width / 2);
				if (value <= R * R) {
					channels[0].at<float>(i, j) = 0.0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}

		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("source", src);
		imshow("final image", dst);
		waitKey();
	}

}

void gaussianCutLPF() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat srcf;

		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, dst, dstf;
		magnitude(channels[0], channels[1], mag);

		float A = 10.0f;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float coefficient = exp(-((i - height / 2) * (i - height / 2) + (j - width / 2) * (j - width / 2)) / (A * A));
				channels[0].at<float>(i, j) *= coefficient;
				channels[1].at<float>(i, j) *= coefficient;
			}
		}

		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);



		imshow("source", src);
		imshow("final image", dst);
		waitKey();

	}
}

void gaussianCutHPF() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat srcf;

		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, dst, dstf;
		magnitude(channels[0], channels[1], mag);

		float A = 10.0f;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				float coefficient = 1 - exp(-((i - height / 2) * (i - height / 2) + (j - width / 2) * (j - width / 2)) / (A * A));
				channels[0].at<float>(i, j) *= coefficient;
				channels[1].at<float>(i, j) *= coefficient;
			}
		}

		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centering_transform(dstf);
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);



		imshow("source", src);
		imshow("final image", dst);
		waitKey();
	}

}

void meanANDstandardDEV()
{
	int L = 255;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0.0 };

		double mean = 0;
		double stdDev = 0;

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = src.data[i * src.step[0] + j];

				h[val] += 1;
			}
		}

		float M = (float)(height * width);

		for (int i = 0; i < 256; i++) {
			p[i] = h[i] / M;
		}

		for (int g = 0; g <= L; g++)
		{
			mean += g * p[g];
		}

		for (int g = 0; g <= L; g++)
		{
			stdDev += (g - mean) * (g - mean) * p[g];
		}

		stdDev = sqrt(stdDev);

		printf("Mean: %f \nStandard Dev: %f\n", mean, stdDev);
		getchar();
		getchar();
	}

}

void regionFilling()
/*
		Mat_<uchar> inverse(src.rows, src.cols);
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src(i, j) == 0) {
					inverse(i, j) = 255;f

				}
				else inverse(i, j) = 0;
			}
		}

	}
	*/
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat srcC = Mat(src.rows, src.cols, CV_8UC1);
		Mat lastIter = Mat(src.rows, src.cols, CV_8UC1);
		Mat iter = Mat(src.rows, src.cols, CV_8UC1);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		int height = src.rows;
		int width = src.cols;

		int Bi[8] = { 0,0,1,-1 };
		int Bj[8] = { 1,-1,0,0 };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				src.at<uchar>(i, j) == 0 ? srcC.at<uchar>(i, j) = 255 : srcC.at<uchar>(i, j) = 0;
				lastIter.at<uchar>(i, j) = 255;
				iter.at<uchar>(i, j) = 255;
				dst.at<uchar>(i, j) = 255;

			}
		}

		lastIter.at<uchar>(height / 2, width / 2) = 0;

		while (true)
		{
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					iter.at<uchar>(i, j) = 255;
				}
			}
			int equalIters = 1;
			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (lastIter.at<uchar>(i, j) == 0) {

						iter.at<uchar>(i, j) = 0;

						for (int k = 0; k < 4; k++)
						{
							iter.at<uchar>(i + Bi[k], j + Bj[k]) = 0;
						}

					}

				}
			}

			for (int i = 1; i < height - 1; i++)
			{
				for (int j = 1; j < width - 1; j++)
				{
					if (srcC.at<uchar>(i, j) == 0 && iter.at<uchar>(i, j) == 0)
					{
						iter.at<uchar>(i, j) = 0;
					}
					else
					{
						iter.at<uchar>(i, j) = 255;
					}
				}
			}

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (lastIter.at<uchar>(i, j) != iter.at<uchar>(i, j))
					{
						equalIters = 0;
					}
				}
			}

			if (equalIters)
				break;

			iter.copyTo(lastIter);

		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 || iter.at<uchar>(i, j) == 0)
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}


		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);


		imshow("input image", src);
		imshow("complement of input image", srcC);
		imshow("filling image", iter);
		imshow("output image", dst);
		waitKey(0);
	}
}


void geometricalFeaturesComputation(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;

	if (event == EVENT_LBUTTONDBLCLK)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		Vec3b color = src->at<Vec3b>(y, x); 

		Mat dstHoriz = Mat(src->rows, src->cols, CV_8UC3);
		Mat dstVert = Mat(src->rows, src->cols, CV_8UC3);

		int area = 0;
		float centerOfMassR = 0.0;
		float centerOfMassC = 0.0;
		float axisOfElongation = 0.0;

		int numberOfEdgePixels = 0;

		float thinnesRatio = 0.0;

		float aspectRatio = 0.0;

		int cmax = 0;
		int cmin = src->cols;
		int rmax = 0;
		int rmin = src->rows;

		int colHoriz = 0;
		int rowVert = 0;
		int colVert = 0;

		for (int i = 0; i < src->rows; i++)
		{
			for (int j = 0; j < src->cols; j++)
			{
				dstHoriz.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				dstVert.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}

		for (int i = 0; i < src->rows; i++)
		{
			colHoriz = 0;
			for (int j = 0; j < src->cols; j++)
			{

				if (src->at<Vec3b>(i, j) == color)
				{
					j > cmax ? cmax = j : cmax = cmax;
					j < cmin ? cmin = j : cmin = cmin;
					i > rmax ? rmax = i : rmax = rmax;
					i < rmin ? rmin = i : rmin = rmin;

					area += 1;
					centerOfMassR += i;
					centerOfMassC += j;

					if (src->at<Vec3b>(i, j) == color)
					{
						bool isEdge = false;
						if (src->at<Vec3b>(i - 1, j - 1) != color) isEdge = true;
						if (src->at<Vec3b>(i - 1, j) != color) isEdge = true;
						if (src->at<Vec3b>(i - 1, j + 1) != color) isEdge = true;
						if (src->at<Vec3b>(i, j - 1) != color) isEdge = true;
						if (src->at<Vec3b>(i, j + 1) != color) isEdge = true;
						if (src->at<Vec3b>(i + 1, j - 1) != color) isEdge = true;
						if (src->at<Vec3b>(i + 1, j) != color) isEdge = true;
						if (src->at<Vec3b>(i + 1, j + 1) != color) isEdge = true;

						if (isEdge)
						{
							numberOfEdgePixels += 1;
						}
					}

					dstHoriz.at<Vec3b>(i, colHoriz++) = color;

				}

			}
		}

		for (int j = 0; j < src->cols; j++)
		{
			rowVert = 0;
			for (int i = 0; i < src->rows; i++)
			{
				if (src->at<Vec3b>(i, j) == color)
				{
					dstVert.at<Vec3b>(rowVert++, j) = color;
				}
			}
		}

		int perimeter = (float)numberOfEdgePixels * (CV_PI / 4);

		centerOfMassR = (float)(centerOfMassR) / ((float)area);
		centerOfMassC = (float)(centerOfMassC) / ((float)area);

		int Y = 0;
		int X = 0;

		for (int i = 0; i < src->rows; i++)
		{
			for (int j = 0; j < src->cols; j++)
			{
				if (src->at<Vec3b>(i, j) == color)
				{
					Y += (i - (int)centerOfMassR) * (j - (int)centerOfMassC);
					X += (j - (int)centerOfMassC) * (j - (int)centerOfMassC) - (i - (int)centerOfMassR) * (i - (int)centerOfMassR);

				}
			}
		}

		Y *= 2;

		axisOfElongation = (float)(atan2(Y, X)) / 2.0;
		axisOfElongation < 0 ? axisOfElongation += CV_PI : axisOfElongation = axisOfElongation;
		float axisOfElongationRads = axisOfElongation;
		axisOfElongation *= 180.0 / CV_PI;

		thinnesRatio = 4 * CV_PI * ((float)(area) / (float)(((float)perimeter * (float)perimeter)));

		aspectRatio = (float)(((float)cmax - (float)cmin + 1.0) / ((float)rmax - (float)rmin + 1.0));

		int cA = cmin;
		int cB = cmax;
		int rA = (int)centerOfMassR + (tan(axisOfElongationRads)) * (cmin - centerOfMassC);
		int rB = (int)centerOfMassR + (tan(axisOfElongationRads)) * (cmax - centerOfMassC);

		printf("Center of mass: \n");
		printf("R bar: %d \n", (int)centerOfMassR);
		printf("C bar: %d \n", (int)centerOfMassC);
		printf("Area: %d \n", area);
		printf("Axis of elongation: %lf\n", axisOfElongation);
		printf("Perimeter: %d\n", perimeter);
		printf("Thinnes ratio: %lf\n", thinnesRatio);
		printf("Aspect ratio: %lf\n", aspectRatio);

		imshow("Horizontal projection", dstHoriz);
		imshow("Vertical projection", dstVert);

		line(*src, Point(cA, rA), Point(cB, rB), Scalar(0, 0, 0), 2);
		imshow("MyImage", *src);

	}
}

void geometricalFeatures()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", geometricalFeaturesComputation, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}


void BFS()
{
	char fname[MAX_PATH];

	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

	while (openFileDlg(fname))
	{

		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		Mat labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));

		int label = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;
					std::queue<Point> Q;
					labels.at<int>(i, j) = label;
					Q.push(Point(j, i));
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							int row = q.y + di[k];
							int col = q.x + dj[k];
							if (row >= 0 && row < height && col >= 0 && col < width)
								if (src.at<uchar>(row, col) == 0 && labels.at<int>(row, col) == 0) {
									labels.at<int>(row, col) = label;
									Q.push(Point(col, row));
								}
						}
					}
				}
			}
		}

		srand(time(NULL));

		std::vector<Vec3b> colors;

		for (int i = 0; i < label; i++) {
			colors.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<int>(i, j) != 0) {
					dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j) - 1];
				}
				else {
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

		imshow("Source", src);
		imshow("Destination", dst);
		waitKey();

	}
}

void twoPass()
{
	int di[4] = { -1,-1,-1,0 };
	int dj[4] = { 1,0,-1,-1 };
	std::queue<int> Q;

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		std::vector<std::vector<int>> edges;
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		Mat labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));

		int label = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					std::vector<int> L;
					for (int k = 0; k < 4; k++) {
						int newX = j + dj[k];
						int newY = i + di[k];
						if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
							if (labels.at<int>(newY, newX) > 0) {
								L.push_back(labels.at<int>(newY, newX));
							}
						}
					}

					if (L.empty()) {
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					}
					else {
						int x = *min_element(L.begin(), L.end());
						labels.at<int>(i, j) = x;
						for (int k = 0; k < L.size(); k++) {
							int y = L[k];
							if (x != y) {
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}
				}
			}
		}

		int newLabel = 0;
		std::vector<int> newLabels(label + 1, 0);
		for (int i = 1; i <= label; i++) {
			if (newLabels[i] == 0) {
				newLabel++;
				newLabels[i] = newLabel;
				Q.push(i);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int k = 0; k < edges[x].size(); k++) {
						int y = edges[x][k];
						if (newLabels[y] == 0) {
							newLabels[y] = newLabel;
							Q.push(y);
						}
					}

				}
			}
		}

		srand(time(NULL));

		std::vector<Vec3b> colors;

		for (int i = 0; i < newLabel; i++) {
			colors.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labels.at<int>(i, j) != 0) {
					labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
					dst.at<Vec3b>(i, j) = colors[labels.at<int>(i, j) - 1];
				}
				else {
					dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

		imshow("Source", src);
		imshow("Destination", dst);
		waitKey();

	}
}


void globThresh()
{
	int L = 255;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		float p[256] = { 0.0 };

		double mean = 0;
		double stdDev = 0;

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);


		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = src.data[i * src.step[0] + j];

				h[val] += 1;
			}
		}

		float M = (float)(height * width);

		int imin = 0;
		int imax = 255;

		while (h[imin] == 0)
		{
			imin++;
		}

		while (h[imax] == 0)
		{
			imax--;
		}

		double Tant;
		double Tcurr = (imin + imax) / 2.0;
		double eps = 0.1;

		do
		{
			Tant = Tcurr;
			double meanG1 = 0, meanG2 = 0;
			double nrG1 = 0;
			for (int g = imin; g <= Tant; g++)
			{
				meanG1 += g * h[g];
				nrG1 += h[g];
			}
			meanG1 /= nrG1;

			double nrG2 = 0;
			for (int g = Tant + 1; g <= imax; g++)
			{
				meanG2 += g * h[g];
				nrG2 += h[g];
			}
			meanG2 /= nrG2;

			Tcurr = (meanG1 + meanG2) / 2;

		} while (abs(Tcurr - Tant) >= eps);

		printf("Threshold: %f\n", Tcurr);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < Tcurr)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("source", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

void brightnessChange()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int offset = -50;

		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) + offset > 255)
				{
					dst.at<uchar>(i, j) = 255;
				}
				else if (src.at<uchar>(i, j) + offset < 0)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = src.at<uchar>(i, j) + offset;
				}
			}
		}

		imshow("source", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

void contrastChange()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int gOutMin = 0;
		int gOutMax = 255;

		int h[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = src.data[i * src.step[0] + j];

				h[val] += 1;
			}
		}

		int gInMin = 0;
		int gInMax = 255;

		while (h[gInMin] == 0)
		{
			gInMin++;
		}

		while (h[gInMax] == 0)
		{
			gInMax--;
		}

		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = gOutMin + (src.at<uchar>(i, j) - gInMin) * ((double)(gOutMax - gOutMin) / (gInMax - gInMin));
			}
		}

		imshow("source", src);
		showHistogram("Histograma source", h, 256, 500);
		int hd[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = dst.data[i * dst.step[0] + j];

				hd[val] += 1;
			}
		}
		imshow("dst", dst);
		showHistogram("Histograma destination", hd, 256, 500);
		waitKey(0);
	}
}

void gammaCorrection()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		double gamma = 2;
		int L = 255;

		int h[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = src.data[i * src.step[0] + j];

				h[val] += 1;
			}
		}


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = L * pow((double)src.at<uchar>(i, j) / L, gamma);
			}
		}

		imshow("source", src);
		showHistogram("Histograma source", h, 256, 500);
		int hd[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = dst.data[i * dst.step[0] + j];

				hd[val] += 1;
			}
		}
		imshow("dst", dst);
		showHistogram("Histograma destination", hd, 256, 500);
		waitKey(0);
	}
}

void histogramEqualization()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int L = 255;

		int h[256] = { 0 };
		double p[256] = { 0.0 };
		double pc[256] = { 0.0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = src.data[i * src.step[0] + j];

				h[val] += 1;
			}
		}

		float M = (float)(height * width);

		for (int i = 0; i < 256; i++) {
			p[i] = h[i] / M;
		}

		pc[0] = p[0];

		for (int g = 1; g < 256; g++) {
			pc[g] = pc[g - 1] + p[g];
		}

		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = L * pc[src.at<uchar>(i, j)];
			}
		}

		imshow("source", src);
		showHistogram("Histograma source", h, 256, 500);
		int hd[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int val = dst.data[i * dst.step[0] + j];

				hd[val] += 1;
			}
		}
		imshow("dst", dst);
		showHistogram("Histograma destination", hd, 256, 500);
		waitKey(0);
	}
}

/*	char fname[MAX_PATH];
	int Sx[3][3] = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1},
	};
	int Sy[3][3] = {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1},
	};
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w;
		float sigma = 0.5;
		printf("%f", sigma * 6);
		w = sigma * 6;
		if (w % 2 == 0)
			w++;

		Mat dstG = Mat(height, width, CV_8UC1);
		double conv[20][20];
		double total = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
				int xy0 = w / 2;
				double z = exp(-((i - xy0) * (i - xy0) + (j - xy0) * (j - xy0)) / (2 * sigma * sigma));
				double zz = 1 / (2 * CV_PI * sigma * sigma);
				conv[i][j] = z * zz;
				total += conv[i][j];
			}
		}
		for (int i = w / 2; i < height - (w / 2); i++) {
			for (int j = w / 2; j < width - (w / 2); j++) {
				int stl = -w / 2;
				double sum = 0;
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						sum += src.at<uchar>(i + stl + k, j + stl + l) * conv[k][l];
					}
				}
				sum = sum / total;
				dstG.at<uchar>(i, j) = sum;
			}
		}

		Mat dstGr = Mat(height, width, CV_32F);
		Mat dstOr = Mat(height, width, CV_32F);
		Mat dstGrDisplay = Mat(height, width, CV_8UC1);
		w = 3;
		for (int i = w / 2; i < height - (w / 2); i++) {
			for (int j = w / 2; j < width - (w / 2); j++) {
				int stl = -w / 2;
				int sumX = 0;
				int sumY = 0;

				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						sumX += dstG.at<uchar>(i + stl + k, j + stl + l) * Sx[k][l];
					}
				}
				for (int k = 0; k < w; k++) {
					for (int l = 0; l < w; l++) {
						sumY += dstG.at<uchar>(i + stl + k, j + stl + l) * Sy[k][l];
					}
				}
				int sum3 = sumX * sumX + sumY * sumY;
				float g = sqrt(sum3) / (4 * sqrt(2));
				float or = atan2(sumY, sumX);
				dstGr.at<float>(i, j) = g;
				dstOr.at<float>(i, j) = or ;
				dstGrDisplay.at<uchar>(i, j) = (uchar)g;


			}
		}

		imshow("gradient mag", dstGrDisplay);



		imshow("original", src);
		imshow("filtered", dstG);




	}*/
int find_partition(float nr) {

	float pi = CV_PI;

	if ((nr >= pi / 8 && nr <= 3 * pi / 8) || (nr >= -7 * pi / 8 && nr <= -5 * pi / 8)) {
		return 1;
	}
	else if ((nr >= 3 * pi / 8 && nr <= 5 * pi / 8) || (nr >= -5 * pi / 8 && nr <= -3 * pi / 8)) {
		return 2;
	}
	else if ((nr >= 5 * pi / 8 && nr <= 7 * pi / 8) || (nr >= -3 * pi / 8 && nr <= -pi / 8)) {
		return 3;
	}
	else return 0;
}
void cannyEdgeDetection()
{
	// 1) Gaussian noise filterning
	float sigma = 0.5;
	int w = 6 * sigma;
	if (w % 2 == 0)
		w++;

	float sumaG = 0;
	float sumaG1 = 0;
	char fname[MAX_PATH];
	double x0 = w / 2;
	double y0 = w / 2;
	double g1[20];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		double g[20][20] = {};
		Mat dst(height, width, CV_8UC1);
		Mat dstfinal(height, width, CV_8UC1);

		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {


				g[i][j] = (1 / (2 * PI * sigma * sigma)) * exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / (2 * sigma * sigma));
				sumaG = sumaG + g[i][j];


			}
		}
		for (int i = 0; i < w; i++)
		{
			g1[i] = g[i][w / 2];
			sumaG1 = sumaG1 + g1[i];
		}

		dst = src.clone();
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float s1 = 0;
				for (int l = 0; l < w; l++) {

					s1 += src.at<uchar>(i - w / 2 + l, j) * g1[l];

				}
				dst.at<uchar>(i, j) = s1 / sumaG1;
			}
		}

		dstfinal = dst.clone();
		for (int i = w / 2; i < height - w / 2; i++) {
			for (int j = w / 2; j < width - w / 2; j++) {
				float s2 = 0;
				for (int l = 0; l < w; l++) {
					s2 += dst.at<uchar>(i, j - w / 2 + l) * g1[l];
				}
				dstfinal.at<uchar>(i, j) = s2 / sumaG1;
			}
		}

		// 2) Gradient magnitude and orientation
		int Sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
		int Sy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
		Mat dst1(height, width, CV_8UC1);
		Mat_<float> dst2(height, width);
		float G;
		float phi;
		Mat_<float> dst3(height, width);

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {

				int valX = 0;
				int valY = 0;

				for (int k = 0; k < 3; k++) {
					for (int p = 0; p < 3; p++) {

						valX += dstfinal.at<uchar>(i - 3 / 2 + k, j - 3 / 2 + p) * Sx[k][p];
						valY += dstfinal.at<uchar>(i - 3 / 2 + k, j - 3 / 2 + p) * Sy[k][p];
					}
				}

				G = sqrt(valX * valX + valY * valY);
				G /= 4 * sqrt(2);
				phi = atan2(valY, valX);

				//store
				dst1.at<uchar>(i, j) = (int)G;
				dst2(i, j) = (float)G;		//normalized
				dst3(i, j) = (float)phi;	//orientation
			}

		}

		// 3) Non-maxima supression
		Mat GS(height, width, CV_8UC1);

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {

				int partition = find_partition(dst3.at<float>(i, j));
				float pixel = dst2.at<float>(i, j);
				int x1 = 0;
				int x2 = 0;
				int y1 = 0;
				int y2 = 0;

				switch (partition) {
				case 0:
					x1 = i;
					x2 = i;
					y1 = j - 1;
					y2 = j + 1;
					break;

				case 1:
					x1 = i + 1;
					x2 = i - 1;
					y1 = j - 1;
					y2 = j + 1;
					break;

				case 2:
					x1 = i + 1;
					x2 = i - 1;
					y1 = j;
					y2 = j;
					break;

				case 3:
					x1 = i + 1;
					x2 = i - 1;
					y1 = j + 1;
					y2 = j - 1;
					break;
				}
				float neigh1 = dst2.at<float>(x1, y1);
				float neigh2 = dst2.at<float>(x2, y2);


				if (!(pixel >= neigh1 && pixel >= neigh2)) {
					GS.at<uchar>(i, j) = 0;	//supressed
				}
				else
				{
					GS.at<uchar>(i, j) = dst2(i, j);
				}
			}
		}

		int h[256] = { 0 };
		float p = 0.1;

		for (int i = 1; i < height; i++)
		{
			for (int j = 1; j < width; j++)
			{
				int val = GS.data[i * GS.step[0] + j];
				h[val] += 1;
			}
		}

		float noEdgePixel = p * ((height - 2) * (width - 2) - h[0]);
		float k = 0.4;
		float tresholdMax = 0;
		float tresholdMin = 0;
		int s = 0;
		for (int i = 255; i >= 0; i--)
		{
			if (noEdgePixel > s) {
				s += h[i];
				tresholdMax = i + 1;
			}
		}
		tresholdMin = k * tresholdMax;

		for (int i = 1; i < height; i++)
		{
			for (int j = 1; j < width; j++) \
			{
				if (GS.at<uchar>(i, j) > tresholdMax)
					GS.at<uchar>(i, j) = 255;
				else if (GS.at<uchar>(i, j) <= tresholdMax && GS.at<uchar>(i, j) >= tresholdMin)
					GS.at<uchar>(i, j) = 127;
				else if (GS.at<uchar>(i, j) < tresholdMin)
					GS.at<uchar>(i, j) = 0;
			}
		}

		int di[8] = { 0,-1,-1,-1,0,1,1,1 };
		int dj[8] = { 1,1,0,-1,-1,-1,0,1 };

		std::queue<Point> Q;

		for (int i = w / 2; i < height - w / 2; i++)
		{
			for (int j = w / 2; j < width - w / 2; j++)
			{
				if (GS.at<uchar>(i, j) == 255)
				{
					Point p = Point(j, i);
					Q.push(p);
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							Point newP = Point(q.x + dj[k], q.y + di[k]);
							if (GS.at<uchar>(newP) == 127) {
								GS.at<uchar>(newP) = 255;
								Q.push(newP);
							}
						}
					}
				}
			}
		}

		imshow("treshold2", GS);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar g = GS.at<uchar>(i, j);
				if (g == 255) {
					GS.at<uchar>(i, j) = 255;
				}
				else
				{
					GS.at<uchar>(i, j) = 0;
				}
			}
		}

		printf("treshold max: %f\ntreshold min: %f", tresholdMax, tresholdMin);
		imshow("supression", GS);
		waitKey(0);

	}
}


void cannyColor() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		Mat hsv;
		cvtColor(src, hsv, COLOR_BGR2HSV);

		std::vector<cv::Mat> channels;
		cv::split(hsv, channels);

		cv::Mat H = channels[0];
		cv::Mat S = channels[1];
		cv::Mat V = channels[2];

		cv::Mat shiftedH = H.clone();
		int shift = 25; // in openCV hue values go from 0 to 180 (so have to be doubled to get to 0 .. 360) because of byte range from 0 to 255
		for (int j = 0; j < shiftedH.rows; ++j)
			for (int i = 0; i < shiftedH.cols; ++i)
			{
				shiftedH.at<unsigned char>(j, i) = (shiftedH.at<unsigned char>(j, i) + shift) % 180;
			}
		Mat cannyH;
		Canny(shiftedH, cannyH, 100, 50);
		Mat cannyS;
		Canny(S, cannyS, 200, 100);
		// extract contours of the canny image:
		std::vector<std::vector<cv::Point> > contoursH;
		std::vector<cv::Vec4i> hierarchyH;
		findContours(cannyH, contoursH, hierarchyH, RETR_TREE, CHAIN_APPROX_SIMPLE);

		// draw the contours to a copy of the input image:
		Mat outputH = src.clone();
		for (int i = 0; i < contoursH.size(); i++)
		{
			cv::drawContours(outputH, contoursH, i, cv::Scalar(0, 0, 255), 2, 8, hierarchyH, 0);
		}
		dilate(cannyH, cannyH, cv::Mat());
		dilate(cannyH, cannyH, cv::Mat());
		dilate(cannyH, cannyH, cv::Mat());
		outputH = src.clone();
		for (int i = 0; i < contoursH.size(); i++)
		{
			if (cv::contourArea(contoursH[i]) < 20) continue; // ignore contours that are too small to be a patty
			if (hierarchyH[i][3] < 0) continue;  // ignore "outer" contours

			drawContours(outputH, contoursH, i, cv::Scalar(0, 0, 255), 2, 8, hierarchyH, 0);
			
		}
		imshow("out",outputH);
		waitKey();
	
	}
}


		

	int main()
	{
		cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
		projectPath = _wgetcwd(0, 0);
		Mat_<uchar> img = imread("files_border_tracing/star.bmp", IMREAD_GRAYSCALE);

		int op;
		do
		{
			system("cls");
			destroyAllWindows();
			printf("Menu:\n");
			printf(" 1 - Open image\n");
			printf(" 2 - Open BMP images from folder\n");
			printf(" 3 - Current Task\n");
			printf(" 4 - Image negative (fast)\n");
			printf(" 5 - BGR->Gray\n");
			printf(" 6 - BGR->Gray (fast, save result to disk) \n");
			printf(" 7 - BGR->HSV\n");
			printf(" 8 - Resize image\n");
			printf(" 9 - Canny edge detection\n");
			printf(" 10 - Edges in a video sequence\n");
			printf(" 11 - Snap frame from live video\n");
			printf(" 12 - Mouse callback demo\n");
			printf(" 0 - Exit\n\n");
			printf("Option: ");
			scanf("%d", &op);
			switch (op)
			{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				//testNegativeImage();
				//addGrey();
				//colorImage();
				///inverseMatrix();
				///RGBChannel();
				//colorToGray();
				///greyscaleToBW();
				//myHist();
				///myMultiLevelConspiracy();
				///myFloyd_Steinberg();
				///borderTracing(img);
				//contourColouring();
				//dilation();
				///dilationNtimes();
				///notDilation();
				//notDilationNtimes();
				///closing();
				//open();
				//boundaryExtraction();
				///gaussian(0.8);
				//cannyEdgeDetection();
				//geometricalFeatures();
				///globThresh();
				///meanANDstandardDEV();
				///contrastChange();
				///histogramEqualization();
				///meanFilter3();
				cannyColor();
				
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				///testBGR2HSV();
				myHSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			}
		} while (op != 0);
		return 0;
	}
