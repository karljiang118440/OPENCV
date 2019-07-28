
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
//OpenCV include
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ml.h>
 
using namespace std;
using namespace cv;
using namespace cv::ml;
static const Size trainingPadding = Size(0, 0);
//static const Size winStride = Size(8, 8);
static const Size winStride = Size(64, 64);
 
 
bool read_num_class_data(Mat* _data, Mat* _responses,int num){
	int icount=0;
	_data->release();
    _responses->release();
    vector<int >train_label;
    vector< vector<float > > train_data;
	for (int count = 0; icount < num ; ++count)
	//while (count<100)
	{
		//capture>>im;
		 char str[100];
		 sprintf(str,"/home/cx/Videos/data/vehicles/KITTI_extracted/%d.png",count);
		Mat im = imread(str,1);
		if(im.empty()){
			//cout <<"loading image failed"<< endl; 
			continue;
			//return 0;
		}
		icount++;
		Mat luvImage;
 
		Mat imageData;
    	cvtColor(im, imageData, CV_RGB2GRAY);		
 
		//hog
		HOGDescriptor hog;
		//HOGDescriptor hog(cvSize(64,64),cvSize(64,64),cvSize(64,64),cvSize(64,64),9); // Use standard parameters here
    	hog.winSize = Size(64, 64); // Default training images size as used in paper
 
    	//calculateFeaturesFromInput(currentImageFile, featureVector, hog);
    	vector<float> featureVector;
	    // Check for mismatching dimensions
	    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
	        featureVector.clear();
	        //printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
	        return 1;
	    }
	    //hog detect
	    vector<Point> locations;
	    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	    
 
 
	 //    Mat edge;
	 //    Canny(imageData,edge,70,200,3);
	 //    for (int i=0;i<im.rows;i++){
		// 	for (int j=0;j<im.cols;j++){
		// 		File_vector<<edge.at<uchar>(j,i)/255.0 <<" ";
		// 	}
		// }
	    //Sobel(imageData,edge,imageData.depth(),1,1);
 
	 //    cvtColor(im, luvImage, CV_RGB2Luv); 
		// vector<Mat> mv;  
		// split(im,mv);
		// fstream f("test");
		// f.open("test.dat", ios::out);
		// f<<mv[2]<<endl;
		// for (int k=0;k<3;k++){
		// 	for (int i=0;i<im.rows;i++){
		// 		for (int j=0;j<im.cols;j++){
		// 			File_vector<<mv[k].at<uchar>(j,i)/255.0 <<" ";
		// 		}
		// 	}
		// }
 
	    //myluv
		// uchar *tmp = mat2uchar(im);
		// int dim0, dim1, dim2;
		// float *I = rgbConvert(tmp, im.rows, im.cols, im.channels(), dim0, dim1, dim2, "luv");
		// int nr=im.rows;
		// int nc=im.cols;
		// for (int i = 0; i < im.channels(); i++){
		// 	for (int j = 0; j <im.rows; j++){
		// 		for (int k = 0; k < im.cols; k++){
		// 			File_vector << I[i*nr*nc + j*nc + k] << " ";
		// 		}
		// 	}
		// }
		// wrFree(I);
		//File_vector<<endl;
 
		train_data.push_back(featureVector);
	    train_label.push_back(1);
	}
 
	icount=0;
	for (int count = 0; icount < num ; ++count)
	//while (count<100)
	{
		//capture>>im;
		 char str[100];
		 //sprintf(str,"/home/cx/Videos/data/vehicles/KITTI_extracted/%d.png",count);
		 //sprintf(str,"/home/cx/Desktop/Vehicle_test/pyramid_test/pic/%d.png",count);
		 sprintf(str,"/home/cx/Videos/data/non-vehicles/Extras/extra%d.png",count);
		 //sprintf(str,"/home/cx/Videos/data/vehicles/GTI_MiddleClose/image%04d.png",count);
		 //sprintf(str,"/home/cx/Videos/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/%010d.png",count);
		Mat im = imread(str,1);
		if(im.empty()){
			//cout <<"loading image failed"<< endl; 
			continue;
			//return 0;
		}
		icount++;
		Mat luvImage;
 
		Mat imageData;
    	cvtColor(im, imageData, CV_RGB2GRAY);		
 
		//hog
		HOGDescriptor hog;
		//HOGDescriptor hog(cvSize(64,64),cvSize(64,64),cvSize(64,64),cvSize(64,64),9); // Use standard parameters here
    	hog.winSize = Size(64, 64); // Default training images size as used in paper
 
    	//calculateFeaturesFromInput(currentImageFile, featureVector, hog);
    	vector<float> featureVector;
	    // Check for mismatching dimensions
	    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
	        featureVector.clear();
	        //printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
	        return 1;
	    }
	    //hog detect
	    vector<Point> locations;
	    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	    
	    
	 //    Mat edge;
	 //    Canny(imageData,edge,70,200,3);
	 //    for (int i=0;i<im.rows;i++){
		// 	for (int j=0;j<im.cols;j++){
		// 		File_vector<<edge.at<uchar>(j,i)/255.0 <<" ";
		// 	}
		// }
	    //Sobel(imageData,edge,imageData.depth(),1,1);
 
	 //    cvtColor(im, luvImage, CV_RGB2Luv); 
		// vector<Mat> mv;  
		// split(im,mv);
		// fstream f("test");
		// f.open("test.dat", ios::out);
		// f<<mv[2]<<endl;
		// for (int k=0;k<3;k++){
		// 	for (int i=0;i<im.rows;i++){
		// 		for (int j=0;j<im.cols;j++){
		// 			File_vector<<mv[k].at<uchar>(j,i)/255.0 <<" ";
		// 		}
		// 	}
		// }
 
	    //myluv
		// uchar *tmp = mat2uchar(im);
		// int dim0, dim1, dim2;
		// float *I = rgbConvert(tmp, im.rows, im.cols, im.channels(), dim0, dim1, dim2, "luv");
		// int nr=im.rows;
		// int nc=im.cols;
		// for (int i = 0; i < im.channels(); i++){
		// 	for (int j = 0; j <im.rows; j++){
		// 		for (int k = 0; k < im.cols; k++){
		// 			File_vector << I[i*nr*nc + j*nc + k] << " ";
		// 		}
		// 	}
		// }
		// wrFree(I);
		//File_vector<<endl;
 
 
		train_data.push_back(featureVector);
	    train_label.push_back(0);
	}
 
	srand((unsigned)time(0));
	for (int i=0;i<num*2;i++){
		int a=rand()%(num*2);
		int b=rand()%(num*2);
		swap(train_label[a],train_label[b]);
		swap(train_data[a],train_data[b]);
	}
 
	Mat(train_label).copyTo(*_responses);
 
	*_data=Mat::zeros(train_data.size(), train_data[0].size(), CV_32F);
	for (int i=0;i<train_data.size();i++){
		for (int j=0;j<train_data[0].size();j++){
			_data->at<float>(i,j)=train_data[i][j];
			//cout<<_data->at<float>(i,j)<<" "<<train_data[i][j]<<endl;
		}
	}
}
 
int main(int argc, char** argv)
{
	
	string pic;
	int count = 0;
    	int num = atoi(argv[1]); //the num of picture
 
	// VideoCapture capture(input_video_file);
 //    if ( !capture.isOpened() ) {
 //        cout << "video could not be opened" << endl;
 //        return -1;
 //    }
 
 //    double num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    Mat im;
 
    fstream File_vector;
    fstream File_label;
 
 
    clock_t start, end;
	start = clock();
		
	//vector<vector<float> > train_data;
	//vector<int > train_label;
	Mat train_data;
    Mat train_label;
    read_num_class_data(&train_data,&train_label,num);
 
	
 
	Ptr<SVM> svm = SVM::create(); 
	svm->setType(SVM::C_SVC);    //设置svm类型
    //svm->setKernel(SVM::POLY); //设置核函数;
	svm->setKernel(SVM::LINEAR);
	// svm->setDegree(0.5);
	// svm->setGamma(1);
	// svm->setCoef0(1);
	// svm->setNu(0.5);
	// svm->setP(0);
	// svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
	svm->setC(1);
	Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
	svm->train(tData);
 
	Mat sample1 = train_data.row(0);
	//cout<<sample1.rows<<endl;
	int icount=0;
	int ans=0;
	for (int count = 0; icount < num ; ++count){
		//capture>>im;
		 char str[100];
		 //sprintf(str,"/home/cx/Videos/data/vehicles/KITTI_extracted/%d.png",count+1000);
		 //sprintf(str,"/home/cx/Desktop/Vehicle_test/pyramid_test/pic/%d.png",count);
		 sprintf(str,"/home/cx/Videos/data/non-vehicles/Extras/extra%d.png",count+1000);
		 //sprintf(str,"/home/cx/Videos/data/vehicles/GTI_MiddleClose/image%04d.png",count);
		 //sprintf(str,"/home/cx/Videos/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/%010d.png",count);
		Mat im = imread(str,1);
		if(im.empty()){
			continue;
		}
		icount++;
		Mat luvImage;
 
		Mat imageData;
    	cvtColor(im, imageData, CV_RGB2GRAY);		
 
		//hog
		HOGDescriptor hog;
		//HOGDescriptor hog(cvSize(64,64),cvSize(64,64),cvSize(64,64),cvSize(64,64),9); // Use standard parameters here
    	hog.winSize = Size(64, 64); // Default training images size as used in paper
 
    	//calculateFeaturesFromInput(currentImageFile, featureVector, hog);
    	vector<float> featureVector;
	    // Check for mismatching dimensions
	    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
	        featureVector.clear();
	        //printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
	        return 1;
	    }
	    //hog detect
	    vector<Point> locations;
	    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	    //Mat sample(featureVector);
	    //cout<<sample.rows<<endl;
	    float r = svm->predict(featureVector);
        if (r==0) ans++;
	}
 
	cout<<ans<<endl;
 
	end = clock();
	printf("time=%f\n", (double)(end - start) / CLOCKS_PER_SEC);
 
	//release_data(det25);
	return 0;
}
