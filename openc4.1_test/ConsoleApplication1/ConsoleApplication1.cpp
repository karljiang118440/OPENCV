// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

 int main()
{
	//读取图片（使用图片的绝对路径）
	Mat src = imread("test.jpg");
	//显示图片
	imshow("Output", src);
	//显示灰度图
	Mat Gray;
	cvtColor(src, Gray, 6);
	imshow("Gray", Gray);

	//不加此语句图片会一闪而过
	waitKey(0);

	return 0;
}
