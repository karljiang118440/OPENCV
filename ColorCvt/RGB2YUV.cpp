#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


#include <fstream>
#include <iostream>
#include <string>

#define TUNE(r) ( r < 0 ? 0 : (r > 255 ? 255 : r) )

static int RGB_Y[256];

static int RGBR_V[256];

static int RGBG_U[256];

static int RGBG_V[256];

static int RGBB_U[256];

static int YUVY_R[256];

static int YUVY_G[256];

static int YUVY_B[256];

static int YUVU_R[256];

static int YUVU_G[256];

static int YUVU_B[256];

static int YUVV_R[256];

static int YUVV_G[256];

static int YUVV_B[256];

static int coff_rv[256];

static int coff_gu[256];

static int coff_gv[256];

static int coff_bu[256];


void RGB2YUV422(unsigned char*pRGB, unsigned char*pYUV, int size);

int main()
{
	//读取图片（使用图片的绝对路径）
	Mat src = imread("D://opencv//Projects//openc4.1_test//test.jpg", 1);



	int H = src.cols;
	int W = src.rows;
	int S = 8;

	Mat out = cv::Mat(H, W, CV_8UC2);

	//unsigned char *UYVY_TEST_DATA[1280*2*720];
	unsigned char *UYVY_TEST_DATA[1280];
	FILE *lpSavefile;
	lpSavefile = fopen("UYVY-TEST", "r");


	 char *pbuf;
	pbuf = (char *)malloc(sizeof(char) * 1280 * 720 * 2);

	fread(pbuf, 1280 * 720 * 2, 1, lpSavefile);

	//printf("%s", pbuf);

	free(pbuf);




	fread(UYVY_TEST_DATA, 1280 , 1, lpSavefile);

	//printf("%s", UYVY_TEST_DATA);

	unsigned char* inputImage = (unsigned char*)(src.data);
	unsigned char* outputImage = (unsigned char*)(out.data);




	unsigned char* outputImage1;

	RGB2YUV422(inputImage, outputImage1, 1280 * 720 * 2);



	waitKey(0);

	return 0;
}


void RGB2YUV422(unsigned char*pRGB, unsigned char*pYUV, int size)

{

	unsigned char r, g, b, u, v, u1, v1, r1, g1, b1;

	//unsigned char *YUVBuff;

	//unsigned char* p;

	//p = YUVBuff;//

	int loop = size / 2;

	int i;

	for (i = 0; i < loop; i++)

	{

		r = *pRGB; pRGB++;

		g = *pRGB; pRGB++;

		b = *pRGB; pRGB++;

		r1 = *pRGB; pRGB++;

		g1 = *pRGB; pRGB++;

		b1 = *pRGB; pRGB++;



		//new method --- right

		int y = ((YUVY_R[r] + YUVY_G[g] + YUVY_B[b] + 128) >> 8) + 16;

		u = ((YUVU_R[r] + YUVU_G[g] + YUVU_B[b] + 128) >> 8) + 128;

		v = ((YUVV_R[r] + YUVV_G[g] + YUVV_B[b] + 128) >> 8) + 128;

		int y1 = ((YUVY_R[r1] + YUVY_G[g1] + YUVY_B[b1] + 128) >> 8) + 16;

		u1 = ((YUVU_R[r1] + YUVU_G[g1] + YUVU_B[b1] + 128) >> 8) + 128;

		v1 = ((YUVV_R[r1] + YUVV_G[g1] + YUVV_B[b1] + 128) >> 8) + 128;



		*pYUV++ = TUNE(y);

		*pYUV++ = (TUNE(u) + TUNE(u1)) >> 1;

		*pYUV++ = TUNE(y);

		*pYUV++ = TUNE(y1);

		*pYUV++ = TUNE(v);

	}



}
