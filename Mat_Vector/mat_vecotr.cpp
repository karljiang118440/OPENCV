#include <opencv2/opencv.hpp>
 
using namespace cv;
using namespace std;
 
/***************** Matתvector **********************/
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));//ͨ�������䣬����תΪһ��
}
 
/****************** vectorתMat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v);//��vector��ɵ��е�mat
	cv::Mat dest = mat.reshape(channels, rows).clone();//PS������clone()һ�ݣ����򷵻س���
	return dest;
}
 
 
int main()
{
	/* char ->CV_8SC
	* unsigned char,uchar ->CV_8UC
	* unsigned short int,ushort->CV_16UC
	* short int->CV_16SC
	* int   ->CV_32SC
	* float ->CV_32FC
	* double->CV_64FC
	*/
	uchar arr[4][3] = { { 1, 1,1 },{ 2, 2,2 },{ 3, 3,3 },{ 4,4, 4 } };
	cv::Mat srcData(4, 3, CV_8UC1, arr);
	cout << "srcData=\n" << srcData << endl;
 
 
	vector<uchar> v = convertMat2Vector<uchar>(srcData);
	cv::Mat dest = convertVector2Mat<uchar>(v, 1, 4);//������תΪ1ͨ����4�е�Mat����
	cout << "dest=\n" << dest << endl;
 
	system("pause");
	waitKey();
	return 0;
}