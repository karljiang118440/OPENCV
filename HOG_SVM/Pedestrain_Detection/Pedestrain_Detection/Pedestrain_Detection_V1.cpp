// pch.cpp: 与预编译标头对应的源文件；编译成功所必需的

#include "pch.h"

// 一般情况下，忽略此文件，但如果你使用的是预编译标头，请保留它。
#include "Tools.h"
#include "_Pedestrain.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

//vedio_dir |  J:\\Download\\SEQ\\set01\\V003.seq
const char* keys =
{
    "{help h|     | show help message}"
    "{pd    |  D:\\opencv\\Projects\\DATA\\INRIAPerson\\96X160H96\\Train\\pos   | path of directory contains possitive images}"
    "{nd    |  D:\\opencv\\Projects\\DATA\\INRIAPerson\\negphoto    | path of directory contains negative images}"
    "{td    |  D:\\opencv\\Projects\\DATA\\INRIAPerson\\Test\\pos   | path of directory contains test images}"
    "{tv    |    | test video file name}"
    "{dw    |  64   | width of the detector}"
    "{dh    |  128   | height of the detector}"
    "{d     |true| train twice}"
    "{t     |false| test a trained detector}"
    "{v     |false| visualize training steps}"
    "{fn    |D:\\opencv\\Projects\\HOG_SVM\\Pedestrain_Detection\\Pedestrain_Detection\\Pedestrain.xml| file name of trained SVM}"
};




string obj_det_filename = "D:\\opencv\\Projects\\HOG_SVM\\Pedestrain_Detection\\Pedestrain_Detection\\Pedestrain.xml";
string test_dir = "D:\\opencv\\Projects\\DATA\\INRIAPerson\\Test\\pos";
string vediofilename = "";

int main(int argc, char** argv)
{
    //数据预处理
    //Tools tool;
    //tool.ImgCut();
    //cout << tool.CropImageCount << endl;


    //训练并测试数据
    _Pedestrain pt;
    //pt.trainAndTest(argc, argv, keys);

    //测试数据
     pt.test_trained_detector(obj_det_filename, test_dir, vediofilename);
    return 0;
}
