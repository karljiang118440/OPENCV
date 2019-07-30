#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdlib.h>
#include<ctime>  //时间

using namespace std;
using namespace cv;
using namespace cv::ml;

#define PosSamNO 518  //正样本数量  519
#define NegSamNO 113000 // 负样本数量 113400
#define HardExampleNO 0 // 难例数量
#define AugPosSamNO 0 //未检测出的正样本数量

#define TRAIN 0  //若值为1，则开始训练

void train_SVM_HOG();
void SVM_HOG_detect();

int main(){

    if (TRAIN)
        train_SVM_HOG();

    SVM_HOG_detect();

    return 0;
}

void train_SVM_HOG()
{

    //                检测窗口(64,128),       块尺寸(16,16),     块步长(8,8),   cell尺寸(8,8), 直方图bin个数9   
    HOGDescriptor hog(Size(64, 128),        Size(16, 16),       Size(8, 8),     Size(8, 8),         9);
    int DescriptorDim; //HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  
    
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
//  svm->setC(0.01); //设置惩罚参数C，默认值为1
    svm->setKernel(SVM::LINEAR); //线性核
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6)); //3000是迭代次数，1e-6是精确度。
    //setTermCriteria是用来设置算法的终止条件， SVM训练的过程就是一个通过 迭代 方式解决约束条件下的二次优化问题，这里我们指定一个最大迭代次数和容许误

差，以允许算法在适当的条件下停止计算


    string ImgName;//图片的名字
    string PosSampleAdress = "D:\\盒装牛奶检测\\64_128__牛奶图片\\";
    string NegSampleAdress = "D:\\盒装牛奶检测\\负样本\\裁剪后负样本数据\\";
    string HardSampleAdress = "";

    ifstream finPos(PosSampleAdress + "PosSamAddressTxt.txt"); //正样本地址txt文件
    ifstream finNeg(NegSampleAdress + "NegSampleAdressTxt.txt");         //负样本地址txt文件

    if (!finPos){
        cout << "正样本txt文件读取失败" << endl;
        return;
    }
    if (!finNeg){
        cout << "负样本txt文件读取失败" << endl;
        return;
    }

    Mat sampleFeatureMat; // 所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数为HOG描述子维数  
    Mat sampleLabelMat;   // 所有训练样本的类别向量，行数等于所有样本的个数， 列数为1： 1表示有目标，-1表示无目标  

    //---------------------------逐个读取正样本图片，生成HOG描述子-------------
    for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++) //getline(finPos, ImgName) 从文件finPos中读取图像的名称ImgName
    {
        system("cls");
        cout << endl << "正样本处理: " << ImgName << endl;
        ImgName = PosSampleAdress + ImgName;
        Mat src = imread(ImgName);

        vector<float> descriptors; //浮点型vector（类似数组），用于存放HOG描述子
        hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

        if (0 == num) //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵 
        {
            DescriptorDim = descriptors.size(); //HOG描述子的维数   
            //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat  
            sampleFeatureMat = Mat::zeros(PosSamNO + AugPosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1); 
            //初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1   
            sampleLabelMat = Mat::zeros(PosSamNO + AugPosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号

整数型
        }

        
        for (int i = 0; i<DescriptorDim; i++)
            sampleFeatureMat.at<float>(num, i) = descriptors[i];

        sampleLabelMat.at<int>(num, 0) = 1;  //样本标签矩阵 值为1
    }

    if (AugPosSamNO > 0)
    {
        cout << endl << "处理在测试集中未被被检测到的样本: " << endl;
        ifstream finAug("DATA/AugPosImgList.txt");
        if (!finAug){
            cout << "Aug positive txt文件读取失败" << endl;
            return;
        }

        for (int num = 0; num < AugPosSamNO && getline(finAug, ImgName); ++num)
        {
            ImgName = "DATA/INRIAPerson/AugPos/" + ImgName;
            Mat src = imread(ImgName);
            vector<float> descriptors;
            hog.compute(src, descriptors, Size(8, 8));
            for (int i = 0; i < DescriptorDim; ++i)
                sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
            sampleLabelMat.at<int>(num + PosSamNO, 0) = 1;
        }
    }

    
    for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
    {
        system("cls");
        cout << "负样本图片处理: " << ImgName << endl;
        ImgName = NegSampleAdress + ImgName;
        Mat src = imread(ImgName);

        vector<float> descriptors;
        hog.compute(src, descriptors, Size(8, 8));

        for (int i = 0; i<DescriptorDim; i++)
            sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];

        sampleLabelMat.at<int>(num + PosSamNO + AugPosSamNO, 0) = -1;
    }


    if (HardExampleNO > 0)
    {
        ifstream finHardExample(HardSampleAdress+"HardSampleAdressTxt.txt");
        if (!finHardExample){
            cout << "难样本txt文件读取失败" << endl;
            return;
        }

        for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
        {
            system("cls");
            cout << endl << "处理难样本图片: " << ImgName << endl;
            ImgName = HardSampleAdress + ImgName;
            Mat src = imread(ImgName);

            vector<float> descriptors;
            hog.compute(src, descriptors, Size(8, 8));

            for (int i = 0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];
            sampleLabelMat.at<int>(num + PosSamNO + AugPosSamNO + NegSamNO, 0) = -1;
        }
    }

    cout << endl << "       开始训练..." << endl;
    svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
//  svm->trainAuto(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat,10);


    svm->save("SVM_HOG.xml");
    cout << "       训练完毕，XML文件已保存" << endl;

}


void SVM_HOG_detect()
{
    
    Ptr<SVM> svm = SVM::load<SVM>("SVM_HOG.xml"); //或者svm = Statmodel::load<SVM>("SVM_HOG.xml"); static function
            //Ptr<SVM> svm = SVM::load(path);
            //  cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::load<SVM>(path);
    // svm->load<SVM>("SVM_HOG.xml"); 这样使用不行

    if (svm->empty()){ //empty()函数 字符串是空的话返回是true
        cout << "读取XML文件失败。" << endl;
        return;
    }
    else{
        cout << "读取XML文件成功。" << endl;
    }

    
    Mat svecsmat = svm->getSupportVectors();//svecsmat元素的数据类型为float

    int svdim = svm->getVarCount();

    int numofsv = svecsmat.rows;

    Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
    Mat svindex = Mat::zeros(1, numofsv, CV_64F);

    Mat Result;
    double rho = svm->getDecisionFunction(0, alphamat, svindex);
    alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F

    cout << "1" << endl;
    Result = -1 * alphamat * svecsmat;//float
    cout << "2" << endl;

    vector<float> vec;
    for (int i = 0; i < svdim; ++i)
    {
        vec.push_back(Result.at<float>(0, i));
    }
    vec.push_back(rho);
    
    //保存HOG检测的文件
    ofstream fout("HOGDetectorForOpenCV.txt");
    for (int i = 0; i < vec.size(); ++i)
    {
        fout << vec[i] << endl;
    }
    cout << "保存完毕" << endl;

    //----------读取图片进行检测----------------------------
//  HOGDescriptor hog_test;
    HOGDescriptor hog_test(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    hog_test.setSVMDetector(vec);

    Mat src = imread("3.jpg",0);
    if (!src.data){
        cout << "测试图片读取失败" << endl;
        return;
    }
    vector<Rect> found, found_filtered;

    int p = 1;
    resize(src, src, Size(src.cols / p, src.rows / p));

    clock_t startTime, finishTime;
    cout << "开始检测" << endl;

    startTime = clock();                                                //1.05
    hog_test.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);   //多尺度检测
    finishTime = clock();
    cout << "检测所用时间为" <<  (finishTime - startTime)*1.0/CLOCKS_PER_SEC << " 秒 " << endl;

    cout << endl << "矩形框的尺寸为 : " << found.size() << endl;

    //找出所有没有嵌套的矩形,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形放入found_filtered中
    for (int i = 0; i < found.size(); i++)
    {
        Rect r = found[i];
        int j = 0;
        for (; j < found.size(); j++)
        if (j != i && (r & found[j]) == r)
            break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    cout << endl << "嵌套矩形框合并完毕" << endl;

    //画矩形框，因为hog检测出的矩形框比实际的框要稍微大些,所以这里需要做一些调整
    for (int i = 0; i<found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];

        r.x += cvRound(r.width*0.1); //int cvRound(double value) 对一个double型的数进行四舍五入，并返回一个整型数！
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);

        rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
    }

    imwrite("ImgProcessed.jpg", src);
    namedWindow("src", 0);
    imshow("src", src);
    waitKey(0);
}
