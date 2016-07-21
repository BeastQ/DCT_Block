#pragma once
#pragma once
#include <vector>;
#include<opencv2/opencv.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <iostream> 
#include<vector>
#include<string>;
#include<opencv2/opencv.hpp>;
using namespace cv;
using namespace std;
using namespace cv;
using namespace std;
/*
嵌入水印的声明
*/	int minEucDistance(double z1, double delta);
	vector<int> giQimDehide_DCT_block_vertical_Glp(Mat stg, double delta, int vlen, double p, 
		int wfLen, int block[2]);
    vector<int> giQimDehide_DCT_block_vertical_High_Glp(Mat stg, double delta, int vlen, double p, 
		int wfLen, int block[2]);
	//初始化水印信息
	void initWaterMark(int wmZoneLen, int msgLen, int bitdepth, int vlen, double p, double delta, double MarginRatio);
	//嵌入水印算法
	void embedWaterMrak();
	//giQimHide_DCT_block_vertical_Glp函数的实现,参数待定
	Mat giQimHide_DCT_block_vertical_Glp(Mat cfData, vector<int> wfData, double delta,
		int vlen, double p, int block[2]);
	Mat giQimHide_DCT_block_vertical_High_Glp(Mat cfData, vector<int> wfData, double delta,
		int vlen, double p, int block[2]);
	int m, n, t;
	int msgLen;
	int wmZoneLen;
	int bitdepth;
	int vlen;
	double p;
	double  delta;
	double MarginRatio;
	int block[2] = { 8,8 };
	std::vector<int> wfData_rand, wfData_L;
	double MarginH;//选取水印区域的水平方向边距
	double MarginV;//选取水印区域的垂直方向边距
	double RowSt, RowEd, ColSt, ColEd;
	Mat srcRGB, srcLAB, srcLABc, srcCMYK,stg_low,wmLAB,wmRGB, wmLABc, wmRGBc;
	vector<Mat> srcLABc_cs,stgLABc_cs,srcLAB_cs;


/*
通用方法的声明
*/
	//attack实现，返回类型和参数待定
	void attack();//可选，参数：攻击类型名，攻击类型相关的参数
				  //calcPBC实现，返回类型和参数待定
	void calcPBC();//误码率（需要），相关系数（需要），峰值信噪比（可选）	
				   //gauLowPass实现，返回类型和参数待定
	Mat gauLowPass(Mat cfData, int n, double sigma); //入参：图像数据，入参：窗口大小，标准差；返回值：滤波后的图像，
													 //getBer
	void getBer();
	//getCorr
	void getCorr();
	//getPsnr
	void getPsnr();
	//RGB2GRAY
	void RGB2GRAY();
	Mat getMatV(Mat cfL, int r1, int r2, int c1, int c2);
	Mat Creshape(Mat v, int rows);
	Mat getPowMat(Mat v, int i, int p);
	Mat getPowMat(Mat v, int p);
	double getLxOrLy(Mat cc, int vlen, double p1);
	//产生高斯噪声
	double generateGaussianNoise(double Dx);
	//加入高斯白噪声
	Mat cAwgn(Mat v, int i, double db);
	Mat cmykToLab(Mat cfData);
	std::vector<int> readMsg(String dir, int msglen);
/*
算法相关方法类声明
*/

	//以下方法的实现,返回类型和参数待定
	//void initDIR(string *dirs);	//初始化目录具体实现，对应prepare.m									
	void addBorder();//入参：图像数据，入参：边框厚度，返回值：加边框后的图像，						 
	void genData();
	//getDiag
	void getDiag();
	//igetDiag
	void igetDiag();
	//zigzagOrder
	void zigzagOrder();
	void izigzagOrder();
	//minEucDistance
	void minEucDistance();
	//quantificate
	double quantificate(double z, int d, int delta);
	void getWaterMarkRect(Mat img);