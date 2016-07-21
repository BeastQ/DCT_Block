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
Ƕ��ˮӡ������
*/	int minEucDistance(double z1, double delta);
	vector<int> giQimDehide_DCT_block_vertical_Glp(Mat stg, double delta, int vlen, double p, 
		int wfLen, int block[2]);
    vector<int> giQimDehide_DCT_block_vertical_High_Glp(Mat stg, double delta, int vlen, double p, 
		int wfLen, int block[2]);
	//��ʼ��ˮӡ��Ϣ
	void initWaterMark(int wmZoneLen, int msgLen, int bitdepth, int vlen, double p, double delta, double MarginRatio);
	//Ƕ��ˮӡ�㷨
	void embedWaterMrak();
	//giQimHide_DCT_block_vertical_Glp������ʵ��,��������
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
	double MarginH;//ѡȡˮӡ�����ˮƽ����߾�
	double MarginV;//ѡȡˮӡ����Ĵ�ֱ����߾�
	double RowSt, RowEd, ColSt, ColEd;
	Mat srcRGB, srcLAB, srcLABc, srcCMYK,stg_low,wmLAB,wmRGB, wmLABc, wmRGBc;
	vector<Mat> srcLABc_cs,stgLABc_cs,srcLAB_cs;


/*
ͨ�÷���������
*/
	//attackʵ�֣��������ͺͲ�������
	void attack();//��ѡ������������������������������صĲ���
				  //calcPBCʵ�֣��������ͺͲ�������
	void calcPBC();//�����ʣ���Ҫ�������ϵ������Ҫ������ֵ����ȣ���ѡ��	
				   //gauLowPassʵ�֣��������ͺͲ�������
	Mat gauLowPass(Mat cfData, int n, double sigma); //��Σ�ͼ�����ݣ���Σ����ڴ�С����׼�����ֵ���˲����ͼ��
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
	//������˹����
	double generateGaussianNoise(double Dx);
	//�����˹������
	Mat cAwgn(Mat v, int i, double db);
	Mat cmykToLab(Mat cfData);
	std::vector<int> readMsg(String dir, int msglen);
/*
�㷨��ط���������
*/

	//���·�����ʵ��,�������ͺͲ�������
	//void initDIR(string *dirs);	//��ʼ��Ŀ¼����ʵ�֣���Ӧprepare.m									
	void addBorder();//��Σ�ͼ�����ݣ���Σ��߿��ȣ�����ֵ���ӱ߿���ͼ��						 
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