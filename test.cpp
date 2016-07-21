#include "test.h";
#include <vector>;
#include<opencv2/opencv.hpp>;
#include<math.h>
#include<opencv2/opencv.hpp>;
#include<math.h>
#include<fstream>
#include<string>
#include<iostream>
#include"stdlib.h"
using namespace cv;
using namespace std;
double quantificate(double z, int d, double delta) {
	double dither = d*delta / 2;
	double z1 = delta * round((z + dither) / delta) - dither;
	return z1;
}
int minEucDistance(double z1, double delta) {
	double m1;
	double q00 = quantificate(z1, 0, delta);
	double q10 = quantificate(z1, 1, delta);
	double zz[2] = { std::abs(z1 - q00),std::abs(z1 - q10) };
	if (zz[0]<zz[1]) {
		m1 = 0;
	}
	if (zz[0]>zz[1]) {
		m1 = 1;
	}
	return m1;
}
Mat gauLowPass(Mat cfData, int n, double sigma) {
	//入参：图像数据，入参：窗口大小，标准差；返回值：滤波后的图像，
	Mat glData;
	GaussianBlur(cfData, glData, cv::Size(n, n), sigma);
	return glData;
}
Mat Creshape(Mat v, int rows) {
	//v = v.t();
	v = v.reshape(0, rows);
	return v;
}
Mat getPowMat(Mat c, int i, int p) {
	Mat v1SrcC = c.col(i).clone();
	Mat v1DstC;
	cv::pow(v1SrcC, p, v1DstC);
	return v1DstC;
}
Mat getPowMat(Mat c, int p) {
	Mat Dst;
	cv::pow(c, p, Dst);
	return Dst;
}
double getLxOrLy(Mat cc, int vlen, double p1) {
	double sum = 0;
	for (int i = 0; i < cc.rows; i++) {
		sum = sum + cc.at<double>(i, 0);
	}
	double sum1 = std::abs(sum);
	double sum2 = sum1 / vlen;
	return std::pow(sum2, p1);
}
Mat cAwgn(Mat v, int i, double db) {
	Mat vv = getPowMat(v, i, 2);
	double Ssum = 0;//输入信号能量
	for (int j = 0; j < vv.rows; j++) {
		Ssum = Ssum + vv.at<double>(j, 0);
	}
	double Dx = Ssum / (vv.rows*(std::pow(10, db / 10)));//计算用于产生高斯噪声的方差
														 //产生高斯噪声0，dx
	static double V1, V2, S;
	static int phase = 0;
	double X;
	if (phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);
	phase = 1 - phase;
	double gauss = X*(std::sqrt(Dx));
	Mat vi = v.col(i).clone();
	Mat vii = vi;
	for (int j = 0; j < vi.rows; j++) {
		vii.at<double>(j, 0) = vi.at<double>(j, 0) + gauss;
	}
	return vii;
}

vector<int> readMsg(String dir, int msglen) {
	ifstream in(dir);
	vector<int> wfData;
	string s;
	int i = 1;
	while (getline(in, s) && (i <= msglen)) {
		int ss = atoi(s.c_str());
		wfData.push_back(ss);
		i++;
	}
	return wfData;
}
void initWaterMark(int wmZoneLen1, int msgLen1, int bitdepth1,
	int vlen1, double p1, double delta1, double MarginRatio1) {
	wmZoneLen = wmZoneLen1;
	msgLen = msgLen1;
	bitdepth = bitdepth1;
	vlen = vlen1;
	p = p1;
	delta = delta1;
	MarginRatio = MarginRatio1;
	srcRGB = imread("1.tif");
	cvtColor(srcRGB, srcLAB, CV_BGR2Lab);
	m = srcLAB.cols;//获取图像矩阵的列数
	n = srcLAB.rows;//行数
	t = 3;//通道数
	MarginH = floor(m * MarginRatio); //平方向边距
	MarginV = floor(n * MarginRatio); //垂直方向边距
	RowSt = MarginH + 1;
	RowEd = n - MarginH;
	ColSt = MarginV + 1;
	ColEd = m - MarginV;
	//选取嵌入区域	
	split(srcLAB,srcLAB_cs);
	srcLAB(Range(RowSt-1, RowEd ), Range(ColSt-1, ColEd )).copyTo(srcLABc);
	split(srcLABc, srcLABc_cs);
	wfData_rand = readMsg("msg.txt", msgLen);
	wfData_L = readMsg("msg_1_320.txt", msgLen);
	//wfData_L = readMsg("msg_1_512.txt", msgLen);
}
void embedWaterMrak() {
	for (int i = 0; i < t; i++) {
		if (i == 0) {
			stg_low = giQimHide_DCT_block_vertical_Glp(srcLABc_cs[i], wfData_rand, delta, vlen, p, block);
			stgLABc_cs.push_back(giQimHide_DCT_block_vertical_High_Glp(stg_low, wfData_L, delta, vlen, p, block));
		}
		else
		{
			stgLABc_cs.push_back(srcLABc_cs[i]);
		}
	}
	merge(stgLABc_cs,wmLABc);
	wmLABc.copyTo(srcLAB(Range(RowSt-1, RowEd), Range(ColSt-1, ColEd)));
	cvtColor(srcLAB,wmRGB,CV_Lab2BGR);
}
Mat giQimHide_DCT_block_vertical_Glp(Mat cfData, vector<int> wfData,
	double delta, int vlen, double p, int block[2]) {
	Mat cfL = gauLowPass(cfData, 3, 0.5);//低频部分	
	Mat cfH = cfData - cfL;//高频部分
	int si[2] = { cfL.rows,cfL.cols };//获取矩阵cfL的行数和列数
	int len = wfData.size();
	int N1 = floor((floor(si[1] / block[1])*floor(si[0] / block[0])) / (2 * vlen));
	if (len < N1)
	{
		for (int i = len; i < N1; i++) {
			wfData.push_back(0);
		}
	}
	if (len > N1)
	{
		wfData.erase(wfData.begin() + N1, wfData.end());
	}

	int N = floor(si[0] / block[0]); //there are N blocks in each row
	int M = floor(si[1] / block[1]); // there are M block in each colomn
	int oddCount = 0;
	int evenCount = 0;
	Mat v1 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v2 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v = Mat::zeros(M, N, CV_64FC1);

	for (int i = 1; i <= M; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			Mat srcV = cfL(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			v.at<double>(i - 1, j - 1) = V.at<double>(1, 0);
			//cout << "v(i,j)=:" << v.at<double>(i-1, j-1) << ";" << endl << endl;
		}
		if ((i % 2) == 0)
		{
			evenCount = evenCount + 1;
			for (int l = 0; l < N; l++) {
				v2.at<double>(evenCount - 1, l) = v.at<double>(i - 1, l);
			}

		}
		else
		{
			oddCount = oddCount + 1;
			for (int l = 0; l < N; l++) {
				v1.at<double>(oddCount - 1, l) = v.at<double>(i - 1, l);
			}
			//cout << "evenCount - 1=:" << evenCount - 1 << endl;
			//cout << "v1.row(evenCount - 1)=:" << v1.row(oddCount - 1) << ";" << endl << endl;
		}
	}

	//cout << "v1=:" << v1<< ";" << endl << endl;
	v1 = Creshape(v1, vlen);
	v2 = Creshape(v2, vlen);

	for (int i = 1; i <= N1; i++) {
		Mat v1Col = getPowMat(v1, i - 1, p);
		Mat v2Col = getPowMat(v2, i - 1, p);
		double lx = getLxOrLy(v1Col, vlen, 1 / p);
		double ly = getLxOrLy(v2Col, vlen, 1 / p);
		if ((std::abs(lx) <= (1e-6)) || (std::abs(ly) <= (1e-6)))
		{
			Mat v1_noise = cAwgn(v1, i - 1, 50);
			Mat v2_noise = cAwgn(v2, i - 1, 50);
			Mat temp_v1_noise = getPowMat(v1_noise, p);
			Mat temp_v2_noise = getPowMat(v2_noise, p);
			lx = getLxOrLy(temp_v1_noise, vlen, 1 / p);
			ly = getLxOrLy(temp_v2_noise, vlen, 1 / p);
		}
		double z = lx / ly;
		int d = wfData[i - 1];
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;

		double z1 = quantificate(z, d, delta);
		if (z1 == 0)
		{
			z1 = delta / 8;
		}
		if (z1 < 0)
		{
			z1 = z1 + delta;
		}
		double k = std::sqrt(z1 / z);
		for (int j = 0; j < v1.rows; j++) {
			v1.at<double>(j, i - 1) = k*(v1.at<double>(j, i - 1));
		}
		for (int j = 0; j < v2.rows; j++) {
			v2.at<double>(j, i - 1) = (v2.at<double>(j, i - 1)) / k;
		}
	}

	int oddP = 0;
	int evenP = 0;
	Mat v11 = Creshape(v1, M / 2);
	Mat v22 = Creshape(v2, M / 2);
	for (int i = 1; i <= M; i++) {
		if (i % 2 == 1) {
			oddP = oddP + 1;
			for (int j = 0; j < v.cols; j++) {
				v.at<double>(i - 1, j) = v11.at<double>(oddP - 1, j);
			}
		}
		else
		{
			evenP = evenP + 1;
			for (int j = 0; j < v.cols; j++) {
				v.at<double>(i - 1, j) = v22.at<double>(evenP - 1, j);
			}
		}
		for (int j = 1; j <= N; j++) {
			Mat srcV = cfL(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			V.at<double>(1, 0) = v.at<double>(i - 1, j - 1);
			Mat dct;
			idct(V, dct);
			dct.copyTo(cfL(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1])));
		}
	}
	Mat stg = cfL + cfH;
	return stg;
}

Mat giQimHide_DCT_block_vertical_High_Glp(Mat cfData, vector<int> wfData,
	double delta, int vlen, double p, int block[2]) {
	Mat cfL = gauLowPass(cfData, 3, 0.5);//低频部分	
	Mat cfH = cfData - cfL;//高频部分
	int si[2] = { cfH.rows,cfH.cols };//获取矩阵cfL的行数和列数
	int len = wfData.size();
	int N1 = floor((floor(si[1] / block[1])*floor(si[0] / block[0])) / (2 * vlen));
	if (len < N1)
	{
		for (int i = len; i < N1; i++) {
			wfData.push_back(0);
		}
	}
	if (len > N1)
	{
		wfData.erase(wfData.begin() + N1, wfData.end());
	}

	int N = floor(si[0] / block[0]); //there are N blocks in each row
	int M = floor(si[1] / block[1]); // there are M block in each colomn
	int oddCount = 0;
	int evenCount = 0;
	Mat v1 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v2 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v = Mat::zeros(M, N, CV_64FC1);

	for (int i = 1; i <= M; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			Mat srcV = cfH(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			v.at<double>(i - 1, j - 1) = V.at<double>(1, 0);
		}
		if ((i % 2) == 0)
		{
			evenCount = evenCount + 1;
			for (int l = 0; l < N; l++) {
				v2.at<double>(evenCount - 1, l) = v.at<double>(i - 1, l);
			}

		}
		else
		{
			oddCount = oddCount + 1;
			for (int l = 0; l < N; l++) {
				v1.at<double>(oddCount - 1, l) = v.at<double>(i - 1, l);
			}
		}
	}
	v1 = Creshape(v1, vlen);
	v2 = Creshape(v2, vlen);

	for (int i = 1; i <= N1; i++) {
		Mat v1Col = getPowMat(v1, i - 1, p);
		Mat v2Col = getPowMat(v2, i - 1, p);
		double lx = getLxOrLy(v1Col, vlen, 1 / p);
		double ly = getLxOrLy(v2Col, vlen, 1 / p);
		if ((std::abs(lx) <= (1e-6)) || (std::abs(ly) <= (1e-6)))
		{
			Mat v1_noise = cAwgn(v1, i - 1, 50);
			Mat v2_noise = cAwgn(v2, i - 1, 50);
			Mat temp_v1_noise = getPowMat(v1_noise, p);
			Mat temp_v2_noise = getPowMat(v2_noise, p);
			lx = getLxOrLy(temp_v1_noise, vlen, 1 / p);
			ly = getLxOrLy(temp_v2_noise, vlen, 1 / p);
		}
		double z = lx / ly;
		int d = wfData[i - 1];
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;

		double z1 = quantificate(z, d, delta);
		if (z1 == 0)
		{
			z1 = delta / 8;
		}
		if (z1<0)
		{
			z1 = z1 + delta;
		}
		double k = std::sqrt(z1 / z);
		for (int j = 0; j < v1.rows; j++) {
			v1.at<double>(j, i - 1) = k*v1.at<double>(j, i - 1);
		}
		for (int j = 0; j < v2.rows; j++) {
			v2.at<double>(j, i - 1) = v2.at<double>(j, i - 1) / k;
		}
	}

	int oddP = 0;
	int evenP = 0;
	Mat v11 = Creshape(v1, M / 2);
	Mat v22 = Creshape(v2, M / 2);
	for (int i = 1; i <= M; i++) {
		if (i % 2 == 1) {
			oddP = oddP + 1;
			for (int j = 0; j < v.cols; j++) {
				v.at<double>(i - 1, j) = v11.at<double>(oddP - 1, j);
			}
		}
		else
		{
			evenP = evenP + 1;
			for (int j = 0; j < v.cols; j++) {
				v.at<double>(i - 1, j) = v22.at<double>(evenP - 1, j);
			}
		}
		for (int j = 1; j <= N; j++) {
			Mat srcV = cfH(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			V.at<double>(1, 0) = v.at<double>(i - 1, j - 1);
			Mat dct;
			idct(V, dct);
			dct.copyTo(cfH(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1])));
		}
	}
	Mat stg = cfL + cfH;
	return stg;
}

vector<int> giQimDehide_DCT_block_vertical_Glp(Mat cfData, double delta, int vlen,
	double p, int wfLen, int block[2]) {
	vector<int> o;
	Mat cfL = gauLowPass(cfData, 3, 0.5);//低频部分
	int si[2] = { cfL.rows,cfL.cols };//获取矩阵cfL的行数和列数
	int N1 = floor((floor(si[1] / block[1])*floor(si[0] / block[0])) / (2 * vlen));
	int N = floor(si[0] / block[0]); //there are N blocks in each row
	int M = floor(si[1] / block[1]); // there are M block in each colomn
	Mat v1 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v2 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v = Mat::zeros(M, N, CV_64FC1);
	int oddCount = 0;
	int evenCount = 0;
	for (int i = 1; i <= M; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			Mat srcV = cfL(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			v.at<double>(i - 1, j - 1) = V.at<double>(1, 0);
		}
		if ((i % 2) == 0)
		{
			evenCount = evenCount + 1;
			for (int l = 0; l < N; l++) {
				v2.at<double>(evenCount - 1, l) = v.at<double>(i - 1, l);
			}

		}
		else
		{
			oddCount = oddCount + 1;
			for (int l = 0; l < N; l++) {
				v1.at<double>(oddCount - 1, l) = v.at<double>(i - 1, l);
			}
		}
	}
	v1 = Creshape(v1, vlen);
	v2 = Creshape(v2, vlen);
	for (int i = 1; i <= N1; i++) {
		Mat v1Col = getPowMat(v1, i - 1, p);
		Mat v2Col = getPowMat(v2, i - 1, p);
		double lx = getLxOrLy(v1Col, vlen, 1 / p);
		double ly = getLxOrLy(v2Col, vlen, 1 / p);
		if ((std::abs(lx) <= (1e-6)) || (std::abs(ly) <= (1e-6)))
		{
			Mat v1_noise = cAwgn(v1, i - 1, 50);
			Mat v2_noise = cAwgn(v2, i - 1, 50);
			Mat temp_v1_noise = getPowMat(v1_noise, p);
			Mat temp_v2_noise = getPowMat(v2_noise, p);
			lx = getLxOrLy(temp_v1_noise, vlen, 1 / p);
			ly = getLxOrLy(temp_v2_noise, vlen, 1 / p);
		}
		double z = lx / ly;
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;
		o.push_back(minEucDistance(z, delta));
	}
	if (wfLen < N1) {
		o.erase(o.begin() + wfLen, o.end());	}
	return o;
}
vector<int> giQimDehide_DCT_block_vertical_High_Glp(Mat cfData, double delta, int vlen,
	double p, int wfLen, int block[2]) {
	vector<int> o;
	Mat cfL = gauLowPass(cfData, 3, 0.5);//低频部分	
	Mat cfH = cfData - cfL;//高频部分
	int si[2] = { cfH.rows,cfH.cols };//获取矩阵cfL的行数和列数
	int N1 = floor((floor(si[1] / block[1])*floor(si[0] / block[0])) / (2 * vlen));
	int N = floor(si[0] / block[0]); //there are N blocks in each row
	int M = floor(si[1] / block[1]); // there are M block in each colomn
	Mat v1 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v2 = Mat::zeros(M / 2, N, CV_64FC1);
	Mat v = Mat::zeros(M, N, CV_64FC1);
	int oddCount = 0;
	int evenCount = 0;
	for (int i = 1; i <= M; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			Mat srcV = cfH(Range((i - 1)*block[0], i*block[0]), Range((j - 1)*block[1], j*block[1]));
			Mat V_64, V;
			srcV.convertTo(V_64, CV_64FC1);
			dct(V_64, V);
			v.at<double>(i - 1, j - 1) = V.at<double>(1, 0);
		}
		if ((i % 2) == 0)
		{
			evenCount = evenCount + 1;
			for (int l = 0; l < N; l++) {
				v2.at<double>(evenCount - 1, l) = v.at<double>(i - 1, l);
			}

		}
		else
		{
			oddCount = oddCount + 1;
			for (int l = 0; l < N; l++) {
				v1.at<double>(oddCount - 1, l) = v.at<double>(i - 1, l);
			}
		}
	}
	v1 = Creshape(v1, vlen);
	v2 = Creshape(v2, vlen);
	for (int i = 1; i <= N1; i++) {
		Mat v1Col = getPowMat(v1, i - 1, p);
		Mat v2Col = getPowMat(v2, i - 1, p);
		double lx = getLxOrLy(v1Col, vlen, 1 / p);
		double ly = getLxOrLy(v2Col, vlen, 1 / p);
		if ((std::abs(lx) <= (1e-6)) || (std::abs(ly) <= (1e-6)))
		{
			Mat v1_noise = cAwgn(v1, i - 1, 50);
			Mat v2_noise = cAwgn(v2, i - 1, 50);
			Mat temp_v1_noise = getPowMat(v1_noise, p);
			Mat temp_v2_noise = getPowMat(v2_noise, p);
			lx = getLxOrLy(temp_v1_noise, vlen, 1 / p);
			ly = getLxOrLy(temp_v2_noise, vlen, 1 / p);
		}
		double z = lx / ly;
		if ((std::abs(lx) <= 1e-6) || (std::abs(ly) <= 1e-6))
			continue;
		o.push_back(minEucDistance(z, delta));
	}
	if (wfLen < N1) {
		o.erase(o.begin() + wfLen, o.end());
	}
	return o;
}

void main() {
	initWaterMark(320, 25, 8, 32, 2.0, 0.7, 0.174);
//	initWaterMark(512, 32, 8, 64, 2.0, 0.7, 0.05);
	embedWaterMrak();
	imshow("srcRGB", srcRGB);
	imshow("WMRGB", wmRGB);
	//imwrite("wmrgb_512.tif",wmRGB);
	imwrite("wmrgb_320.tif", wmRGB);
	vector<int> msg320_random, msg320_L, msg512_random, msg512_L;
	/*msg512_random = giQimDehide_DCT_block_vertical_Glp(stgLABc_cs[0], 0.7, 64,
		2.0, 32, block);
	msg512_L= giQimDehide_DCT_block_vertical_High_Glp(stgLABc_cs[0], 0.7, 64,
		2.0, 32, block);*/
	msg320_random = giQimDehide_DCT_block_vertical_Glp(stgLABc_cs[0], 0.7, 32,
		2.0, 25, block);
	msg320_L = giQimDehide_DCT_block_vertical_High_Glp(stgLABc_cs[0], 0.7, 32,
		2.0, 25, block);
	for (int i = 0; i < msg320_random.size();i++) {
		cout << msg320_random[i]<< endl;
	}
	waitKey(0);
	system("pause");
}



