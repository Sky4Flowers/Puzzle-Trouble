#pragma once

#include "MarkerCodes.h"
#include "PoseEstimation.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

// Struct holding all infos about each strip, e.g. length
struct MyStrip {
	int stripeLength;
	int nStop;
	int nStart;
	Point2f stripVecX;
	Point2f stripVecY;
};

static void on_trackbar(int pos, void* slider_value);
void bw_trackbarHandler(int pos, void* slider_value);
int subpixSampleSafe(const Mat& pSrc, const Point2f& p);
Mat calculateStrip(double dx, double dy, MyStrip& myStrip);
void CaptureLoop(); 
int Sign(float x);
void DrawPuzzlePiece(Mat puzzlePiece, float xPos, float yPos, float rotAngle);
void initUI();
void updateUI();
void callBackFunc(int event, int x, int y, int flags, void* userdata);