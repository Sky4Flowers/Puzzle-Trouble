#pragma once

#include "MarkerCodes.h"
#include "PoseEstimation.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Windows.h>

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
void VoronoiImageSlicing(const Size &size);
void CreatePuzzlePieces(Mat &img);
void DrawPuzzlePiece(Mat puzzlePiece, float xPos, float yPos, float rotAngle);
Mat TranslateImage(Mat &img, int offsetx, int offsety);
Mat RotateImage(Mat &img, float rotAngle);
bool TestEndConfiguration(const std::map<int, Point2f> &endconfig);
void createPuzzle(string puzzleName);
void initUI();
void drawLevelSelection();
void updateUI();
void drawGameRectangle();
void drawButton(Mat &display, Rect &button, int textureIndex, Vec3b &position, String buttonText);
void callBackFunc(int event, int x, int y, int flags, void* userdata);
int Sign(float x);
void cheatWin();
