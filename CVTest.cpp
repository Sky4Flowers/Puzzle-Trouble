#include "CVTest.h"

const int thresholdMaxValue = 255;
const int thresholdMaxType = 4;
const float markerSize = 0.043;
const bool cheatingEnabled = true;
int thresholdValue = 148;
int thresholdType = 0;
int adaptiveValue;
int levels = 5;

const int cameraXRes = 640;
const int cameraYRes = 480;
char* markerWindowName = "MarkerWindow";
char* streamWindowName = "StreamWindow";
char* trackbarName = "value";
char* buttonName = "Normal: 0\n Adaptive: 1";
char* trackbarWindowName = "TrackbarWindow";

bool isFirstMarker = true;
bool couldNotLoadImage = false;
VideoCapture cap(0);

vector<Vec4i> hierarchy;
Mat frame, gray, finalImage;

RNG rng;
const int numberOfPieces = 4;
const double imageSize = 200;
vector<vector<Point2f> > voronoiMasks;
vector<Point2f> voronoiCenters;
vector<Mat> puzzlePieces;
const double distanceThreshold = 10.0;
const double winRecognitionDuration = 3.0;
double startWinTime;
bool detectingWin;

//Variables for UI
Rect startButton;
Rect levelButton;
Rect menuButton;
Rect rerollButton;
Rect quitButton;

int levelCount = 6;
Mat* levelImages;
Rect* levelButtons;

//enum controls state of game loop
enum State { main_menu, level_select, playing, win, quit };

State state = main_menu;

//freezed image
Mat freezed;
Mat levelSelectionBg;

int main(int, void*)
{
	if (!cap.isOpened()) {
		cout << "No capture" << endl;
		return -1;
	}

	namedWindow(streamWindowName, CV_WINDOW_FULLSCREEN);
	/*createTrackbar(trackbarName,
		streamWindowName, &thresholdValue,
		thresholdMaxValue, on_trackbar, &thresholdValue);
	createTrackbar(buttonName, streamWindowName, &adaptiveValue, 1, on_trackbar, &adaptiveValue);

	createTrackbar("levels+3", streamWindowName, &levels, 7, on_trackbar, &levels);*/
	//namedWindow(markerWindowName, CV_WINDOW_NORMAL);
	//resizeWindow(markerWindowName, 120, 120);

	initUI();
	Mat menuBg = Mat(cameraYRes, cameraXRes, CV_8UC3, Scalar(0, 0, 0)); //backgroundImage for main menu
	levelImages[levelCount + 7].copyTo(menuBg);
	levelImages[levelCount + 8].copyTo(menuBg, levelImages[levelCount + 8] != 0);
	levelSelectionBg = menuBg.clone();
	
	while (cap.read(frame)) {

		if (waitKey(1) == 27)
			break;
		if (couldNotLoadImage) {
			return -1;
		}

		//set gui for the main menu
		if (state == main_menu) {
			drawButton(menuBg, startButton, levelCount, Vec3b(200, 200, 200), "");
			drawButton(menuBg, levelButton, levelCount + 1, Vec3b(200, 200, 200), "");
			drawButton(menuBg, quitButton, levelCount + 3, Vec3b(200, 200, 200), "");

			imshow(streamWindowName, menuBg);
		}

		//stuff before the level select button is pressed
		else if (state == level_select) {
			//set gui for level selection
			//once a level has been selected, change the image matrix to the selected image
			drawLevelSelection();
			imshow(streamWindowName, levelSelectionBg);
		}

		//begin a game
		else if (state == playing) {
			//game logic without displaying image
			CaptureLoop();

			//display Buttons
			//drawButton(finalImage, rerollButton, levelCount + 3, Vec3b(200, 200, 200), ""); //No quit button in playing state
			drawButton(finalImage, menuButton, levelCount + 2, Vec3b(200, 200, 200), "");

			imshow(streamWindowName, finalImage);
		}
		//game is won
		else if (state == win) {
			//freeze frame / other post win logic here
			imshow(streamWindowName, freezed);
			drawButton(freezed, menuButton, levelCount + 2, Vec3b(200, 200, 200), "");
			//to do: still need to draw the you win title
		}
		else if (state == quit) {
			break;
		}
	}

	destroyWindow(streamWindowName);
	//destroyWindow(markerWindowName);
	return 0;
}

static void on_trackbar(int pos, void* slider_value) {
	*((int*)slider_value) = pos;
}

int subpixSampleSafe(const Mat& finalImage, const Point2f& subPixel) {
	// Point is float, slide 14
	int fx = int(floorf(subPixel.x));
	int fy = int(floorf(subPixel.y));

	if (fx < 0 || fx >= finalImage.cols - 1 ||
		fy < 0 || fy >= finalImage.rows - 1)
		return 127;

	// Slides 15
	int px = int(256 * (subPixel.x - fx));
	int py = int(256 * (subPixel.y - fy));

	//from here no idea
	// Here we get the pixel of the starting point
	unsigned char* i = (unsigned char*)((finalImage.data + fy * finalImage.step) + fx);

	// Internsity, shift 3
	int a = i[0] + ((px * (i[1] - i[0])) >> 8);
	i += finalImage.step;
	int b = i[0] + ((px * (i[1] - i[0])) >> 8);

	// We want to return Intensity for the subpixel
	return a + ((py * (b - a)) >> 8);
}

Mat calculateStrip(double dx, double dy, MyStrip& myStrip) {
	//Set length of strip with min value
	const int minStripLength = 5; // should be odd
	double diffLength = sqrt(dx * dx + dy * dy); //Length of direction Vector
	myStrip.stripeLength = (int)(0.8 * diffLength);
	if (myStrip.stripeLength < minStripLength)
		myStrip.stripeLength = minStripLength;

	// Make stripeLength odd 
	myStrip.stripeLength |= 1;//bitwise or operation => +1 if even / +0 if odd

	//=> indices = -length/2 to length/2(had to be odd because of 0)
	myStrip.nStop = myStrip.stripeLength / 2; //round down
	myStrip.nStart = -myStrip.nStop;

	Size stripSize;
	stripSize.width = 3;
	stripSize.height = myStrip.stripeLength;

	// Normalized direction vector - vector from point to point(= edge vector)
	myStrip.stripVecX.x = dx / diffLength;
	myStrip.stripVecX.y = dy / diffLength;

	// Normalized perpendicular vector
	myStrip.stripVecY.x = myStrip.stripVecX.y;
	myStrip.stripVecY.y = -myStrip.stripVecX.x;

	// 8 bit unsigned char with 1 channel, gray
	return Mat(stripSize, CV_8UC1);
}

void CaptureLoop() {
	finalImage = frame.clone();
	//Convert to greyscale
	cvtColor(frame, gray, CV_BGR2GRAY);

	//Threshold:
	if (adaptiveValue == 1)
		adaptiveThreshold(gray, gray, thresholdMaxValue, ADAPTIVE_THRESH_MEAN_C, thresholdType, 11, 12);
	else
		threshold(gray, gray, thresholdValue, thresholdMaxValue, thresholdType);

	vector<vector<Point> > contours;
	findContours(gray, contours,
		RETR_LIST, CHAIN_APPROX_SIMPLE);
	std::map<int, Point2f> endConfiguration;
	for (size_t i = 0; i < contours.size(); i++) {
		//Approximate the contour as a polygon:
		vector<Point> approx_contour;
		approxPolyDP(contours[i], approx_contour, arcLength(contours[i], true) * 0.02, true);
		Rect r = boundingRect(approx_contour);
		//Test polygon:
		if (approx_contour.size() != 4)
			continue;
		if (r.height < 20 || r.width < 20 || r.width > finalImage.cols - 10 || r.height > finalImage.rows - 10) {
			continue;
		}
		//Draw polygon if it survives test //Debug output
		//polylines(finalImage, approx_contour, true, CV_RGB(255, 0, 0), 4);

		float lineParams[16];
		Mat lineParamsMat(Size(4, 4), CV_32F, lineParams);

		//Draw 7 circles with equivalent distance on polygon lines => 7 intervalls
		for (size_t j = 0; j < approx_contour.size(); j++) {

			//First circle on position of polygon vertex //Debug output
			//circle(finalImage, approx_contour[j], 3, CV_RGB(0, 255, 0), -1);
			// calculate stepsize %4 because it is a rectangle; /7 because we want 7 circles
			double dx = ((double)approx_contour[(j + 1) % 4].x - (double)approx_contour[j].x) / 7.0;
			double dy = ((double)approx_contour[(j + 1) % 4].y - (double)approx_contour[j].y) / 7.0;

			//create stripes
			MyStrip strip;
			Mat imagePixelStrip = calculateStrip(dx, dy, strip);
			Point2f edgePointCenters[6];
			//draw other 6 circles(delimiters)
			for (int k = 1; k < 7; k++) {
				double px = (double)approx_contour[j].x + (double)k * dx;
				double py = (double)approx_contour[j].y + (double)k * dy;

				Point p;
				p.x = (int)px;
				p.y = (int)py;
				//Debug output
				//circle(finalImage, p, 2, CV_RGB(0, 0, 255), -1);

				//add stripes for each circle
				// Columns: Loop over 3 pixels
				for (int m = -1; m <= 1; m++) {
					// Rows: From bottom to top of the stripe, e.g. -3 to 3
					for (int n = strip.nStart; n <= strip.nStop; n++) {
						Point2f subPixel;

						// m -> going over the 3 pixel thickness of the stripe, n -> over the length of the stripe, direction comes from the orthogonal vector in st
						// Going from bottom to top and defining the pixel coordinate for each pixel belonging to the stripe
						subPixel.x = (double)p.x + ((double)m * strip.stripVecX.x) + ((double)n * strip.stripVecY.x);
						subPixel.y = (double)p.y + ((double)m * strip.stripVecX.y) + ((double)n * strip.stripVecY.y);

						Point p2;
						p2.x = (int)subPixel.x;
						p2.y = (int)subPixel.y;

						//Debug output
						//circle(finalImage, p2, 1, CV_RGB(0, 255, 255), -1);

						// Combined Intensity of the subpixel
						int pixelIntensity = subpixSampleSafe(gray, subPixel);

						// Converte from index to pixel coordinate
						// m (Column, real) -> -1,0,1 but we need to map to 0,1,2 -> add 1 to 0..2
						int w = m + 1;

						// n (Row, real) -> add stripeLenght >> 1 to shift to 0..stripeLength
						// n=0 -> -length/2, n=length/2 -> 0 ........ + length/2
						int h = n + (strip.stripeLength >> 1);

						// Set pointer to correct position and safe subpixel intensity
						imagePixelStrip.at<uchar>(h, w) = (uchar)pixelIntensity;
					}
				}
				// The first and last row must be excluded from the sobel calculation because they have no top or bottom neighbors
				vector<double> sobelValues(strip.stripeLength - 2.);

				// To use the kernel we start with the second row (n) and stop before the last one
				for (int n = 1; n < (strip.stripeLength - 1); n++) {
					// Take the intensity value from the stripe 
					unsigned char* stripePtr = &(imagePixelStrip.at<uchar>(n - 1, 0));

					// Calculation of the gradient with the sobel for the first row
					double r1 = -stripePtr[0] - 2. * stripePtr[1] - stripePtr[2];

					// Go two lines for the third line of the sobel, step = size of the data type, here uchar
					stripePtr += 2 * imagePixelStrip.step;

					// Calculation of the gradient with the sobel for the third row
					double r3 = stripePtr[0] + 2. * stripePtr[1] + stripePtr[2];

					// Writing the result into our sobel value vector
					unsigned int ti = n - 1;
					sobelValues[ti] = r1 + r3;
				}

				double maxIntensity = -1;
				int maxIntensityIndex = 0;

				// Finding the max value
				for (int n = 0; n < strip.stripeLength - 2; n++) {
					if (sobelValues[n] > maxIntensity) {
						maxIntensity = sobelValues[n];
						maxIntensityIndex = n;
					}
				}

				//Find Parabola:
				// f(x) slide 7 -> y0 .. y1 .. y2
				double y0, y1, y2;

				// Point before and after
				unsigned int max1 = maxIntensityIndex - 1, max2 = maxIntensityIndex + 1;

				// If the index is at the border we are out of the stripe, then we will take 0
				y0 = (maxIntensityIndex <= 0) ? 0 : sobelValues[max1];
				y1 = sobelValues[maxIntensityIndex];
				// If we are going out of the array of the sobel values
				y2 = (maxIntensityIndex >= strip.stripeLength - 3) ? 0 : sobelValues[max2];

				// Formula for calculating the x-coordinate of the vertex of a parabola, given 3 points with equal distances 
				// (xv means the x value of the vertex, d the distance between the points): 
				// xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)

				// Equation system
				// d = 1 because of the normalization and x1 will be added later
				double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

				// What happens when there is no solution? -> /0 or Number = other Number
				// If the found pos is not a number -> there is no solution
				if (isnan(pos)) {
					continue;
				}

				// Exact point with subpixel accuracy
				Point2d edgeCenter;

				// Back to Index positioning, Where is the edge (max gradient) in the picture?
				int maxIndexShift = maxIntensityIndex - (strip.stripeLength >> 1);

				// Shift the original edgepoint accordingly -> Is the pixel point at the top or bottom?
				edgeCenter.x = (double)p.x + (((double)maxIndexShift + pos) * strip.stripVecY.x);
				edgeCenter.y = (double)p.y + (((double)maxIndexShift + pos) * strip.stripVecY.y);

				// Highlight the subpixel with blue color
				//Debug output
				//circle(finalImage, edgeCenter, 2, CV_RGB(0, 0, 255), -1);
				edgePointCenters[k - 1].x = edgeCenter.x;
				edgePointCenters[k - 1].y = edgeCenter.y;
			}//End of approxContour point loop

			Mat highIntensityPoints(Size(1, 6), CV_32FC2, edgePointCenters);
			fitLine(highIntensityPoints, lineParamsMat.col(j), CV_DIST_L2, 0, 0.01, 0.01);
			Point p1;
			p1.x = (int)lineParams[8 + j] - (int)(50.0 * lineParams[j]);
			p1.y = (int)lineParams[12 + j] - (int)(50.0 * lineParams[4 + j]);

			Point p2;
			p2.x = (int)lineParams[8 + j] + (int)(50.0 * lineParams[j]);
			p2.y = (int)lineParams[12 + j] + (int)(50.0 * lineParams[4 + j]);

			//Debug output
			//line(finalImage, p1, p2, CV_RGB(0, 255, 255), 1, 8, 0);
		}//end of circles

		float centerXSum = 0, centerYSum = 0;
		//We have now exact line params but not the corners yet
		Point2f corners[4];
		for (int i = 0; i < 4; i++) {
			//i = current point Index
			int j = (i + 1) % 4;//next point Index
			double x0, x1, y0, y1, u0, u1, v0, v1;

			// We have to jump through the 4x4 matrix, meaning the next value for the wanted line is in the next row -> +4
			// We want to have the point first
			// g1 = (x0,y0) + a*(u0,v0) == g2 = (x1,y1) + b*(u1,v1)
			x0 = lineParams[i + 8]; y0 = lineParams[i + 12];
			x1 = lineParams[j + 8]; y1 = lineParams[j + 12];

			// Direction vector
			u0 = lineParams[i]; v0 = lineParams[i + 4];
			u1 = lineParams[j]; v1 = lineParams[j + 4];

			//Math:
			// Cramer's rule
			// 2 unknown a,b -> Equation system
			double a = x1 * u0 * v1 - y1 * u0 * u1 - x0 * u1 * v0 + y0 * u0 * u1;
			double b = -x0 * v0 * v1 + y0 * u0 * v1 + x1 * v0 * v1 - y1 * v0 * u1;

			// Calculate the cross product to check if both direction vectors are parallel -> = 0
			// c -> Determinant = 0 -> linear dependent -> the direction vectors are parallel -> No division with 0
			double c = v1 * u0 - v0 * u1;
			if (fabs(c) < 0.001) {
				std::cout << "lines parallel" << std::endl;
				continue;
			}

			// We have checked for parallelism of the direction vectors
			// -> Cramer's rule, now divide through the main determinant
			a /= c;
			b /= c;

			// Exact corner
			corners[i].x = a;
			corners[i].y = b;
			centerXSum += a;
			centerYSum += b;
			//Draw Exact corners
			Point p;
			p.x = (int)corners[i].x;
			p.y = (int)corners[i].y;

			//Debug output
			//circle(finalImage, p, 5, CV_RGB(255, 255, 0), -1);
		}

		//Marker Identification:
		// Coordinates on the original marker images to go to the actual center of the first pixel -> 6x6
		Point2f targetCorners[4];
		targetCorners[0].x = -0.5; targetCorners[0].y = -0.5;
		targetCorners[1].x = 5.5; targetCorners[1].y = -0.5;
		targetCorners[2].x = 5.5; targetCorners[2].y = 5.5;
		targetCorners[3].x = -0.5; targetCorners[3].y = 5.5;

		Mat homographyMatrix(Size(3, 3), CV_32FC1);
		// Corner which we calculated and our target Mat, find the transformation
		homographyMatrix = getPerspectiveTransform(corners, targetCorners);

		Mat markerImage(Size(6, 6), CV_8UC1);
		warpPerspective(gray, markerImage, homographyMatrix, Size(6, 6));

		// Now we have a B/W image of a supposed Marker and we need to get the code of the marker
		int code = 0;
		for (int i = 0; i < 6; ++i) {
			// Check if border is black
			int pixel1 = markerImage.at<uchar>(0, i); //top
			int pixel2 = markerImage.at<uchar>(5, i); //bottom
			int pixel3 = markerImage.at<uchar>(i, 0); //left
			int pixel4 = markerImage.at<uchar>(i, 5); //right

			// 0 -> black
			if ((pixel1 > 0) || (pixel2 > 0) || (pixel3 > 0) || (pixel4 > 0)) {
				code = -1;
				break;
			}
		}

		if (code < 0) {
			continue;
		}

		// Copy the BW values into cP -> codePixel on the marker 4x4 (inner part of the marker, no black border)
		int cP[4][4];
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				// +1 -> no borders!
				cP[i][j] = markerImage.at<uchar>(i + 1, j + 1);
				// If black then 1 else 0
				cP[i][j] = (cP[i][j] == 0) ? 1 : 0;
			}
		}

		// Save the ID of the marker, for each side
		int codes[4];
		codes[0] = codes[1] = codes[2] = codes[3] = 0;

		// Calculate the code from all sides at once
		for (int i = 0; i < 16; i++) {
			// /4 to go through the rows
			int row = i >> 2;
			int col = i % 4;

			// Multiplied by 2 to check for black values -> 0*2 = 0
			codes[0] <<= 1;
			codes[0] |= cP[row][col]; // 0�

			// 4x4 structure -> Each column represents one side 
			codes[1] <<= 1;
			codes[1] |= cP[3 - col][row]; // 90�

			codes[2] <<= 1;
			codes[2] |= cP[3 - row][3 - col]; // 180�

			codes[3] <<= 1;
			codes[3] |= cP[col][3 - row]; // 270�
		}
		// Account for symmetry -> One side complete white or black
		if ((codes[0] == 0) || (codes[0] == 0xffff)) {
			continue;
		}

		// Search for the smallest marker ID
		code = codes[0];
		int angle = 0;
		for (int i = 1; i < 4; ++i) {
			if (codes[i] < code) {
				code = codes[i];
				angle = i;
			}
		}

		// Print ID
		//printf("Found: %04x\n", code);
		//cout << markerIds[code] << endl;
		// Show the first detected marker in the image
		if (isFirstMarker) {
			//imshow(markerWindowName, markerImage);
			isFirstMarker = false;
		}

		Point2f corrected_corners[4];
		if (angle != 0) {
			// Smallest id represents the x-axis, we put the values in the corrected_corners array
			for (int i = 0; i < 4; i++)	corrected_corners[(i + angle) % 4] = corners[i];
			// We put the values back in the array in the sorted order
			for (int i = 0; i < 4; i++)	corners[i] = corrected_corners[i];
		}
		float xDirection, yDirection;
		xDirection = corners[0].x - corners[1].x;
		yDirection = corners[0].y - corners[1].y;
		//cout << corners[0] - corners[1] << " ";
		float normalizer = sqrtf(xDirection * xDirection + yDirection * yDirection);
		float rotAngle = 0;
		if (normalizer != 0) {
			xDirection /= normalizer;
			rotAngle = -Sign(yDirection) * acos(xDirection * 1) / 3.1415 * 180;
			//cout << rotAngle << endl;
		}
		else {
			cout << "normalizer = 0 - " << xDirection << endl;
		}
		// Normally we should do a camera calibration to get the camera paramters such as focal length
		// Two ways: Inner parameters, e.g. focal length (intrinsic parameters); camera with 6 dof (R|T) (extrinsic parameters)
		// Transfer screen coords to camera coords -> To get to the principel point
		for (int i = 0; i < 4; i++) {
			// Here you have to use your own camera resolution (x) * 0.5
			corners[i].x -= cameraXRes / 2;
			// -(corners.y) -> is neeeded because y is inverted
			// Here you have to use your own camera resolution (y) * 0.5
			corners[i].y = -corners[i].y + cameraYRes / 2;
		}
		centerXSum /= 4;
		centerYSum /= 4;

		// 4x4 -> Rotation | Translation
		//        0  0  0  | 1 -> (Homogene coordinates to combine rotation, translation and scaling)
		float resultMatrix[16];
		float rotationMatrixArray[9];

		// Marker size in meters!
		estimateSquarePose(resultMatrix, (Point2f*)corners, markerSize);

		int index = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				resultMatrix[index++] = resultMatrix[4 * i + j];
				//cout << resultMatrix[4 * i + j] << " ";
			}
			//cout << endl;
		}
		int puzzleIndex = markerIds[code] - 1;
		//cout << puzzleIndex << endl;
		if (puzzleIndex >= 0 && puzzleIndex < puzzlePieces.size()) {
			endConfiguration[puzzleIndex] = Point2f(centerXSum, centerYSum);
			DrawPuzzlePiece(puzzlePieces[puzzleIndex], centerXSum, centerYSum, rotAngle);
		}
		// Copy the pixels from src to dst.
	// Copy the pixels from src to dst.
		float x, y, z;
		// Translation values in the transformation matrix to calculate the distance between the marker and the camera
		x = resultMatrix[3];
		y = resultMatrix[7];
		z = resultMatrix[11];
		// Euclidian distance
		//cout << "Translation: " << x << " - " << y << " - " << z << "\n";
		//cout << "\n";

	}//End of contour loop
	if (cheatingEnabled) {
		cheatWin();
	}
	if (endConfiguration.size() == numberOfPieces && TestEndConfiguration(endConfiguration)) {
		if (detectingWin) {
			cout << time(NULL) << " - " << startWinTime << endl;
			if (time(NULL) - startWinTime < winRecognitionDuration) {
				winGame();
			}
		}
		else {
			detectingWin = true;
			startWinTime = time(NULL);
		}
	}
	else {
		detectingWin = false;
	}

	//Add UI
	//updateUI(); //removed since not all ui elements should be called in all states
	drawGameRectangle();
	isFirstMarker = true;
}

void DrawPuzzlePiece(Mat puzzlePiece, float xPos, float yPos, float rotAngle) {
	Mat rotatedPuzzlePiece = RotateImage(puzzlePiece, rotAngle);
	if (xPos - rotatedPuzzlePiece.cols / 2 > 0 && xPos + rotatedPuzzlePiece.cols / 2 < finalImage.cols &&
		yPos - rotatedPuzzlePiece.rows / 2 > 0 && yPos + rotatedPuzzlePiece.rows / 2 < finalImage.rows) {
		rotatedPuzzlePiece.copyTo(finalImage(Rect(xPos - rotatedPuzzlePiece.cols / 2, yPos - rotatedPuzzlePiece.rows / 2, rotatedPuzzlePiece.cols, rotatedPuzzlePiece.rows)), rotatedPuzzlePiece != 0);
	}
	else {
		cout << "PuzzlePiece outside of frame" << endl;
	}
}

//Param<size> determines size of to be sliced image
//Result<voronoiMasks> vector< vector<Point2f> > = List of pointLists used for polygon of mask
//Result<vornoiCenters> vector<Point2f> = List of centerpoints of polygons
void VoronoiImageSlicing(const Size &size) {
	int maxTries = 30;
	int currentTry = 0;
	bool tryAgain = true;
	while (currentTry < maxTries && tryAgain) {
		voronoiMasks.clear();
		voronoiCenters.clear();
		tryAgain = false;
		vector<Point2f> points;
		for (int i = 0; i < numberOfPieces; i++) {
			points.push_back(Point2f(rng.uniform(0.0, imageSize), rng.uniform(0.0, imageSize)));
		}
		Rect rect(0, 0, size.width, size.height);
		Subdiv2D subdiv(rect);
		for each(Point2f point in points) {
			subdiv.insert(point);
		}
		subdiv.getVoronoiFacetList(vector<int>(), voronoiMasks, voronoiCenters);

		for (int i = 0; i < numberOfPieces; i++) {
			for (int j = 0; j < numberOfPieces; j++) {
				if (i != j) {
					Point2f targetVector = voronoiCenters[i] - voronoiCenters[j];
					double targetDistance = sqrtf(targetVector.x * targetVector.x + targetVector.y * targetVector.y);
					if (targetDistance < 50) {
						tryAgain = true;
					}
				}
			}
		}
		currentTry++;
	}
}

//Param<size> size of image
//Param<pts> points of the polygon creating the mask
//Return<mask> Mat of the mask
Mat CreateMask(const Size &size, const vector<Point2f> &_pts) {
	vector<Point> pts(_pts.size());
	for (int i = 0; i < _pts.size(); i++) {
		pts[i] = _pts[i];
	}
	const Point* elementPoints[1] = { &pts[0] };
	int numPoints = (int)pts.size();
	Mat mMask = Mat::zeros(size, CV_8U);
	fillPoly(mMask, elementPoints, &numPoints, 1, Scalar(255));
	return mMask;
}

//Creates the image slices from the voronoiMasks and puts them into the puzzlePieces List
void CreatePuzzlePieces(Mat &img) {
	Size imageSize = img.size();
	VoronoiImageSlicing(imageSize);
	puzzlePieces.clear();
	for (int i = 0; i < numberOfPieces; i++) {
		Mat dst;
		Mat mask = CreateMask(img.size(), voronoiMasks[i]);
		img.copyTo(dst, mask);
		Point2f imageCenter = voronoiCenters[i];
		dst = TranslateImage(dst, imageSize.width / 2 - imageCenter.x, imageSize.height / 2 - imageCenter.y); //Recenter image
		//dst = RotateImage(dst, rng.uniform(0.0, 360.0)); //Give image random rotation
		puzzlePieces.push_back(dst);
	}
}

bool TestEndConfiguration(const std::map<int, Point2f> &endconfig) {
	int counter = 0;
	double directionDiffSum = 0;
	double distanceDiffSum = 0;
	double rotationDiffSum = 0;
	for (int i = 0; i < numberOfPieces; i++) {
		for (int j = 0; j < numberOfPieces; j++) {
			if (i != j) {
				Point2f liveVector = endconfig.find(i)->second - endconfig.find(j)->second;
				double liveDistance = sqrtf(liveVector.x * liveVector.x + liveVector.y * liveVector.y);
				liveVector = liveVector / liveDistance;
				Point2f targetVector = voronoiCenters[i] - voronoiCenters[j];
				double targetDistance = sqrtf(targetVector.x * targetVector.x + targetVector.y * targetVector.y);
				targetVector = targetVector / targetDistance;
				directionDiffSum += acosf(targetVector.x * liveVector.x + targetVector.y * liveVector.y) * 180.0 / 3.14159265;
				distanceDiffSum += abs(liveDistance - targetDistance);
				counter++;
			}
		}
	}
	//cout << "Avg Similarity = " << (directionDiffSum / counter) << endl;
	cout << "Avg Distance = " << (distanceDiffSum / counter) << endl;
	return (distanceDiffSum / counter) < distanceThreshold;
}

Mat TranslateImage(Mat &img, int offsetx, int offsety) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}

Mat RotateImage(Mat &img, float rotAngle) {
	Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	Mat rotationMatrix = getRotationMatrix2D(center, rotAngle, 1.0);
	// determine bounding rectangle, center not relevant
	Rect2f bbox = RotatedRect(Point2f(), img.size(), rotAngle).boundingRect2f();
	// adjust transformation matrix
	rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

	cv::Mat rotatedImg;
	cv::warpAffine(img, rotatedImg, rotationMatrix, bbox.size());
	return rotatedImg;
}

void winGame() {
	freezed = Mat(cameraYRes, cameraXRes, CV_8UC3, Scalar(0, 0, 0)); //backgroundImage for main menu
	cout << "You Win" << endl;
	state = win; //change the game state to win

	//cvtColor(finalImage, freezed, CV_BGR2GRAY);//for bw does not work 
	finalImage.copyTo(freezed);
	//add the win title to the image

	Mat win_title = imread("images/WinTitle_Orange.png", -1);

	Mat mask;
	vector<Mat> rgbLayer;
	split(win_title, rgbLayer);

	if (win_title.channels() == 4)
	{
		split(win_title, rgbLayer);
		Mat cs[3] = { rgbLayer[0],rgbLayer[1],rgbLayer[2] };
		merge(cs, 3, win_title);
		mask = rgbLayer[3];
	}

	win_title.copyTo(freezed, mask);

	//win_title.copyTo(freezed(Rect(50, 50, win_title.cols, win_title.rows))); // check this
}
void createPuzzle(int levelID) {
	rng.state = time(NULL); //initialize RNG Seed

	resize(levelImages[levelID], levelImages[levelID], Size(imageSize, imageSize));

	CreatePuzzlePieces(levelImages[levelID]); //slice up the image
}

void initUI() {
	
	startButton = Rect(50, 350, 150, 75);
	levelButton = Rect(235, 350, 150, 75);
	menuButton = Rect(5, 5, 50, 25);
	rerollButton = Rect(0, 80, 150, 75);
	quitButton = Rect(420, 350, 150, 75);

	levelButtons = new Rect[levelCount];
	levelImages = new Mat[levelCount + 9];
	for (int i = 0; i < 6; i++) {
		levelButtons[i] = Rect(49 + 94 * i, 340, 75, 75);
	}
	levelImages[0] = imread("images/puzzle_0.png", CV_LOAD_IMAGE_COLOR);
	levelImages[1] = imread("images/puzzle_1.png", CV_LOAD_IMAGE_COLOR);
	levelImages[2] = imread("images/puzzle_2.png", CV_LOAD_IMAGE_COLOR);
	levelImages[3] = imread("images/puzzle_3.png", CV_LOAD_IMAGE_COLOR);
	levelImages[4] = imread("images/puzzle_4.png", CV_LOAD_IMAGE_COLOR);
	levelImages[5] = imread("images/puzzle_5.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount] = imread("images/Button_Play.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount+1] = imread("images/Button_Levels.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 2] = imread("images/Button_Left_Arrow.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 3] = imread("images/Quit_Button.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 4] = imread("images/Button_Empty.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 5] = imread("images/Title.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 6] = imread("images/WinTitle.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 7] = imread("images/title_background.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 8] = imread("images/title_text.png", CV_LOAD_IMAGE_COLOR);
	levelImages[levelCount + 7].resize(cameraYRes, cameraXRes);
	levelImages[levelCount + 8].resize(cameraYRes, cameraXRes);
	finalImage = Mat(cameraYRes, cameraXRes, CV_8UC3, Scalar(0, 0, 0));
	// Setup callback function
	setMouseCallback(streamWindowName, callBackFunc);
}

void drawLevelSelection() {
	for (int i = 0; i < 6; i++) {
		drawButton(levelSelectionBg, levelButtons[i], i, Vec3b(200, 200, 200), "");
	}
	drawButton(levelSelectionBg, menuButton, levelCount + 2, Vec3b(200, 200, 200), "");
}

void updateUI() {
	// The canvas
	//canvas = Mat3b(finalImage.rows + button.height, finalImage.cols, Vec3b(0, 0, 0));

	// Draw the buttons
	drawButton(finalImage, startButton, levelCount, Vec3b(200, 200, 200), "");
	drawButton(finalImage, levelButton, levelCount + 1, Vec3b(200, 200, 200), "");
	drawButton(finalImage, menuButton, levelCount + 2, Vec3b(200, 200, 200), "");
	drawButton(finalImage, rerollButton, levelCount + 3, Vec3b(200, 200, 200), "");
	drawButton(finalImage, quitButton, levelCount + 4, Vec3b(200, 200, 200), "");

	for (int i = 0; i < levelCount; i++) {
		drawButton(finalImage, levelButtons[i], i, Vec3b(200, 200, 200), "Level " + i);
	}

	putText(finalImage, "You won", Point(150, 150), FONT_HERSHEY_SIMPLEX, 2, Scalar(128), 2);
}

void drawButton(Mat &display, Rect &button, int textureIndex, Vec3b &color, String buttonText) {
	display(button) = color;

	if (textureIndex != -1) {
		Mat texture = levelImages[textureIndex];
		resize(texture, texture, Size(button.width, button.height));
		texture.copyTo(display(button));
	}

	if (!buttonText.empty()) {
		putText(display(button), buttonText, Point(button.width*0.35, button.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));
	}
}

void drawGameRectangle() {
	Point2f point1 = Point2f(imageSize / 2, imageSize / 2), point2 = Point2f(imageSize / 2, finalImage.rows - imageSize / 2),
		point3 = Point2f(finalImage.cols - imageSize / 2, finalImage.rows - imageSize / 2), point4 = Point2f(finalImage.cols - imageSize / 2, imageSize / 2);
	vector<Point2f> points;
	points.push_back(point1);
	points.push_back(point2);
	points.push_back(point3);
	points.push_back(point4);
	line(finalImage, point1, point2, Scalar(255));

	line(finalImage, point2, point3, Scalar(255));

	line(finalImage, point3, point4, Scalar(255));

	line(finalImage, point4, point1, Scalar(255));
}

void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		Point point = Point(x, y);
		if (startButton.contains(point) && state == main_menu)
		{
			cout << "Start Clicked!" << endl;
			rectangle(finalImage(startButton), startButton, Scalar(0, 0, 255), 2);
			createPuzzle(5);
			state = playing;
		}
		else if (levelButton.contains(point) && state == main_menu) {
			rectangle(finalImage(levelButton), levelButton, Scalar(0, 0, 255), 2);
			state = level_select;
		}
		else if (menuButton.contains(point) && (state == playing || state == level_select || state == win)) {
			rectangle(finalImage(menuButton), menuButton, Scalar(0, 0, 255), 2);
			state = main_menu;
		}
		else if (rerollButton.contains(point) && state == playing) {
			rectangle(finalImage(rerollButton), rerollButton, Scalar(0, 0, 255), 2);
		}
		else if (quitButton.contains(point) && state == main_menu) {
			rectangle(finalImage(quitButton), quitButton, Scalar(0, 0, 255), 2);
			state = quit;
		}
		else {
			for (int i = 0; i < levelCount; i++) {
				if (levelButtons[i].contains(point) && state == level_select) {
					createPuzzle(i);
					state = playing;
					break;
				}
			}
		}
	}
	if (event == EVENT_LBUTTONUP)
	{
		//rectangle(finalImage, startButton, Scalar(200, 200, 200), 2);
	}

	waitKey(1);
}

int Sign(float x) {
	if (x < 0)
		return -1;
	else
		return 1;
}

//automatically win if you press spacebar
void cheatWin()
{
	if (waitKey(1) == 32)  //spacebar
	{
		winGame();
	}
}