//===============================================================================
// This file is part of the software Fast Box-Counting:
// https://www.ugr.es/~demiras/fbc
//
// Copyright(c)2022 University of Granada - SPAIN
//
// FOR RESEARCH PURPOSES ONLY.THE SOFTWARE IS PROVIDED "AS IS," AND THE
// UNIVERSITY OF GRANADA DO NOT MAKE ANY WARRANTY, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE, NOR DO THEY ASSUME ANY LIABILITY OR RESPONSIBILITY FOR THE USE
// OF THIS SOFTWARE.
//
// For more information see license.html
// ===============================================================================
//
// Authors: Juan Ruiz de Miras and Miguel �ngel Posadas, 2022
// Contact: demiras@ugr.es

// compile: nvcc -O3 -o test bcCUDA4D.cu bcCUDA3D.cu bcCUDA2D.cu bcCPU.cpp test.cpp

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring> 
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "bcCPU.h"

using namespace cv;
using namespace std;

char* getCmdOption(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

Mat readImage(string inFile, bool display = false) {
	Mat image = imread(inFile, IMREAD_GRAYSCALE);

	// Check if image loaded successfully
	if (image.empty()) {
		std::cerr << "Error loading image: " << inFile << std::endl;
		return image;
	}

	if (display) {
		std::cout << "Loaded image '" << inFile << "' of size " << image.size() << std::endl;

		imshow("Image of " + inFile, image);

		waitKey(0);
	}

	return image;
}

Mat detectEdges(Mat image, double threshold1 = 100, double threshold2 = 200) {
	Mat edges;
	Canny(image, edges, threshold1, threshold2);

	return edges;
}

void writeEdgeData(Mat edges, string outFile = "edges.txt") {
	ofstream fout(outFile);

	fout << edges.rows << '\n' << edges.cols << '\n';
	int colIndex = 0;

	for (auto pData = edges.datastart; pData != edges.dataend; pData++, colIndex++) {
		fout << ((int)*pData == 0 ? 0 : 1);
		if (colIndex == edges.cols - 1) {
			fout << '\n';
			colIndex = -1;
		}
	}
	fout.close();
}

int calculateMatrixSize(int cols, int rows) {
	int maxSize = rows > cols ? rows : cols;
	int matSize = 1;
	for (; matSize < maxSize; matSize *= 2);

	return matSize;
}

int calculateNumOfNs(int matSize) {
	return (log(matSize) / log(2) - 1);
}

unsigned char* createImageMatrix(Mat edges, int matSize, bool writeImageToFile = false) {
	unsigned char* imageMat = new unsigned char[matSize * matSize];

	for (int i = 0; i < matSize; i++) {
		for (int j = 0; j < matSize; j++) {
			imageMat[i * matSize + j] = 0;
		}
	}

	int rowIndex = 0, colIndex = 0;
	for (auto pData = edges.datastart; pData != edges.dataend; pData++, colIndex++) {
		imageMat[rowIndex * edges.cols + colIndex] = ((int)*pData == 0 ? 0 : 1);

		if (colIndex == edges.cols - 1) {
			colIndex = -1;
			rowIndex += 1;
		}
	}

	if (writeImageToFile) {
		ofstream fout("imageMat.txt");
		for (int i = 0; i < edges.rows; i++) {
			for (int j = 0; j < edges.cols; j++) {
				fout << (int)imageMat[i * edges.cols + j];
			}
			fout << '\n';
		}
		fout.close();
	}

	return imageMat;
}

unsigned int* createBoxArray(int numOfNs) {
	unsigned int* boxArray = new unsigned int[numOfNs];
	return boxArray;
}

chrono::duration<double> computeCPUAlgorithm(unsigned char* imageMat, unsigned int* boxArray, int matSize, int numOfNs) {
	chrono::time_point<chrono::system_clock> start, stop;

	for (int i = 0; i < numOfNs; i++) { 
		boxArray[i] = 0;
	}
	
	unsigned char* imageMatCopy = new unsigned char[matSize * matSize];
	memcpy(imageMatCopy, imageMat, sizeof(unsigned char) * matSize * matSize);
	
	start = std::chrono::system_clock::now();
	seqBC2D(imageMatCopy, matSize, boxArray);
	stop = std::chrono::system_clock::now();

	std::chrono::duration<double> timeCPU = stop - start;

	return timeCPU;
}

void writeResultsToFile(unsigned int* boxArray, int numOfNs, string outFile = "results.txt") {
	ofstream fout(outFile);
	// First write s, then n(s).
	for (int i = 0; i < numOfNs; i++) {
		fout << (2 << i) << ' ' << boxArray[i] << endl;
	}

	fout.close();
}

void clearResources(unsigned char* imageMat, unsigned int* boxArray) {
	delete[] imageMat;
	delete[] boxArray;
}

int main(int argc, char* argv[])
{
	char* filename = new char[0];
	string errors;
	int runCount = 10;
	if (cmdOptionExists(argv, argv + argc, "-f"))
	{
		filename = getCmdOption(argv, argv + argc, "-f");
	}
	else {
		errors += "File name must be provided!";
	}

	if (cmdOptionExists(argv, argv + argc, "-t"))
	{
		runCount = atoi(getCmdOption(argv, argv + argc, "-t"));
	}

	if (!errors.empty()) {
		ofstream fout("errors.txt");
		fout << errors;
		fout.close();
		return -1;
	}

	Mat image = readImage(filename);
	Mat edges = detectEdges(image);
	// writeEdgeData(edges);

	int matSize = calculateMatrixSize(edges.cols, edges.rows);
	int numOfNs = calculateNumOfNs(matSize);

	double* timeSpent = new double[runCount];
	for (int i = 0; i < runCount; i++) {
		unsigned char* imageMat = createImageMatrix(edges, matSize);
		unsigned int* boxArray = createBoxArray(numOfNs);

		chrono::duration<double> timeCPU = computeCPUAlgorithm(imageMat, boxArray, matSize, numOfNs);
		string resultsFile = string("results_").append(to_string(i)).append(string(".txt"));
		writeResultsToFile(boxArray, numOfNs, resultsFile);

		clearResources(imageMat, boxArray);
		timeSpent[i] = timeCPU.count();
		cout << "Time CPU box-counting of run " << (i + 1) << ": " << timeCPU.count() << " seconds." << endl;
	}

	ofstream fout("results_time.txt");
	for (int i = 0; i < runCount; i++) {
		fout << timeSpent[i] << endl;
	}
	fout.close();
	delete[] timeSpent;

	return 0;
}
