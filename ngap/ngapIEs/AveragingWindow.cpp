#include "AveragingWindow.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AveragingWindow::AveragingWindow(){}  

	AveragingWindow::~AveragingWindow(){} 

	void AveragingWindow::setAveragingWindow(long value)
	{
		averagingwindow = value;
	}
	bool AveragingWindow::getAveragingWindow(long &value)
	{
		value = averagingwindow;
		
		return true;
	}
	
	bool AveragingWindow::encode2AveragingWindow(Ngap_AveragingWindow_t *averagingWindow)
	{
		*averagingWindow = averagingwindow;
	
		return true;
	}
	bool AveragingWindow::decodefromAveragingWindow(Ngap_AveragingWindow_t *averagingWindow)
	{
		averagingwindow = *averagingWindow;
		
		return true;
	}
}
