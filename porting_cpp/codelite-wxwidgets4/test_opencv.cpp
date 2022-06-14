#include <opencv2/opencv.hpp>

void tst_opencv()
{
    cv::Mat inImg = cv::imread("/home/thermalview/Desktop/ThermalView/saved_images/2020_11_23__13_31_03/2020_11_23__13_31_03_FULL_COLOR_SOURCE.jpg", 0);
    if (inImg.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }
    cv::Mat outImg = cv::Mat();
    cv::resize(inImg, outImg, cv::Size(), 0.75, 0.75);
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "Display window", I ); 
    //cv::waitKey(0);

}