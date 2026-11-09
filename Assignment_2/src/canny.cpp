#include "canny.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

// Save as out_dir/4_<name>_<image_name>.png, e.g. out/4_canny_hand_tuned_rectification.png
static void save(const cv::Mat& img8u, const std::filesystem::path& out_dir,
                 const std::string& name, const std::string& image_name) {
    cv::imwrite((out_dir / ("4_" + name + "_" + image_name + ".png")).string(), img8u);
}

// White text on a black box in the top-left corner, readable on any background.
static void label(cv::Mat& img, const std::string& text) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    cv::rectangle(img, cv::Rect(0, 0, ts.width + 20, ts.height + baseline + 20), cv::Scalar(0), cv::FILLED);
    cv::putText(img, text, cv::Point(10, 10 + ts.height), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
}

void run_canny(const cv::Mat& src, const std::filesystem::path& out_dir,
               const std::string& image_name) {

    std::filesystem::create_directories(out_dir);

    cv::Mat src_gray, blurred, otsu_bin;

    //convert to gray
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    //apply gaussian blur
    cv::GaussianBlur(src_gray, blurred, cv::Size(5, 5), 1.4); //sigma = 1.4: width of gaussian in px

    // Otsu on the grey histogram -> high threshold
    double otsu = cv::threshold(blurred, otsu_bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int ratio = 2; // could also be 3 as mentioned in the slides
    double highThreshold = otsu;
    double lowThreshold = otsu / ratio;

    // gradients, non-max suppression, double threshold and hysteresis via cv::Canny
    int kernel_size = 3;
    cv::Mat edges_otsu;
    cv::Canny(blurred, edges_otsu, lowThreshold, highThreshold, kernel_size);

    // save with otsu thresholds
    save(edges_otsu, out_dir, "canny_otsu", image_name);
    std::cout << image_name << ": Otsu = " << otsu
              << "  ->  low = " << lowThreshold << ", high = " << highThreshold << '\n';

    //save with hand-tuned thresholds
    double hand_low = 30, hand_high = 90;
    cv::Mat edges_hand;
    cv::Canny(blurred, edges_hand, hand_low, hand_high, kernel_size);
    save(edges_hand, out_dir, "canny_hand_tuned", image_name);
    std::cout << "  ->  hand-tuned low = " << hand_low << ", hand-tuned high = " << hand_high << '\n';

    // number of edge pixels for the report: Canny output is binary (edge = 255, rest = 0),
    // so countNonZero counts the edge pixels
    int total = edges_otsu.rows * edges_otsu.cols;
    int n_otsu = cv::countNonZero(edges_otsu);
    int n_hand = cv::countNonZero(edges_hand);
    std::cout << "  ->  edge pixels: otsu = " << n_otsu << " (" << 100.0 * n_otsu / total << " %)"
              << ", hand-tuned = " << n_hand << " (" << 100.0 * n_hand / total << " %)\n";

    //-------------
    // Task 4.2, the sobel and laplacian edge detectors
    //-------------

    // Sobel
    // same as in lecture slides
    int ddepth = CV_16S;
    int scale_s = 1, delta = 0;
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, sobel;


    //Gradient X (input_img, output, output_type, dx = 1, dy = 0, kernel_size = 3, scale (mult), delta (add))
    cv::Sobel(blurred, grad_x, ddepth, 1, 0, 3, scale_s, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    //Gradient Y (input_img, output, output_type, dx = 0, dy = 1, kernel_size = 3, scale (mult), delta (add))
    cv::Sobel(blurred, grad_y, ddepth, 0, 1, 3, scale_s, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // weigh both x and y sobels 50:50
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);

    save(sobel, out_dir, "sobel", image_name);


    // Laplacian
    // also same as in lecture slides
    cv::Mat lap, abs_lap;
    cv::Laplacian(blurred, lap, ddepth, kernel_size, scale_s, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(lap, abs_lap, 4);

    save(abs_lap, out_dir, "laplacian", image_name);

    //-------------
    // Figures for the report: labelled panels side by side
    //-------------
    
    
    // threshold values as text without trailing zeros, e.g. "55" or "64.5"
    auto num = [](double v) {
        std::string s = std::to_string(v);
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
        return s;
    };

    // copies, so label() does not draw into the images saved above
    cv::Mat p_sobel = sobel.clone(), p_lap = abs_lap.clone(),
            p_hand = edges_hand.clone(), p_otsu = edges_otsu.clone();
    label(p_sobel, "Sobel");
    label(p_lap, "Laplacian (x4)");
    label(p_hand, "Canny hand " + num(hand_low) + "/" + num(hand_high));
    label(p_otsu, "Canny Otsu " + num(lowThreshold) + "/" + num(highThreshold));

    // grey input on its own (the reader has to see what the detectors got)
    save(src_gray, out_dir, "input", image_name);

    // 4.2 overview: 2 x 2 grid of the four detectors (all CV_8UC1, same size)
    cv::Mat row1, row2, grid;
    cv::hconcat(p_sobel, p_lap, row1);
    cv::hconcat(p_hand, p_otsu, row2);
    cv::vconcat(row1, row2, grid);
    save(grid, out_dir, "compare", image_name);

    // hand-tuned vs. Otsu next to each other
    cv::Mat hand_vs_otsu;
    cv::hconcat(p_hand, p_otsu, hand_vs_otsu);
    save(hand_vs_otsu, out_dir, "hand_vs_otsu", image_name);
}