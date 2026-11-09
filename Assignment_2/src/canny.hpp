#pragma once
#include <filesystem>
#include <string>
#include <opencv2/core/mat.hpp>

// image_name is appended to every output file, e.g. out/4_canny_otsu_rectification.png
void run_canny(const cv::Mat& img, const std::filesystem::path& out_dir,
               const std::string& image_name);
