#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>

void run_gabor(const cv::Mat &img, const std::filesystem::path &out_dir);
void run_gabor_sweep(const cv::Mat &img, const std::filesystem::path &out_dir);
