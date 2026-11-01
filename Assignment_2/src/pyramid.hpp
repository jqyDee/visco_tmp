#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>

void run_pyramid(const cv::Mat &img8u, const std::filesystem::path &out_dir);
