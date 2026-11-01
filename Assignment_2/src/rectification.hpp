#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core/types.hpp>

using Quad = std::array<cv::Point2f, 4>;

Quad select_corners(const cv::Mat &img);

void run_rectify(const cv::Mat &img, const Quad &pts,
                 const std::filesystem::path &out_dir);
