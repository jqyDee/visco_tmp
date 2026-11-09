#pragma once

#include <filesystem>
#include <opencv2/core/mat.hpp>

void run_feature_bench(const cv::Mat &img, const std::filesystem::path &out_dir);

void run_feature_homography(const cv::Mat &img1, const cv::Mat &img2,
                            const std::filesystem::path &out_dir);
