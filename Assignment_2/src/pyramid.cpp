#include "pyramid.hpp"
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

const cv::Mat k_gauss = cv::Mat_<float>({1, 4, 6, 4, 1}) / 16.0f;
static int levels_below = 4;
static float laplacian_gain = 2.0f;

cv::Mat blur(const cv::Mat &img, float gain) {
  cv::Mat k = k_gauss * gain, out;
  cv::sepFilter2D(img, out, -1, k, k, cv::Point(-1, -1), 0);
  return out;
}

cv::Mat reduce(const cv::Mat &img) {
  cv::Mat blurred = blur(img, 1.0f);
  cv::Mat out((img.rows + 1) / 2, (img.cols + 1) / 2, img.type());
  for (int y = 0; y < out.rows; y++) {
    for (int x = 0; x < out.cols; x++) {
      out.at<cv::Vec3f>(y, x) = img.at<cv::Vec3f>(2 * y, 2 * x);
    }
  }
  return out;
}

cv::Mat expand(const cv::Mat &img, cv::Size size) {
  cv::Mat up = cv::Mat::zeros(size, img.type());
  for (int y = 0; y < img.rows && 2 * y < size.height; ++y)
    for (int x = 0; x < img.cols && 2 * x < size.width; ++x)
      up.at<cv::Vec3f>(2 * y, 2 * x) = img.at<cv::Vec3f>(y, x);

  // Only every 4th pixel is non-zero, so the kernel gets gain 2 per axis.
  return blur(up, 2.0f);
}

cv::Mat to_laplacian_view(const cv::Mat &lap) {
  cv::Mat view;
  lap.convertTo(view, CV_8UC3, laplacian_gain, 128);
  return view;
}

// Original on the left, the smaller levels stacked on the right
cv::Mat mosaic(const std::vector<cv::Mat> &levels) {
  int w = levels[0].cols + (levels.size() > 1 ? levels[1].cols : 0);
  cv::Mat canvas(levels[0].rows, w, CV_8UC3, cv::Scalar::all(255));
  levels[0].copyTo(canvas(cv::Rect(0, 0, levels[0].cols, levels[0].rows)));

  int y = 0;
  const int x = levels[0].cols;
  for (size_t i = 1; i < levels.size(); ++i) {
    levels[i].copyTo(canvas(cv::Rect(x, y, levels[i].cols, levels[i].rows)));
    y += levels[i].rows;
  }
  return canvas;
}

// The same center region cut from every level, scaled back up to the size
// of the level-0 crop (nearest neighbour, so one pixel = one block).
cv::Mat zoomStrip(const std::vector<cv::Mat> &levels) {
  int side = std::min(levels[0].rows, levels[0].cols) / 4;
  side -= side % (1 << (levels.size() - 1)); // divisible on every level
  cv::Point center(levels[0].cols / 2, levels[0].rows / 2);

  const int gap = 8;
  cv::Mat strip(side, levels.size() * (side + gap) - gap, CV_8UC3,
                cv::Scalar::all(255));
  for (size_t i = 0; i < levels.size(); ++i) {
    int s = side >> i;
    cv::Rect roi((center.x >> i) - s / 2, (center.y >> i) - s / 2, s, s);
    cv::Mat up;
    cv::resize(levels[i](roi), up, cv::Size(side, side), 0, 0,
               cv::INTER_NEAREST);
    up.copyTo(strip(cv::Rect(i * (side + gap), 0, side, side)));
  }
  return strip;
}

std::vector<cv::Mat> gaussian_pyramid(const cv::Mat &img, int levels_below) {
  std::vector<cv::Mat> gauss = {img};
  for (int i = 0; i < levels_below; i++) {
    gauss.push_back(reduce(gauss.back()));
  }
  return gauss;
}

std::vector<cv::Mat> laplacian_pyramid(const std::vector<cv::Mat> &gauss) {
  std::vector<cv::Mat> laplacian;
  for (size_t i = 0; i + 1 < gauss.size(); i++) {
    laplacian.push_back(gauss[i] - expand(gauss[i + 1], gauss[i].size()));
  }
  laplacian.push_back(gauss.back());
  return laplacian;
}

cv::Mat reconstruct(const std::vector<cv::Mat> &lap) {
  cv::Mat img = lap.back();
  for (int i = static_cast<int>(lap.size()) - 2; i >= 0; --i)
    img = lap[i] + expand(img, lap[i].size());
  return img;
}

void run_pyramid(const cv::Mat &img8u, const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);

  // convert the image into float to show negative laplacian values
  cv::Mat img;
  img8u.convertTo(img, CV_32FC3);

  std::vector<cv::Mat> gauss = gaussian_pyramid(img, levels_below);
  std::vector<cv::Mat> laplacian = laplacian_pyramid(gauss);
  cv::Mat rec = reconstruct(laplacian);

  std::vector<cv::Mat> gauss_view, laplacian_view;
  for (size_t i = 0; i < gauss.size(); i++) {
    cv::Mat g_convert, l_convert;
    gauss[i].convertTo(g_convert, CV_8UC3);
    l_convert = (i + 1 < laplacian.size()) ? to_laplacian_view(laplacian[i])
                                           : g_convert;
    gauss_view.push_back(g_convert);
    laplacian_view.push_back(l_convert);

    double lapMin, lapMax;
    cv::minMaxLoc(laplacian[i].reshape(1), &lapMin, &lapMax);
    std::cout << "level " << i << ": " << gauss[i].cols << " x "
              << gauss[i].rows << "   L range [" << lapMin << ", " << lapMax
              << "]\n";

    std::string n = std::to_string(i);
    cv::imwrite((out_dir / ("2_gauss_" + n + ".png")).string(), g_convert);
    cv::imwrite((out_dir / ("2_lap_" + n + ".png")).string(), l_convert);
  }

  cv::imwrite((out_dir / "2_gauss_pyramid.png").string(), mosaic(gauss_view));
  cv::imwrite((out_dir / "2_lap_pyramid.png").string(), mosaic(laplacian_view));
  cv::imwrite((out_dir / "2_gauss_zoom.png").string(), zoomStrip(gauss_view));
  cv::imwrite((out_dir / "2_lap_zoom.png").string(), zoomStrip(laplacian_view));

  cv::Mat rec8u;
  rec.convertTo(rec8u, CV_8UC3);
  cv::imwrite((out_dir / "2_reconstructed.png").string(), rec8u);

  double errFloat = cv::norm(rec, img, cv::NORM_INF);
  double err8u = cv::norm(rec8u, img8u, cv::NORM_INF);
  std::cout << "Laplacian view: 128 + " << laplacian_gain << " * L\n"
            << "max |reconstructed - original| (float): " << errFloat
            << "\nmax |reconstructed - original| (8 bit): " << err8u
            << "\nwrote " << out_dir / "2_*.png" << '\n';
}
