#include "gabor.hpp"
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

struct GaborParams {
  int ksize = 31;
  double sigma = 4.0;
  double lambda = 10.0;
  double gamma = 0.5;
  double psi = 0.0;
};

static constexpr std::array<double, 4> thetas = {0.0, 45.0, 90.0, 135.0};

static cv::Mat to_gray_float(const cv::Mat &img) {
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F, 1.0 / 255.0);
  return gray;
}

static std::vector<cv::Mat> make_kernels(const GaborParams &p) {
  std::vector<cv::Mat> kernels;
  for (double deg : thetas) {
    cv::Mat k = cv::getGaborKernel(cv::Size(p.ksize, p.ksize), p.sigma,
                                   deg * CV_PI / 180.0, p.lambda, p.gamma,
                                   p.psi, CV_32F);
    k -= cv::mean(k)[0]; // zero mean: flat areas give 0
    kernels.push_back(k);
  }
  return kernels;
}

static std::vector<cv::Mat> filter_all(const cv::Mat &gray,
                                       const std::vector<cv::Mat> &kernels) {
  std::vector<cv::Mat> responses;
  for (const cv::Mat &k : kernels) {
    cv::Mat r;
    cv::filter2D(gray, r, CV_32F, k);
    responses.push_back(r);
  }
  return responses;
}

static cv::Mat combine_abs(const std::vector<cv::Mat> &responses) {
  cv::Mat combined = cv::abs(responses[0]);
  for (size_t i = 1; i < responses.size(); i++) {
    cv::Mat a = cv::abs(responses[i]);
    cv::max(combined, a, combined);
  }
  return combined;
}

static cv::Mat to_u8(const cv::Mat &m) {
  cv::Mat view;
  cv::normalize(m, view, 0, 255, cv::NORM_MINMAX, CV_8U);
  return view;
}

static cv::Mat kernel_beside(const cv::Mat &kernel, const cv::Mat &result_u8) {
  cv::Mat kview = to_u8(kernel);
  cv::resize(kview, kview, cv::Size(result_u8.rows, result_u8.rows), 0, 0,
             cv::INTER_NEAREST);
  cv::Mat pair;
  cv::hconcat(kview, result_u8, pair);
  return pair;
}

// Report figures as JPEG (quality 90, as in exercise 1).
static void save(const std::filesystem::path &out_dir, const std::string &name,
                 const cv::Mat &img) {
  cv::imwrite((out_dir / (name + ".jpg")).string(), img,
              {cv::IMWRITE_JPEG_QUALITY, 90});
}

void run_gabor(const cv::Mat &img, const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);
  cv::Mat gray = to_gray_float(img);

  std::vector<cv::Mat> kernels = make_kernels(GaborParams{});
  std::vector<cv::Mat> responses = filter_all(gray, kernels);

  cv::Mat raw = responses[0].clone();
  for (size_t i = 0; i < responses.size(); i++) {
    cv::max(raw, responses[i], raw);
    std::string n = std::to_string(static_cast<int>(thetas[i]));
    save(out_dir, "3_gabor_pair_" + n,
         kernel_beside(kernels[i], to_u8(responses[i])));
  }
  save(out_dir, "3_gabor_combined_raw", to_u8(raw));

  cv::Mat gabor = to_u8(combine_abs(responses));
  save(out_dir, "3_gabor_combined", gabor);

  cv::Mat gx, gy, mag, cmp;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, mag);
  cv::Mat sobel = to_u8(mag);
  save(out_dir, "3_sobel_mag", sobel);
  cv::hconcat(gabor, sobel, cmp);
  save(out_dir, "3_gabor_vs_sobel", cmp);
}

// Defaults with one field changed per value; saves 0 deg kernel + result
// for each value and all results side by side as one strip.
static void sweep(const cv::Mat &gray, const std::filesystem::path &out_dir,
                  const std::string &name, std::initializer_list<double> values,
                  void (*set)(GaborParams &, double)) {
  std::vector<cv::Mat> strip;
  for (double v : values) {
    GaborParams p;
    set(p, v);
    std::vector<cv::Mat> kernels = make_kernels(p);
    cv::Mat result = to_u8(combine_abs(filter_all(gray, kernels)));

    std::ostringstream label;
    label << v;
    save(out_dir, "3_sweep_" + name + "_" + label.str(),
         kernel_beside(kernels[0], result));
    strip.push_back(result);
  }
  cv::Mat strip_img;
  cv::hconcat(strip, strip_img);
  save(out_dir, "3_sweep_" + name, strip_img);
}

void run_gabor_sweep(const cv::Mat &img, const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);
  cv::Mat gray = to_gray_float(img);

  // ksize 61 for every sigma, otherwise sigma = 8 is cut off (needs ~6 * sigma)
  sweep(gray, out_dir, "sigma", {2, 4, 8},
        [](GaborParams &p, double v) { p.sigma = v; p.ksize = 61; });
  sweep(gray, out_dir, "lambda", {5, 10, 20},
        [](GaborParams &p, double v) { p.lambda = v; });
  sweep(gray, out_dir, "gamma", {0.25, 0.5, 1},
        [](GaborParams &p, double v) { p.gamma = v; });
  sweep(gray, out_dir, "psi", {0, 90}, // degrees
        [](GaborParams &p, double v) { p.psi = v * CV_PI / 180.0; });
  sweep(gray, out_dir, "ksize", {15, 31, 61},
        [](GaborParams &p, double v) { p.ksize = static_cast<int>(v); });
}
