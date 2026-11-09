#include "rectification.hpp"
#include "pyramid.hpp"
#include "gabor.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include <opencv2/imgproc.hpp>

static int usage(const char *prog) {
  std::cerr << "usage:\n"
            << "  " << prog
            << " rectify <image> [x_tl y_tl x_tr y_tr x_br y_br x_bl y_bl]\n"
            << "  " << prog << " pyramid <image> [max_side]\n"
            << "  " << prog << " gabor <image> [max_side]\n"
            << "  " << prog << " gabor-sweep <image> [max_side]\n";
  return EXIT_FAILURE;
}

// Shrink img so its longer side is at most max_side (never enlarges).
static void downscale(cv::Mat &img, const char *max_side) {
  const double scale = std::stod(max_side) / std::max(img.cols, img.rows);
  if (scale < 1.0)
    cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
}

int main(int argc, char **argv) {
  if (argc < 3)
    return usage(argv[0]);

  std::string cmd = argv[1];
  cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "could not read " << argv[2] << '\n';
    return EXIT_FAILURE;
  }

  if (cmd == "rectify") {
    Quad corners;
    if (argc == 11) {
      for (int i = 0; i < 4; ++i)
        corners[i] = {std::stof(argv[3 + 2 * i]), std::stof(argv[4 + 2 * i])};
    } else if (argc == 3) {
      corners = select_corners(img);
    } else {
      return usage(argv[0]);
    }
    run_rectify(img, corners, "out");
    return EXIT_SUCCESS;
  }

  if (cmd == "pyramid" || cmd == "gabor" || cmd == "gabor-sweep") {
    if (argc > 4)
      return usage(argv[0]);
    if (argc == 4)
      downscale(img, argv[3]);

    if (cmd == "pyramid")
      run_pyramid(img, "out");
    else if (cmd == "gabor")
      run_gabor(img, "out");
    else
      run_gabor_sweep(img, "out");
    return EXIT_SUCCESS;
  }

  return usage(argv[0]);
}
