#include "features.hpp"
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/geometry/3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>

struct Detector {
  std::string name;
  std::function<cv::Ptr<cv::Feature2D>()> create;
};

static std::vector<Detector> detectors() {
  return {
      {"ORB", [] { return cv::ORB::create(2000); }},
      {"AKAZE", [] { return cv::xfeatures2d::AKAZE::create(); }},
      {"BRISK", [] { return cv::xfeatures2d::BRISK::create(); }},
  };
}

struct Transform {
  std::string kind; // "rotation", "scale", "tilt"
  double value;     // degrees, scale factor, or tilt fraction
};

static cv::Mat make_homography(const Transform &t, cv::Size size) {
  const float w = static_cast<float>(size.width);
  const float h = static_cast<float>(size.height);

  if (t.kind == "tilt") {
    const float d = static_cast<float>(t.value) * w;
    const std::vector<cv::Point2f> src = {{0, 0}, {w, 0}, {w, h}, {0, h}};
    const std::vector<cv::Point2f> dst = {{d, 0}, {w - d, 0}, {w, h}, {0, h}};
    cv::Mat H = cv::getPerspectiveTransform(src, dst);
    H.convertTo(H, CV_64F);
    return H;
  }

  if (t.kind == "rotation" || t.kind == "scale") {
    const double angle = t.kind == "rotation" ? t.value : 0.0;
    const double scale = t.kind == "scale" ? t.value : 1.0;
    const cv::Point2f center(w / 2.0f, h / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat H;
    cv::vconcat(M, cv::Mat(cv::Matx13d(0, 0, 1)), H);
    return H;
  }

  CV_Error(cv::Error::StsBadArg, "unknown transform kind: " + t.kind);
}

// ---------------------------------------------------------------------------
// Matching + evaluation
// ---------------------------------------------------------------------------

static std::vector<cv::DMatch>
match_ratio(const cv::Mat &desc1, const cv::Mat &desc2, float ratio = 0.8f) {
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc1, desc2, knn, 2);

  std::vector<cv::DMatch> kept;
  for (const auto &pair : knn) {
    if (pair.size() == 2 && pair[0].distance < ratio * pair[1].distance) {
      kept.push_back(pair[0]);
    }
  }
  return kept;
}

static int count_correct(const std::vector<cv::KeyPoint> &kp1,
                         const std::vector<cv::KeyPoint> &kp2,
                         const std::vector<cv::DMatch> &matches,
                         const cv::Mat &H, double max_px = 3.0) {
  if (matches.empty())
    return 0;

  std::vector<cv::Point2f> kp1Matches;
  for (const auto match : matches) {
    cv::Point2f p = kp1[match.queryIdx].pt;
    kp1Matches.push_back(p);
  }
  std::vector<cv::Point2f> projected;
  cv::perspectiveTransform(kp1Matches, projected, H);

  int correct = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    cv::Point2f claimed = kp2[matches[i].trainIdx].pt;
    double divergence = cv::norm(projected[i] - claimed);
    if (divergence < max_px) {
      correct++;
    }
  }
  return correct;
}

struct Result {
  int kp1 = 0, kp2 = 0, kept = 0, correct = 0;
  double ms = 0.0;
};

static Result evaluate(const Detector &det, const cv::Mat &gray1,
                       const cv::Mat &gray2, const cv::Mat &H) {
  Result r;
  auto feat = det.create();
  std::vector<cv::KeyPoint> kp1, kp2;
  cv::Mat desc1, desc2;

  auto t0 = std::chrono::steady_clock::now();
  feat->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
  feat->detectAndCompute(gray2, cv::noArray(), kp2, desc2);
  std::vector<cv::DMatch> matches;
  if (!desc1.empty() && !desc2.empty())
    matches = match_ratio(desc1, desc2);
  auto t1 = std::chrono::steady_clock::now();

  r.kp1 = static_cast<int>(kp1.size());
  r.kp2 = static_cast<int>(kp2.size());
  r.kept = static_cast<int>(matches.size());
  r.correct = count_correct(kp1, kp2, matches, H);
  r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return r;
}

// ---------------------------------------------------------------------------
// A + B: benchmark
// ---------------------------------------------------------------------------

void run_feature_bench(const cv::Mat &img,
                       const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);
  cv::Mat gray1;
  cv::cvtColor(img, gray1, cv::COLOR_BGR2GRAY);

  const std::vector<Transform> transforms = {
      {"rotation", 0.0},  {"rotation", 30.0}, {"rotation", 60.0},
      {"rotation", 90.0}, {"scale", 0.25},    {"scale", 0.5},
      {"scale", 2.0},     {"scale", 3.0},     {"tilt", 0.2},
  };

  std::ofstream csv(out_dir / "bench.csv");
  csv << "detector,kind,value,kp1,kp2,kept,correct,precision,ms\n";

  // warm detectors: first call pays one-off setup cost, don't time it
  for (const Detector &det : detectors()) {
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    det.create()->detectAndCompute(gray1, cv::noArray(), kp, desc);
  }

  for (const Transform &t : transforms) {
    cv::Mat H = make_homography(t, gray1.size());
    cv::Mat gray2;
    cv::warpPerspective(gray1, gray2, H, gray1.size());

    std::string tag = t.kind + "_" + std::to_string(t.value);
    cv::imwrite((out_dir / ("bench_" + tag + ".png")).string(), gray2);

    for (const Detector &det : detectors()) {
      Result r = evaluate(det, gray1, gray2, H);
      double precision = r.kept > 0 ? double(r.correct) / r.kept : 0.0;
      csv << det.name << ',' << t.kind << ',' << t.value << ',' << r.kp1 << ','
          << r.kp2 << ',' << r.kept << ',' << r.correct << ',' << precision
          << ',' << r.ms << '\n';
      std::cout << det.name << "\t" << tag << "\tkept=" << r.kept
                << "\tcorrect=" << r.correct << "\tprec=" << precision << "\t"
                << r.ms << " ms\n";
    }
  }
  std::cout << "wrote " << (out_dir / "bench.csv") << '\n';
}

// ---------------------------------------------------------------------------
// Homography from real photos
// ---------------------------------------------------------------------------

void run_feature_homography(const cv::Mat &img1, const cv::Mat &img2,
                            const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);
  cv::Mat gray1, gray2;
  cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

  // using ORB because it was fastest and is free to use
  auto feat = cv::ORB::create(5000);
  std::vector<cv::KeyPoint> kp1, kp2;
  cv::Mat desc1, desc2;
  feat->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
  feat->detectAndCompute(gray2, cv::noArray(), kp2, desc2);
  std::vector<cv::DMatch> matches = match_ratio(desc1, desc2);
  std::cout << "kp1=" << kp1.size() << " kp2=" << kp2.size()
            << " kept=" << matches.size() << '\n';

  if (matches.size() < 4) {
    std::cerr
        << "need >= 4 matches for a homography (implement match_ratio?)\n";
    return;
  }

  // estimate H with RANSAC: 3 px reprojection threshold, mask marks inliers
  std::vector<cv::Point2f> pts1, pts2;
  for (const auto &m : matches) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
  }
  cv::Mat mask;
  cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);
  int inliers = H.empty() ? 0 : cv::countNonZero(mask);
  std::cout << "RANSAC inliers: " << inliers << " / " << matches.size() << '\n';

  if (H.empty()) {
    std::cerr << "findHomography failed: RANSAC found no valid model\n";
    return;
  }
  std::cout << "H =\n" << H << '\n';

  // mask (Mat of 0/1) -> vector<char> for drawMatches; inverted for outliers
  std::vector<char> inlier_mask(mask.begin<uchar>(), mask.end<uchar>());
  std::vector<char> outlier_mask(inlier_mask.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i)
    outlier_mask[i] = !inlier_mask[i];

  // inliers green
  cv::Mat vis;
  cv::drawMatches(img1, kp1, img2, kp2, matches, vis, cv::Scalar(0, 255, 0),
                  cv::Scalar::all(-1), inlier_mask,
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  // outliers red, drawn on top of the same image
  cv::drawMatches(img1, kp1, img2, kp2, matches, vis, cv::Scalar(0, 0, 255),
                  cv::Scalar::all(-1), outlier_mask,
                  cv::DrawMatchesFlags::DRAW_OVER_OUTIMG |
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // warp photo 1 into photo 2's view, then blend: aligned plane = sharp,
  // misalignment = double edges
  cv::Mat warped, overlay;
  cv::warpPerspective(img1, warped, H, img2.size());
  cv::addWeighted(warped, 0.5, img2, 0.5, 0.0, overlay);

  // outline of photo 1's borders in photo 2: shows where H maps the frame
  const float w = static_cast<float>(img1.cols);
  const float h = static_cast<float>(img1.rows);
  std::vector<cv::Point2f> frame = {{0, 0}, {w, 0}, {w, h}, {0, h}};
  std::vector<cv::Point2f> frame_in_2;
  cv::perspectiveTransform(frame, frame_in_2, H);
  std::vector<cv::Point> poly(frame_in_2.begin(), frame_in_2.end());
  cv::Mat outlined = img2.clone();
  cv::polylines(outlined, poly, true, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

  cv::imwrite((out_dir / "5c_matches.jpg").string(), vis);
  cv::imwrite((out_dir / "5c_warped.jpg").string(), warped);
  cv::imwrite((out_dir / "5c_overlay.jpg").string(), overlay);
  cv::imwrite((out_dir / "5c_frame.jpg").string(), outlined);
  std::cout << "wrote 5c_{matches,warped,overlay,frame}.jpg to " << out_dir
            << '\n';
}
