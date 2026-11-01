#include "rectification.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

const char *k_corner_names[4] = {
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
};

struct ClickState {
  std::vector<cv::Point2f> pts;
  double scale;
};

void onMouse(int event, int x, int y, int, void *state_in) {
  ClickState *state = static_cast<ClickState *>(state_in);

  if (event == cv::EVENT_LBUTTONDOWN && state->pts.size() < 4) {
    // points are stored as actual image coordinates
    cv::Point2f pt = cv::Point2f(x * state->scale, y * state->scale);
    state->pts.push_back(pt);
    std::cout << "  " << k_corner_names[state->pts.size() - 1] << ": ("
              << state->pts.back().x << ", " << state->pts.back().y << ")"
              << std::endl;
  }
}

Quad select_corners(const cv::Mat &img) {
  const int max_side_pxls = 1200;

  // set the scale to 1.0 (if image is smaller than 1200px) or to the scale
  // value derived by dividing the longest side with our `max_side_pxls`
  ClickState state{{},
                   std::max(1.0, std::max(img.cols, img.rows) /
                                     static_cast<double>(max_side_pxls))};

  cv::Mat view;
  cv::resize(img, view, cv::Size(), 1.0 / state.scale, 1.0 / state.scale,
             cv::INTER_AREA);

  const std::string win = "Click TL, TR, BR, BL (`r` reset, `enter` submit)";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(win, onMouse, &state);

  std::cout << "Click the 4 corners in this order: " << k_corner_names[0]
            << ", " << k_corner_names[1] << ", " << k_corner_names[2] << ", "
            << k_corner_names[3] << std::endl;

  while (true) {
    cv::Mat canvas = view.clone();

    std::optional<cv::Point2f> p_wind_bef;
    for (const cv::Point2f &p_img : state.pts) {
      // convert back to window coordinates
      cv::Point2f p_wind = p_img * (1.0 / state.scale);
      cv::circle(canvas, p_wind, 5, {0, 0, 255}, cv::FILLED);

      // draw line between the last 2 points
      if (p_wind_bef) {
        cv::line(canvas, *p_wind_bef, p_wind, {0, 255, 0}, 2, cv::LINE_AA);
      }
      p_wind_bef = p_wind;

      // draw line between point 4 and the first
      if (state.pts.size() == 4) {
        cv::line(canvas, state.pts[3] * (1.0 / state.scale),
                 state.pts[0] * (1.0 / state.scale), {0, 255, 0}, 2,
                 cv::LINE_AA);
      }
    }

    cv::imshow(win, canvas);

    // handle keys
    int key = cv::waitKey(20);
    if (key == 'r') {
      state.pts.clear();
    }
    if ((key == '\r' || key == '\n') && state.pts.size() == 4) {
      break;
    }
    if (key == '\x1b') { // esc key
      std::exit(EXIT_FAILURE);
    }
  }

  cv::destroyWindow(win);

  Quad q;
  std::copy(state.pts.begin(), state.pts.end(), q.begin());
  std::cout << "Re-run without the pts selection:  rectify <image>";
  for (auto &p : q) {
    std::cout << ' ' << p.x << ' ' << p.y;
  }
  std::cout << '\n';
  return q;
}

float dist(const cv::Point2f p1, const cv::Point2f p2) {
  return cv::norm(p1 - p2);
}

int thickness(const cv::Mat &img) {
  return std::max(2, std::max(img.cols, img.rows) / 400);
}

void draw_quad(cv::Mat &img, const Quad &q, cv::Scalar color) {
  for (int i = 0; i < 4; ++i)
    cv::line(img, q[i], q[(i + 1) % 4], color, thickness(img), cv::LINE_AA);
}

void draw_cross(cv::Mat &img, cv::Point2f p, cv::Scalar color) {
  int t = thickness(img), r = 6 * t;
  cv::drawMarker(img, p, color, cv::MARKER_CROSS, 2 * r, t, cv::LINE_AA);
  cv::circle(img, p, r, color, t, cv::LINE_AA);
}

// Saves a report figure: downscaled to 1600 px on the longest side, as JPEG.
// Only the saved file shrinks, all measurements stay in full-resolution pixels.
void save_figure(const std::filesystem::path &path, const cv::Mat &img) {
  const int max_side = 1600;
  const double scale = std::min(
      1.0, max_side / static_cast<double>(std::max(img.cols, img.rows)));
  cv::Mat small;
  cv::resize(img, small, cv::Size(), scale, scale, cv::INTER_AREA);
  cv::imwrite(path.string(), small, {cv::IMWRITE_JPEG_QUALITY, 90});
}

// Angle in degrees between the directions of two segments (0 = parallel).
double angle_between(cv::Point2f a0, cv::Point2f a1, cv::Point2f b0,
                     cv::Point2f b1) {
  cv::Point2f u = a1 - a0, v = b1 - b0;
  double c = std::abs(u.dot(v)) / (cv::norm(u) * cv::norm(v));
  return std::acos(std::min(1.0, c)) * 180.0 / CV_PI;
}

void run_rectify(const cv::Mat &img, const Quad &pts,
                 const std::filesystem::path &out_dir) {
  std::filesystem::create_directories(out_dir);

  const auto &[tl, tr, br, bl] = pts;

  float w = std::max(dist(tl, tr), dist(bl, br));
  float h = std::max(dist(tl, bl), dist(tr, br));
  Quad dst = {cv::Point2f(0, 0), cv::Point2f(w - 1, 0),
              cv::Point2f(w - 1, h - 1), cv::Point2f(0, h - 1)};

  // 1.1 Perspective rectification
  cv::Mat H = cv::getPerspectiveTransform(pts.data(), dst.data());
  cv::Mat perspective;
  cv::warpPerspective(img, perspective, H,
                      cv::Size(std::lround(w), std::lround(h)));
  save_figure(out_dir / "1_perspective.jpg", perspective);

  // 1.2 Affine rectification
  // add a padding of 0.15f to the dst points
  const float m = 0.15f * std::max(w, h);
  cv::Point2f off(m, m);
  cv::Point2f src_tri[3] = {tl, tr, bl};
  cv::Point2f dst_tri[3] = {dst[0] + off, dst[1] + off, dst[3] + off};
  cv::Mat A = cv::getAffineTransform(src_tri, dst_tri);

  // check point bottom right with the affine transform
  std::vector<cv::Point2f> in = {br};
  std::vector<cv::Point2f> out;
  cv::transform(in, out, A);
  cv::Point2f br_mapped = out[0];
  cv::Point2f br_target =
      dst[2] + off; // + offset as A transform matrix with offset applied
  float err_dist = dist(br_mapped, br_target);

  cv::Size canvas(std::lround(std::max(w + 2 * m, br_mapped.x + m)),
                  std::lround(std::max(h + 2 * m, br_mapped.y + m)));
  cv::Mat affine;
  cv::warpAffine(img, affine, A, canvas);

  Quad target = {dst[0] + off, dst[1] + off, dst[2] + off, dst[3] + off};
  Quad mapped_quad = {dst_tri[0], dst_tri[1], br_mapped, dst_tri[2]};
  draw_quad(affine, target, {0, 255, 0});      // desired rectangle
  draw_quad(affine, mapped_quad, {0, 0, 255}); // actual object outline
  draw_cross(affine, br_target, {0, 255, 0});
  draw_cross(affine, br_mapped, {0, 0, 255});
  cv::line(affine, br_target, br_mapped, {0, 255, 255}, thickness(affine),
           cv::LINE_AA);
  save_figure(out_dir / "1_affine.jpg", affine);

  // Same error seen in the photo: an affine map keeps parallelograms, so it
  // "expects" the 4th corner at TL + (TR-TL) + (BL-TL).
  cv::Point2f br_affine_src = tr + bl - tl;
  float errSrc = dist(br_affine_src, br);

  // 1.3 Evidence for perspective
  cv::Mat annotated = img.clone();
  draw_quad(annotated, pts, {0, 255, 0});
  Quad parallelogram = {tl, tr, br_affine_src, bl};
  draw_quad(annotated, parallelogram, {0, 0, 255});
  draw_cross(annotated, br, {0, 255, 0});
  draw_cross(annotated, br_affine_src, {0, 0, 255});
  save_figure(out_dir / "1_input_corners.jpg", annotated);

  double ang_top_bottom = angle_between(tl, tr, bl, br);
  double ang_left_right = angle_between(tl, bl, tr, br);

  std::cout << "\n Exercise 1: perspective rectification \n"
            << "corners (TL TR BR BL): ";
  for (auto &p : pts)
    std::cout << '(' << p.x << ", " << p.y << ") ";
  std::cout << "\noutput size: " << cvRound(w) << " x " << cvRound(h)
            << "\nH =\n"
            << H << "\nA =\n"
            << A << "\n4th corner (BR) after affine warp: (" << br_mapped.x - m
            << ", " << br_mapped.y - m << "), target (" << w - 1 << ", "
            << h - 1 << ")"
            << "\ndistance in rectified image: " << err_dist << " px"
            << "\ndistance in input photo (parallelogram vs. real BR): "
            << errSrc << " px"
            << "\nangle top vs. bottom edge:  " << ang_top_bottom << " deg"
            << "\nangle left vs. right edge:  " << ang_left_right << " deg"
            << "\nwrote " << out_dir / "1_*.jpg" << '\n';
}
