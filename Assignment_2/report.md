# Visual Computing - Assignment 2

Deadline: 23.59 09.11.2026

Group: `TMP`

Group-members: 
- Tim Janush (csbb9638)
- Paul Kretz (csXXXXXX)
- Matti Fischbach (csbb8241)

## Setup

Change dir into src, create a `build`-dir and `cd` into it:

```bash
cd src && mkdir build && cd build
```

Run cmake in `build` with the underlying `CMakeLists.txt` (and if needed
`compile_commands.json`). 

```bash
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1
```

Optional: If compile commands generated symlink them into the src dir for your lsp to pick it up

```bash
cd .. && ln -s build/compile_commands.json
```

To now compile the program run
```bash
cd build && make
```
in the build dir.

And to run the program use:
```bash
./ass02 rectify <image>                     # click corners TL, TR, BR, BL, Enter
./ass02 rectify <image> x_tl y_tl x_tr y_tr x_br y_br x_bl y_bl
./ass02 pyramid <image> [max_side]          # optional: downscale longest side first
./ass02 gabor <image> [max_side]            # Ex 3.1: filter bank + Sobel comparison
./ass02 gabor-sweep <image> [max_side]      # Ex 3.2: one parameter changed at a time
```
Results are written to `./build/out/`.

## 1. Perspective rectification

### What we did and why

We photographed `rectification.jpg` at a clear angle (Fig. 1.1). The four corners are
picked by mouse clicks in the order top-left, top-right, bottom-right,
bottom-left. The image is shown downscaled (so OpenCV does not crash and the
window does not use the full screen or more), and the clicks are
scaled back to full-resolution pixel coordinates. The program prints the
picked coordinates so a run can be repeated exactly.

The output size is the longer of each pair of opposite edges, so the
rectified object is not downsampled:

```cpp
float w = std::max(dist(tl, tr), dist(bl, br));
float h = std::max(dist(tl, bl), dist(tr, br));
Quad dst = {cv::Point2f(0, 0), cv::Point2f(w - 1, 0),
            cv::Point2f(w - 1, h - 1), cv::Point2f(0, h - 1)};
```

**Perspective.** A perspective transform has 8 degrees of freedom (fixed 1 in
the bottom right of the matrix). With 8 DOF's we the transformation matrix can
rotate, scale x, y and translate x, y. The 4 point pairs give exactly 8 linear
equations, so `getPerspectiveTransform` solves for H directly:

```cpp
cv::Mat H = cv::getPerspectiveTransform(src.data(), dst.data());
cv::warpPerspective(img, persp, H, cv::Size(cvRound(w), cvRound(h)));
```

**Affine.** An affine map has 6 degrees of freedom (bottom row fixed to {0, 0,
1}), so 3 pairs (TL, TR, BL) fix it completely. This has us missing x and y
translation. The 4th corner is then not used. We push it through the same matrix
to see where it lands. The output canvas has a margin, so the point stays
visible even if it lands outside the target rectangle:

```cpp
cv::Point2f src_tri[3] = {tl, tr, bl};
cv::Point2f dst_tri[3] = {dst[0] + off, dst[1] + off, dst[3] + off};
cv::Mat A = cv::getAffineTransform(src_tri, dst_tri);
cv::transform(in, out, A);                            // where BR really lands
float err_dist = cv::norm(out[0] - (dst[2] + off));   // vs. where it should land
```

### Results

![Input](out/1_input_corners.jpg)
*Fig. 1.1: Input photo. Green: the 4 picked corners. Red: the parallelogram that
an affine map assumes, spanned by TL, TR and BL. The red cross is where it
expects the 4th corner; the green cross is the real one.*

![Perspective](out/1_perspective.jpg)
*Fig. 1.2: Rectified with `getPerspectiveTransform` + `warpPerspective`
(computed at 4195x4874 px, shown downscaled).*

![Affine](out/1_affine.jpg)
*Fig. 1.3: Warped with `getAffineTransform` from TL, TR and BL. Green: the
target rectangle. Red: the real object outline after the warp. Red cross: where
the 4th corner (BR) lands. Green cross: where it should be.*

| Quantity | Value |
|---|---|
| Distance of the 4th corner to its target, rectified image | 2588.48 px |
| Same distance in the input photo (parallelogram corner vs. real BR) | 1492.38 px |
| Angle between top and bottom edge in the photo | 13.3306 deg |
| Angle between left and right edge in the photo | 17.2288 deg |

*Table 1.1: Error of the affine approach (program output).*

### Observations

The perspective transform maps all 4 corners exactly onto the rectangle.
Straight lines stay straight, and the text and edges of the object appear
frontal. The affine warp matches TL, TR and BL exactly, because they define it.
The 4th corner misses its target by 2588.48 px, so the object becomes a tilted
quadrilateral, not a rectangle. The scale inside the object is still uneven: the
part that was further from the camera stays smaller.

**Why affine cannot do this.** An affine map `x' = A x + t` has 6 degrees of
freedom and always maps parallel lines to parallel lines. It also keeps ratios
of lengths along a line, and its scale is the same everywhere in the image. It
therefore maps parallelograms to parallelograms. Once TL, TR and BL are fixed,
BR must land at `TR + BL - TL` exactly for affine transformation to do the same
as the perspective transform. The rectangle's 4th corner is then forced,
whatever the photo shows. A camera viewing a plane at an angle performs a
projective map:

```
x' = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
```

The denominator depends on the position. It models the division by depth: points
further away get smaller. That gives 8 degrees of freedom, so we need 4 point
pairs. An affine map is the special case `h31 = h32 = 0`. To undo a perspective
image we need the inverse projective map, and no affine map can produce it.

**How the photo shows it is perspective.** In the photo, opposite edges of the
object are not parallel (Table 1.1: 13.3306 deg and 17.2288 deg). Extended, they
would meet in a vanishing point. Under an affine map they would stay parallel,
so the four corners would form a parallelogram. The real 4th corner lies 1492.38
px away from that parallelogram corner (Fig. 1.1). Also, the top and bottom
edges of the poster have the same real length, but in the photo the far top edge
measures about 0.8 % smaller than the bottom. An affine map scales parallel
segments by the same factor, so they would keep equal length.

## 2. Gaussian and Laplacian pyramid

### Note

We noticed too late that there is OpenCV functions to do the Gaussian and
Laplacian directly. This is why the implementation does most of this from
scratch.

### What we did and why

Input: the photo from Exercise 1 (`rectification.jpg`, 8064 x 6048). We first
downscaled it with `INTER_AREA` to 1024 px on the longest side
(`./ass02 pyramid rectification.jpg 1024`), giving 768 x 1024 px. At full
resolution the finest levels contain mostly sensor noise and JPEG artifacts,
and the figures would be too large for the report. All computation is done in
`CV_32FC3`. This keeps the negative values of the Laplacian levels, and
rounding cannot spoil the reconstruction.

**Gaussian pyramid (A).** We build `reduce` ourselves instead of calling
`cv::pyrDown`. The step blurs with the sampled Gauss filter [1 4 6 4 1]/16
separably in x and y (`BORDER_REFLECT_101`), then keeps every second row and
column:

```cpp
const cv::Mat kGauss = cv::Mat_<float>({1, 4, 6, 4, 1}) / 16.0f;
cv::sepFilter2D(img, blurred, -1, kGauss, kGauss, {-1, -1}, 0,
                cv::BORDER_REFLECT_101);
for (int y = 0; y < out.rows; ++y)
  for (int x = 0; x < out.cols; ++x)
    out.at<cv::Vec3f>(y, x) = blurred.at<cv::Vec3f>(2 * y, 2 * x);
```

Blurring before subsampling is needed. It removes frequencies above the new
Nyquist limit, which would otherwise alias into the smaller image. We apply it
4 times, which gives levels G0 (original) to G4.

**Laplacian pyramid (B).** `expand` is the inverse step. It writes level i+1
onto the even pixels of a zero image of the size of level i. Then it blurs
with the same kernel times 2 per axis: only every 4th pixel is non-zero, so
the brightness has to be multiplied by 4. Passing the target size handles odd
image sizes.

```cpp
L[i] = G[i] - expand(G[i + 1], G[i].size());   // i = 0..3
L[4] = G[4];                                   // top: smallest Gaussian level
// reconstruction, coarse to fine:
img = L[4];
for (int i = 3; i >= 0; --i) img = L[i] + expand(img, L[i].size());
```

Laplacian values are signed and small. For display we show `128 + 2 * L`, so
grey means 0, brighter means positive and darker means negative.

### Results

![Gaussian pyramid](out/2_gauss_pyramid.png)
*Fig. 2.1: Gaussian pyramid. Left: G0 (768 x 1024). Right, top to bottom:
G1 (384 x 512), G2 (192 x 256), G3 (96 x 128), G4 (48 x 64).*

![Gaussian zoom](out/2_gauss_zoom.png)
*Fig. 2.2: The same center crop from G0 ... G4, scaled up to the same size with
nearest neighbour. One block in the right image is one pixel of G4.*

![Laplacian pyramid](out/2_lap_pyramid.png)
*Fig. 2.3: Laplacian pyramid L0 ... L3 (shown as 128 + 2 * L), and the
smallest Gaussian level G4 as the top level (bottom right, normal colours).*

![Laplacian zoom](out/2_lap_zoom.png)
*Fig. 2.4: The same center crop from L0 ... L3 and G4, as in Fig. 2.2.*

| Level | Size | min L | max L |
|---|---|---|---|
| 0 | 768 x 1024 | -120.8 | 63.9 |
| 1 | 384 x 512 | -78.6 | 41.5 |
| 2 | 192 x 256 | -45.2 | 50.8 |
| 3 | 96 x 128 | -36.2 | 52.2 |
| 4 (= G4) | 48 x 64 | 2.2 | 202.4 |

*Table 2.1: Level sizes and value range of each Laplacian level (all channels).*

| Comparison | max \|reconstructed - original\| |
|---|---|
| float (`CV_32FC3`) | 3.8e-6 |
| after conversion to 8 bit | 0 |

*Table 2.2: Reconstruction error from L0 ... L3 and G4.*

### Observations

**Low frequencies** are the parts of the image that change slowly over space.
Examples are the overall brightness and colour of regions (white remote,
brown table), large smooth gradients from the lighting, and the coarse shape
of objects. They survive down to G4: at 48 x 64 px you can still see a white
bar on a brown background.

**Details that disappear first** are the finest structures, only 1 to 2 px
wide in G0. Here these are the wood grain and scratches on the table, the thin
dark outlines around the buttons, the small text ("prime video",
"Rakuten TV", "Disney+"), the braille dots, and image noise. In G1 the text is
already hard to read. In G2 the button labels are colour blobs, and the ring
around the center button blurs together (Fig. 2.2). In G3 and G4 only the
buttons as spots and the outline of the remote are left. Each reduce step
halves the highest frequency the image can show, so the smallest structures
go first.

**Each Laplacian level contains** the band of frequencies that is lost between
two Gaussian levels: `L_i = G_i - expand(G_{i+1})`. It is a band-pass image
with mean about 0. Most of it is grey, and it is non-zero only at edges and
texture of the matching size.
- L0: the finest details: wood grain, thin outlines, text strokes, noise.
- L1: slightly coarser structure: button symbols and outlines, thicker grain.
- L2 and L3: the outlines of the buttons, the large ring and the remote's
  edge, as broad light/dark bands.
- The top level G4: the remaining low-pass image (average colours).

The value range shrinks from L0 to L3 (Table 2.1). Most of the contrast of
this image sits in the sharp edges.

**Reconstruction** is exact up to float rounding (3.8e-6). After conversion to
8 bit the difference is 0. This is expected, because by construction
`G_i = L_i + expand(G_{i+1})`. The same `expand` is used to build and to
reconstruct, so the pyramid does not lose any information.

**Uses of the pyramid:**
- **Compression:** Laplacian levels are mostly about 0 and can be quantized
  strongly. This was the original use (Burt & Adelson 1983).
- **Image blending / mosaics:** blend each level with a mask of matching
  blur, then reconstruct. This gives seamless transitions, e.g. the
  "orapple".
- **Multi-scale analysis:** finding features at different sizes, e.g. the
  scale space in SIFT, or coarse-to-fine optical flow and image
  registration.
- **Editing:** sharpening or denoising by amplifying or damping single
  frequency bands, and progressive transmission (send G4 first, then the
  L levels).

## 3. Gabor filter bank

### What we did and why

Input: the photo from Exercise 1 (`rectification.jpg`, 8064 x 6048). We first
downscaled it with `INTER_AREA` to 1000 px on the longest side
(`./ass02 gabor rectification.jpg 1000`), giving 750 x 1000 px. At full
resolution a wavelength of 10 px only matches tiny details like wood grain and
noise, and every output image would be 10 to 16 MB. The image is converted to
greyscale and to float in [0, 1], because the filter responses are signed.

**Filter bank (1).** For each orientation 0, 45, 90 and 135 deg we build one
kernel with `getGaborKernel` (theta in radians). We use `ksize = 31`,
`sigma = 4` (width of the Gaussian envelope), `lambda = 10` (wavelength of the
stripes in px), `gamma = 0.5` (aspect ratio of the envelope, < 1 means
stretched along the stripes) and `psi = 0` (phase, symmetric kernel with a
bright center stripe). `ksize` has to cover about +-3 sigma, otherwise the
kernel is cut off.

```cpp
cv::Mat k = cv::getGaborKernel(cv::Size(p.ksize, p.ksize), p.sigma,
                               deg * CV_PI / 180.0, p.lambda, p.gamma,
                               p.psi, CV_32F);
k -= cv::mean(k)[0]; // zero mean: flat areas give 0
```

Removing the mean is needed. With `psi = 0` the values of the OpenCV kernel
sum to about 8, not 0. Such a kernel also acts like a blur: on a flat area it
returns 8 times the brightness. Our first results looked like a low-contrast
copy of the photo, because the brightness dominated the stripe response. After
removing the mean (the DC part), flat areas give 0 and only structure remains.

Each kernel is applied with `filter2D` and a float output (`CV_32F`). The four
responses are combined by the per-pixel maximum. We compared two variants: the
maximum of the signed responses ("raw") and the maximum of the absolute values
("abs"):

```cpp
cv::filter2D(gray, r, CV_32F, k);              // for each orientation
cv::Mat combined = cv::abs(responses[0]);
for (size_t i = 1; i < responses.size(); i++) {
  cv::Mat a = cv::abs(responses[i]);
  cv::max(combined, a, combined);
}
```

For display every float image is stretched to 0 to 255 with `normalize`
(`NORM_MINMAX`). For kernels and signed responses, grey therefore means 0,
brighter means positive and darker means negative. The kernels are scaled up
with nearest neighbour to the height of the result and placed to its left.

**Parameter sweep (2).** `./ass02 gabor-sweep` runs the whole bank (abs
variant) again with the default values from above and changes exactly one
parameter at a time. Only in the sigma sweep we use `ksize = 61` for all three
values, otherwise the kernel for sigma = 8 would be cut off.

**Sobel (2).** For the comparison we compute the gradient magnitude on the same
greyscale image:

```cpp
cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
cv::magnitude(gx, gy, mag);
```

### Results

![Gabor 0](out/3_gabor_pair_0.jpg)
*Fig. 3.1: Left: kernel 0 deg (vertical stripes). Right: its signed response
(grey = 0).*

![Gabor 45](out/3_gabor_pair_45.jpg)
*Fig. 3.2: Left: kernel 45 deg. Right: its signed response.*

![Gabor 90](out/3_gabor_pair_90.jpg)
*Fig. 3.3: Left: kernel 90 deg (horizontal stripes). Right: its signed
response.*

![Gabor 135](out/3_gabor_pair_135.jpg)
*Fig. 3.4: Left: kernel 135 deg. Right: its signed response.*

![Combined raw](out/3_gabor_combined_raw.jpg)
*Fig. 3.5: The four responses combined by the maximum of the signed values
("raw").*

![Combined abs](out/3_gabor_combined.jpg)
*Fig. 3.6: The four responses combined by the maximum of the absolute values
("abs"). This variant is used for the sweep and the Sobel comparison.*

![Sweep sigma](out/3_sweep_sigma.jpg)
*Fig. 3.7: Combined result for sigma = 2, 4 and 8 (left to right, ksize = 61).*

![Sweep lambda](out/3_sweep_lambda.jpg)
*Fig. 3.8: Combined result for lambda = 5, 10 and 20.*

![Sweep gamma](out/3_sweep_gamma.jpg)
*Fig. 3.9: Combined result for gamma = 0.25, 0.5 and 1.*

![Sweep psi](out/3_sweep_psi.jpg)
*Fig. 3.10: Combined result for psi = 0 and 90 deg.*

![Sweep ksize](out/3_sweep_ksize.jpg)
*Fig. 3.11: Combined result for ksize = 15, 31 and 61.*

![Gabor vs Sobel](out/3_gabor_vs_sobel.jpg)
*Fig. 3.12: Left: Gabor bank (abs, default parameters). Right: Sobel gradient
magnitude.*

### Observations

**Each orientation** responds to lines and edges that run parallel to its
stripes (Fig. 3.1 ... 3.4). The 0 deg kernel marks the steep long edges of the
poster and the vertical strokes of the letters, while the horizontal wood grain
almost disappears. The 90 deg kernel is the opposite: the wood grain and the
top and bottom edges of the poster are strong. The 45 deg kernel matches the
long poster edges best, because the poster lies tilted in the photo. The
135 deg kernel shows the shorter poster edges and strokes in the other diagonal
direction. Thin bright lines give a bright response, thin dark lines a dark
one.

**Raw vs. abs.** The raw maximum only keeps positive responses, so dark lines
on a bright background lose (Fig. 3.5). At corners the four orientations
disagree, and star-shaped streaks from the side stripes of the kernels appear.
With the absolute value, bright and dark lines count the same and flat areas
are black (Fig. 3.6). Every edge now gives a double line. With `psi = 0` the
kernel is a line detector (bright center, dark sides), so a step edge gives a
positive response on one side and a negative one on the other.

**What each parameter does:**
- **sigma** sets the size of the envelope and so the number of visible
  stripes (Fig. 3.7). With sigma = 2 the envelope is smaller than one
  wavelength and only one stripe is left. The kernel works like a blob or edge
  detector with wide halos. With sigma = 8 there are many stripes and the
  kernel is very selective. Only long straight structures with period lambda
  remain, with striped ringing, and the letters fall apart.
- **lambda** sets the distance between the stripes (Fig. 3.8). lambda = 5 only
  finds very fine details and is weak and noisy. lambda = 20 gives thick lines,
  the double contours move apart, and the result is smoother.
- **gamma** sets the shape of the envelope: 0.25 is long and narrow, 1 is
  round (Fig. 3.9). A small gamma prefers long straight edges and is more
  selective in orientation, but gives streaks at corners. gamma = 1 also picks
  up short structures and more background texture.
- **psi** shifts the phase (Fig. 3.10). psi = 0 gives a symmetric kernel (line
  detector), psi = 90 deg an antisymmetric dark/bright kernel (edge detector).
  With psi = 90 deg each edge gives one contour instead of the double line.
- **ksize** only cuts the kernel (Fig. 3.11). With 15 px the outer stripes are
  cut off (15 px < 6 sigma). 31 and 61 look almost the same, because the
  envelope is already about 0 at the border.

**Best setting.** For this photo the defaults (sigma = 4, lambda = 10,
gamma = 0.5, ksize = 31) find the structures best. The poster border, the
letters, the pole and the outlines of bottle and glass are clean closed
contours, and the wood grain stays weak. For an edge map with single lines,
psi = 90 deg is the better choice.

**Gabor vs. Sobel** (Fig. 3.12):
- **Gabor finds, Sobel does not:** the bank is selective in orientation and in
  scale. It answers strongly to structures with a width of about lambda / 2,
  such as the poster border and the pole, and suppresses finer noise and
  texture. Each orientation can be read out on its own (Fig. 3.1 ... 3.4),
  while Sobel only gives one magnitude without frequency information. A thin
  line gives one response with Gabor, but two edges with Sobel.
- **Sobel finds, Gabor does not:** Sobel uses a 3 x 3 kernel, so its contours
  are thin and sharp. It finds every edge regardless of its width and
  direction. This includes the small dots on the rings and the fine signature,
  which are smaller than the Gabor wavelength. In turn, Sobel also responds to
  more fine wood texture and noise.
