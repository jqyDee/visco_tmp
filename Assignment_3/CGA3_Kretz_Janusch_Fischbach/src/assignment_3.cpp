#include <algorithm>
#include <cstdlib>

#include "math/matrix4d.h"
#include "math/vector3d.h"
#include "mygl/base.h"
#include "mygl/camera.h"
#include "mygl/geometry.h"
#include "mygl/mesh.h"
#include "mygl/shader.h"

#include "ground.h"

// Helper functions you may need for the wing flapping animation.
struct WingFlapAnimation {
  // t stores the time since the engine start / flapping start.
  float t = 0.0f;

  // The flapping speed increases over time, meaning the flapping cycle time
  // decreases over time.
  // T(t) = (T_start - T_end) * e^(-0.5*t) + T_end.
  float cycleTimeStart = 3.0f;
  float cycleTimeEnd = 0.1f;

  float cycleTime() const {
    return (cycleTimeStart - cycleTimeEnd) * std::exp(-0.5f * t) + cycleTimeEnd;
  }

  // The rotation of the wings during the flapping animation is modelled by a
  // Sinus function: a * sin(2*pi*t/T(t)).
  float angleZMax = to_radians(3.0f);
  float angleZ() const {
    return angleZMax *
           std::sin(2.0f * static_cast<float>(M_PI) * t / cycleTime());
  }

  // fraction of the full wing flapping speed
  float speedFraction() const { return 1.0f - std::exp(-0.5f * t); }
};

inline float smoothstep(float x) { return x * x * (3.0f - 2.0f * x); }

enum WingPhase {
  WING_REST,
  WING_OPENING,
  WING_FLIPPING_TO,
  WING_FLIPPING_BACK,
  WING_FLAPPING
};

struct WingAnimation {
  WingPhase phase = WING_REST;
  float phaseTime = 0.0f;
  float openDuration = 3.0f;
  float flipDuration = 1.5f;

  float restSplay = to_radians(90.0f - 16.0f);
  float splay = to_radians(90.0f - 16.0f);
  float flip = 0.0f;
  WingFlapAnimation flap;

  void start() {
    if (phase == WING_REST) {
      phase = WING_OPENING;
      phaseTime = 0.0f;
    }
  }

  void flipStep() {
    float s = smoothstep(std::min(phaseTime / flipDuration, 1.0f));
    if (phase == WING_FLIPPING_BACK) {
      s = -s;
    }
    flip = s * static_cast<float>(M_PI); // 0deg to 180deg
    if (phaseTime >= flipDuration) {
      phase = (phase == WING_FLIPPING_TO) ? phase = WING_FLIPPING_BACK
                                          : WING_FLAPPING;
      phaseTime = 0.0f;
    }
  }

  void update(float dt) {
    phaseTime += dt;

    switch (phase) {
    case WING_REST:
      break;
    case WING_OPENING: {
      const float s = smoothstep(std::min(phaseTime / openDuration, 1.0f));
      splay = (1.0f - s) * restSplay; // 74deg to 0deg
      if (phaseTime >= openDuration) {
        phase = WING_FLIPPING_TO;
        phaseTime = 0.0f;
      }
      break;
    }
    case WING_FLIPPING_TO: {
      flipStep();
      break;
    }
    case WING_FLIPPING_BACK: {
      flipStep();
      break;
    }
    case WING_FLAPPING:
      flap.t += dt;
      break;
    }
  }

  float flapAngle() const {
    return phase == WING_FLAPPING ? flap.angleZ() : 0.0f;
  }

  // Wings count as "full speed" once 95% of the final flap rate is reached.
  static constexpr float fullSpeedThreshold = 0.95f;

  // Take-off progress 0..1, eased, reaches exactly 1 at full speed.
  float takeoffProgress() const {
    if (phase != WING_FLAPPING)
      return 0.0f;
    return smoothstep(
        std::min(flap.speedFraction() / fullSpeedThreshold, 1.0f));
  }

  bool fullSpeed() const {
    return phase == WING_FLAPPING && flap.speedFraction() >= fullSpeedThreshold;
  }
};

enum CubeColor {
  BODY = 0,         // 102 = 0.4
  WINGS = 1,        // 25 = 0.1
  FOOT = 2,         // 150 = 0.6
  GROUND_PLATE = 3, // 50 = 0.2
};

struct Object {
  CubeColor color;
  Matrix4D scaling;
  Matrix4D translation;
  Matrix4D transformation = Matrix4D::identity();
  // pivot the origin of the object to a different position (e.g. wings)
  Matrix4D pivot = Matrix4D::identity();

  // applied right to left
  Matrix4D model() const {
    return translation * transformation * pivot * scaling;
  }
};

struct Wing {
  Matrix4D scaling = Matrix4D::scale(3.75f, 0.05f, 0.15f);
  Matrix4D pivot = Matrix4D::translation({3.75f, 0.0f, 0.0f});

  // Left-side root points (plate-centered frame). Right side = mirrored.
  Vector3D roots[4] = {
      {0.55f, 2.3f, -0.15f}, // top outer
      {0.4f, 2.3f, -1.0f},   // top inner
      {0.6f, 1.35f, 0.0f},   // bottom outer
      {0.5f, 1.35f, -0.9f},  // bottom inner
  };
  // Top and bottom wings flap in opposite directions.
  float flapDir[4] = {1.0f, 1.0f, -1.0f, -1.0f};

  Matrix4D local(float splay, float flip, float flapAngle) const {
    return Matrix4D::rotationY(splay) * Matrix4D::rotationZ(flapAngle) *
           Matrix4D::rotationX(flip) * pivot * scaling;
  }
};

struct Ornithopter {
  // elevation on "engine" start
  float groundY = 2.0f;
  float liftHeight = 2.0f;
  bool controlsUnlocked = false;

  Vector3D position = {0.0f, groundY, 0.0f};
  float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
  float speed = 4.0f;
  float turnSpeed = M_PI / 2.0f;
  float maxTilt = to_radians(10.0f);
  float tiltSpeed = 4.0f;

  Matrix4D model() const {
    return Matrix4D::translation(position) * Matrix4D::rotationY(yaw) *
           Matrix4D::rotationX(pitch) * Matrix4D::rotationZ(roll);
  }
};

// Struct holding all necessary state variables for scene.
struct {
  // Camera.
  Camera camera;
  Camera cameraChase;
  CameraMode camMode;
  float zoomSpeedMultiplier;

  // Game objects.
  Ground ground;

  // ornithopter
  Mesh cubeMeshes[4];
  Object objs[4];
  Wing wing;
  WingAnimation animation;
  Ornithopter ornithopter;

  // Shader.
  ShaderProgram shaderColor;
} sScene;

enum Key {
  KEY_W,
  KEY_S,
  KEY_A,
  KEY_D,
  KEY_Q,
  KEY_E,
  KEY_UP,
  KEY_DOWN,
  // this is very smart i believe, i feel great
  KEY_COUNT
};

// Struct holding all state variables for input.
struct {
  bool mouseLeftButtonPressed = false;
  Vector2D mousePressStart;
  bool keyPressed[KEY_COUNT] = {};
} sInput;

// GLFW callback function for keyboard events.
void callbackKey(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  // Close window on escape.
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }

  // Make screenshot and save in work directory.
  if (key == GLFW_KEY_P && action == GLFW_PRESS) {
    screenshotToPNG("screenshot.png");
  }

  // Start engine (start animation)
  if (key == GLFW_KEY_F && action == GLFW_PRESS) {
    sScene.animation.start();
  }

  // Input for camera control.
  if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
    sScene.camMode = CAM_NORMAL;
    sScene.camera.lookAt = {0.0f, 0.0f, 0.0f};
    cameraUpdateOrbit(sScene.camera, {0.0f, 0.0f}, 0.0f);
  }
  if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
    sScene.camMode = CAM_FOLLOW;
  }
  if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
    sScene.camMode = CAM_CHASE;
  }

  // Ornithopter controls
  const bool down = (action != GLFW_RELEASE);
  switch (key) {
  case GLFW_KEY_W:
    sInput.keyPressed[KEY_W] = down;
    break;
  case GLFW_KEY_S:
    sInput.keyPressed[KEY_S] = down;
    break;
  case GLFW_KEY_A:
    sInput.keyPressed[KEY_A] = down;
    break;
  case GLFW_KEY_D:
    sInput.keyPressed[KEY_D] = down;
    break;
  case GLFW_KEY_Q:
    sInput.keyPressed[KEY_Q] = down;
    break;
  case GLFW_KEY_E:
    sInput.keyPressed[KEY_E] = down;
    break;
  case GLFW_KEY_SPACE:
    sInput.keyPressed[KEY_UP] = down;
    break;
  case GLFW_KEY_LEFT_SHIFT:
    sInput.keyPressed[KEY_DOWN] = down;
    break;
  }
}

// GLFW callback function for mouse position events.
void callbackMousePos(GLFWwindow *window, double x, double y) {
  if (sInput.mouseLeftButtonPressed) {
    Vector2D diff = sInput.mousePressStart - Vector2D(x, y);
    cameraUpdateOrbit(sScene.camera, diff, 0.0f);
    sInput.mousePressStart = Vector2D(x, y);
  }
}

// GLFW callback function for mouse button events.
void callbackMouseButton(GLFWwindow *window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    sInput.mouseLeftButtonPressed =
        (action == GLFW_PRESS || action == GLFW_REPEAT);

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    sInput.mousePressStart = Vector2D(x, y);
  }
}

// GLFW callback function for mouse scroll events.
void callbackMouseScroll(GLFWwindow *window, double xoffset, double yoffset) {
  cameraUpdateOrbit(sScene.camera, {0, 0},
                    -sScene.zoomSpeedMultiplier * yoffset);
}

// GLFW callback function for window resize event.
void callbackWindowResize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  sScene.camera.width = width;
  sScene.camera.height = height;
  sScene.cameraChase.width = width;
  sScene.cameraChase.height = height;
}

// Function to setup and initialize the whole scene.
void sceneInit(float width, float height) {
  sScene.ground =
      groundCreate({150.0f / 255.0f, 95.0f / 255.0f, 55.0f / 255.0f},
                   {226.0f / 255.0f, 180.0f / 255.0f, 120.0f / 255.0f});

  // ====== ornithopter ======
  // ------ Meshes ------
  sScene.cubeMeshes[BODY] =
      meshCreate(cube::vertexPos, cube::indices, {0.4f, 0.4f, 0.4f, 1.0f},
                 GL_STATIC_DRAW, GL_STATIC_DRAW);
  sScene.cubeMeshes[WINGS] =
      meshCreate(cube::vertexPos, cube::indices, {0.1f, 0.1f, 0.1f, 1.0f},
                 GL_STATIC_DRAW, GL_STATIC_DRAW);
  sScene.cubeMeshes[FOOT] =
      meshCreate(cube::vertexPos, cube::indices, {0.6f, 0.6f, 0.6f, 1.0f},
                 GL_STATIC_DRAW, GL_STATIC_DRAW);
  sScene.cubeMeshes[GROUND_PLATE] =
      meshCreate(cube::vertexPos, cube::indices, {0.2f, 0.2f, 0.2f, 1.0f},
                 GL_STATIC_DRAW, GL_STATIC_DRAW);

  // ----- objs (body) ------
  sScene.objs[0] = {BODY, Matrix4D::scale(0.7f, 0.7f, 2.7f),
                    Matrix4D::translation({0.0f, 1.5f, 0.4f}),
                    Matrix4D::rotationX(to_radians(8.0f))};
  sScene.objs[1] = {BODY, Matrix4D::scale(0.4f, 0.3f, 3.0f),
                    Matrix4D::translation({0.0f, 2.0f, -5.0f})};
  // ----- objs plane and foot -----
  sScene.objs[2] = {FOOT, Matrix4D::scale(0.1f, 0.5f, 1.0f),
                    Matrix4D::translation({0.0f, 0.5f, 0.0f})};
  // this is the ornithopter origin
  sScene.objs[3] = {GROUND_PLATE, Matrix4D::scale(0.7f, 0.1f, 1.5f),
                    Matrix4D::translation({0.0f, 0.0f, 0.0f})};

  // ===== camera =====
  sScene.cameraChase = cameraCreate(width, height, to_radians(45.0f), 0.01f,
                                    500.0f, {12.0f, 4.0f, -12.0f});
  sScene.camera = cameraCreate(width, height, to_radians(45.0f), 0.01f, 500.0f,
                               {12.0f, 4.0f, -12.0f});
  sScene.camMode = CAM_NORMAL;
  sScene.zoomSpeedMultiplier = 0.05f;

  // ===== shader =====
  sScene.shaderColor = shaderLoad("shader/default.vert", "shader/default.frag");
}

// Function to move and update objects in scene.
void sceneUpdate(float dt) {
  Ornithopter &o = sScene.ornithopter;

  WingAnimation &anim = sScene.animation;
  anim.update(dt);

  // Take-off: rise with the flap speed, unlock controls at full speed.
  if (!o.controlsUnlocked) {
    o.position.y = o.groundY + o.liftHeight * anim.takeoffProgress();
    if (anim.fullSpeed()) {
      o.controlsUnlocked = true;
    }
  }

  const bool *k = sInput.keyPressed;
  const float lock = o.controlsUnlocked ? 1.0f : 0.0f;
  const float fwd = lock * (float(k[KEY_W]) - float(k[KEY_S]));
  const float side = lock * (float(k[KEY_A]) - float(k[KEY_D]));
  const float vert = lock * (float(k[KEY_UP]) - float(k[KEY_DOWN]));
  const float turn = lock * (float(k[KEY_Q]) - float(k[KEY_E]));

  // turn right/left
  o.yaw += turn * o.turnSpeed * dt;

  // Local axes rotated by yaw. Nose = +z, left = +x.
  const Vector3D forward =
      Matrix3D::rotationY(o.yaw) * Vector3D(0.0f, 0.0f, 1.0f);
  const Vector3D left = Matrix3D::rotationY(o.yaw) * Vector3D(1.0f, 0.0f, 0.0f);
  o.position = o.position + forward * (fwd * o.speed * dt) +
               left * (side * o.speed * dt) +
               Vector3D(0.0f, vert * o.speed * dt, 0.0f);

  // flight animations
  const float targetPitch = fwd * o.maxTilt;
  const float targetRoll = -side * o.maxTilt;
  const float animationEasing = std::min(1.0f, o.tiltSpeed * dt);
  o.pitch += (targetPitch - o.pitch) * animationEasing;
  o.roll += (targetRoll - o.roll) * animationEasing;

  if (sScene.camMode == CAM_CHASE) {
    cameraChase(sScene.cameraChase, o.position, o.yaw);
  }
  if (sScene.camMode == CAM_FOLLOW) {
    cameraFollow(sScene.camera, o.position);
  }

  sScene.animation.update(dt);
}

void drawCube(const Mesh &m, const Matrix4D pos) {
  shaderUniform(sScene.shaderColor, "uModel", pos);
  glBindVertexArray(m.vao);
  glDrawElements(GL_TRIANGLES, m.size_ibo, GL_UNSIGNED_INT, nullptr);
}

// Function to draw all objects in the scene.
void sceneDraw() {
  // Clear framebuffer color.
  glClearColor(135.0f / 255, 206.0f / 255, 235.0f / 255, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Render scene — use shader and set uniforms (names match the ones in the
  // shader).
  {
    const Camera &cam =
        (sScene.camMode == CAM_CHASE) ? sScene.cameraChase : sScene.camera;
    glUseProgram(sScene.shaderColor.id);
    shaderUniform(sScene.shaderColor, "uProj", cameraProjection(cam));
    shaderUniform(sScene.shaderColor, "uView", cameraView(cam));

    // Draw ground.
    shaderUniform(sScene.shaderColor, "uModel", Matrix4D::identity());
    glBindVertexArray(sScene.ground.mesh.vao);
    glDrawElements(GL_TRIANGLES, sScene.ground.mesh.size_ibo, GL_UNSIGNED_INT,
                   nullptr);

    // draw body, foot, ground plate
    const Matrix4D base = sScene.ornithopter.model();
    for (const auto &p : sScene.objs) {
      drawCube(sScene.cubeMeshes[p.color], base * p.model());
    }

    // draw wings
    const WingAnimation &anim = sScene.animation;
    const Matrix4D mirrorX = Matrix4D::scale(-1.0f, 1.0f, 1.0f);
    const Mesh wingMesh = sScene.cubeMeshes[WINGS];
    for (size_t i = 0; i < 4; i++) {
      const Vector3D root = sScene.wing.roots[i];
      const Matrix4D m =
          Matrix4D::translation(root) *
          sScene.wing.local(anim.splay, anim.flip,
                            sScene.wing.flapDir[i] * anim.flapAngle());
      // left
      drawCube(wingMesh, base * m);
      // right
      drawCube(wingMesh, base * mirrorX * m);
    }
  }
  glCheckError();

  // Cleanup OpenGL state.
  glBindVertexArray(0);
  glUseProgram(0);
}

int main(int argc, char **argv) {
  // Create window/context.
  int width = 1280;
  int height = 720;
  GLFWwindow *window = windowCreate(
      "Assignment 3 - Transformations, User Input and Camera", width, height);
  if (!window) {
    return EXIT_FAILURE;
  }

  // Set window callbacks.
  glfwSetKeyCallback(window, callbackKey);
  glfwSetCursorPosCallback(window, callbackMousePos);
  glfwSetMouseButtonCallback(window, callbackMouseButton);
  glfwSetScrollCallback(window, callbackMouseScroll);
  glfwSetFramebufferSizeCallback(window, callbackWindowResize);

  // Init OpenGL.
  glEnable(GL_DEPTH_TEST);

  // Setup scene.
  sceneInit(width, height);

  // Main loop.
  double timeStamp = glfwGetTime();
  double timeStampNew = 0.0;

  while (!glfwWindowShouldClose(window)) {
    // Poll and process input and window events.
    glfwPollEvents();

    // Update camera and model matrices.
    timeStampNew = glfwGetTime();
    sceneUpdate(timeStampNew - timeStamp);
    timeStamp = timeStampNew;

    // Draw all objects in the scene.
    sceneDraw();

    // Swap front and back buffer.
    glfwSwapBuffers(window);
  }

  // Cleanup — delete OpenGL shader and buffers.
  shaderDelete(sScene.shaderColor);
  groundDelete(sScene.ground);

  // cleanup meshes
  for (const auto &m : sScene.cubeMeshes)
    meshDelete(m);

  // Cleanup GLFW/GL context.
  windowDelete(window);

  return EXIT_SUCCESS;
}
