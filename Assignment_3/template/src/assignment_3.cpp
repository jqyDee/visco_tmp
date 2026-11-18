#include <cstdlib>
#include <iostream>

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
};

// Struct holding all necessary state variables for scene.
struct {
  // Camera.
  Camera camera;
  bool cameraFollowOrnithopter;
  float zoomSpeedMultiplier;

  // Game objects.
  Ground ground;

  // Template objects, transformation matrices and animation properties.
  Mesh cubeMesh;
  Mesh cylinderMesh;
  Matrix4D cubeScalingMatrix;
  Matrix4D cubeTranslationMatrix;
  Matrix4D cubeTransformationMatrix;
  Matrix4D cylinderScalingMatrix;
  Matrix4D cylinderTranslationMatrix;
  Matrix4D cylinderTransformationMatrix;
  float cubeSpinRadPerSecond;

  // Shader.
  ShaderProgram shaderColor;
} sScene;

// Struct holding all state variables for input.
struct {
  bool mouseLeftButtonPressed = false;
  Vector2D mousePressStart;
  bool keyPressed[4] = {false, false, false, false};
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

  // Input for camera control.
  if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
    sScene.cameraFollowOrnithopter = false;
    sScene.camera.lookAt = {0.0f, 0.0f, 0.0f};
    cameraUpdateOrbit(sScene.camera, {0.0f, 0.0f}, 0.0f);
  }
  if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
    sScene.cameraFollowOrnithopter = true;
  }

  // Input for ornithopter control.
  if (key == GLFW_KEY_W) {
    sInput.keyPressed[0] = (action == GLFW_PRESS || action == GLFW_REPEAT);
  }
  if (key == GLFW_KEY_S) {
    sInput.keyPressed[1] = (action == GLFW_PRESS || action == GLFW_REPEAT);
  }

  if (key == GLFW_KEY_A) {
    sInput.keyPressed[2] = (action == GLFW_PRESS || action == GLFW_REPEAT);
  }
  if (key == GLFW_KEY_D) {
    sInput.keyPressed[3] = (action == GLFW_PRESS || action == GLFW_REPEAT);
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
}

// Function to setup and initialize the whole scene.
void sceneInit(float width, float height) {
  sScene.ground =
      groundCreate({150.0f / 255.0f, 95.0f / 255.0f, 55.0f / 255.0f},
                   {226.0f / 255.0f, 180.0f / 255.0f, 120.0f / 255.0f});

  // Template objects.
  sScene.cubeMesh =
      meshCreate(cube::vertices, cube::indices, GL_STATIC_DRAW, GL_STATIC_DRAW);
  sScene.cylinderMesh =
      meshCreate(cylinder::vertexPos, cylinder::indices,
                 {0.3f, 0.3f, 0.3f, 1.0f}, GL_STATIC_DRAW, GL_STATIC_DRAW);

  // Set up the transformation matrices for the template objects.
  sScene.cubeScalingMatrix = Matrix4D::scale(2.0f, 2.0f, 2.0f);
  sScene.cubeTranslationMatrix = Matrix4D::translation({2.0f, 3.0f, 0.0f});
  sScene.cubeTransformationMatrix = Matrix4D::identity();

  sScene.cylinderScalingMatrix = Matrix4D::identity();
  sScene.cylinderTranslationMatrix = Matrix4D::translation({-2.0f, 3.0f, 0.0f});
  sScene.cylinderTransformationMatrix = Matrix4D::identity();

  // Animation properties for the template objects.
  sScene.cubeSpinRadPerSecond = M_PI / 2.0f;

  // Initialize camera.
  sScene.camera = cameraCreate(width, height, to_radians(45.0f), 0.01f, 500.0f,
                               {12.0f, 4.0f, -12.0f});
  sScene.cameraFollowOrnithopter = false;
  sScene.zoomSpeedMultiplier = 0.05f;

  // Load shader from file.
  sScene.shaderColor = shaderLoad("shader/default.vert", "shader/default.frag");
}

// Function to move and update objects in scene.
void sceneUpdate(float dt) {
  // W and S rotate around the X axis.
  int rotationDirX = 0;
  if (sInput.keyPressed[0]) {
    rotationDirX = -1;
  } else if (sInput.keyPressed[1]) {
    rotationDirX = 1;
  }

  // A and D rotate around the Y axis.
  int rotationDirY = 0;
  if (sInput.keyPressed[2]) {
    rotationDirY = -1;
  } else if (sInput.keyPressed[3]) {
    rotationDirY = 1;
  }

  // Update the cube transformation matrices.
  // They will be used in the sceneDraw() function and passed to the shader
  // program.
  if (rotationDirX != 0 || rotationDirY != 0) {
    sScene.cubeTransformationMatrix =
        Matrix4D::rotationY(rotationDirY * sScene.cubeSpinRadPerSecond * dt) *
        Matrix4D::rotationX(rotationDirX * sScene.cubeSpinRadPerSecond * dt) *
        sScene.cubeTransformationMatrix;
  }
}

// Function to draw all objects in the scene.
void sceneDraw() {
  // Clear framebuffer color.
  glClearColor(135.0f / 255, 206.0f / 255, 235.0f / 255, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Render scene — use shader and set uniforms (names match the ones in the
  // shader).
  {
    glUseProgram(sScene.shaderColor.id);
    shaderUniform(sScene.shaderColor, "uProj", cameraProjection(sScene.camera));
    shaderUniform(sScene.shaderColor, "uView", cameraView(sScene.camera));

    // Draw ground.
    shaderUniform(sScene.shaderColor, "uModel", Matrix4D::identity());
    glBindVertexArray(sScene.ground.mesh.vao);
    glDrawElements(GL_TRIANGLES, sScene.ground.mesh.size_ibo, GL_UNSIGNED_INT,
                   nullptr);

    // Draw the cube. This requires to calculate the final model matrix from all
    // transformations.
    shaderUniform(sScene.shaderColor, "uModel",
                  sScene.cubeTranslationMatrix *
                      sScene.cubeTransformationMatrix *
                      sScene.cubeScalingMatrix);
    glBindVertexArray(sScene.cubeMesh.vao);
    glDrawElements(GL_TRIANGLES, sScene.cubeMesh.size_ibo, GL_UNSIGNED_INT,
                   nullptr);

    // Draw the cylinder.
    shaderUniform(sScene.shaderColor, "uModel",
                  sScene.cylinderTranslationMatrix *
                      sScene.cylinderTransformationMatrix *
                      sScene.cylinderScalingMatrix);
    glBindVertexArray(sScene.cylinderMesh.vao);
    glDrawElements(GL_TRIANGLES, sScene.cylinderMesh.size_ibo, GL_UNSIGNED_INT,
                   nullptr);
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

  // Cleanup GLFW/GL context.
  windowDelete(window);

  return EXIT_SUCCESS;
}
