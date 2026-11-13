#pragma once

#include "mygl/base.h"
#include "mygl/mesh.h"

struct WaveParams {
  float amplitude;
  float omega;
  Vector2D direction;
};

struct Ground {
  Mesh mesh;
  std::vector<Vertex> vertices;

  std::vector<WaveParams> waveParamsVec = {
      {0.1f, 0.01f, normalize(Vector2D{0.0f, 1.0f})},
      {0.3f, 0.15f, normalize(Vector2D{0.0f, 1.0f})},
      {0.7f, 0.2f, normalize(Vector2D{1.0f, 0.0f})},
      {0.5f, 0.05f, normalize(Vector2D{-2.0f, 1.0f})},
      {0.3f, 0.4f, normalize(Vector2D{-1.0f, -2.5f})},
  };
};

// Initializes a plane grid to visualize the ground surface. Creates a vector
// of grid vertices and sets up a mesh that can be drawn with OpenGL.
//
// @param lowHeightColor  Color at the lowest height of the ground.
// @param highHeightColor Color at the highest height of the ground.
Ground groundCreate(const Vector3D &lowHeightColor,
                    const Vector3D &highHeightColor);

// Cleanup and delete all OpenGL buffers of the ground mesh.
void groundDelete(Ground &ground);

// Computes the displacement of the ground at the given position
// according to the wave parameters.
//
// @param waveParamsVec Vector of wave parameters defining the waves.
// @param position      2D position on the ground to evaluate.
float groundDisplacement(const std::vector<WaveParams> &waveParamsVec,
                         Vector2D position);
