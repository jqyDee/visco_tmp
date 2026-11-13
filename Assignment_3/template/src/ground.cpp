#include <algorithm>

#include "ground.h"
#include "mygl/geometry.h"

float getDisplacementValue(WaveParams waveParams, Vector2D position) {
  return waveParams.amplitude *
         static_cast<float>(cos(dot(normalize(waveParams.direction), position) *
                                waveParams.omega));
}

float groundDisplacement(const std::vector<WaveParams> &waveParamsVec,
                         Vector2D position) {
  float displacement = 0.0f;
  for (WaveParams waveParams : waveParamsVec) {
    displacement += getDisplacementValue(waveParams, position);
  }
  return displacement;
}

Ground groundCreate(const Vector3D &lowHeightColor,
                    const Vector3D &highHeightColor) {
  Ground ground;
  ground.vertices.resize(grid::vertexPos.size());

  // Find highest and lowest displacement.
  float minYPos = std::numeric_limits<float>::max();
  float maxYPos = std::numeric_limits<float>::lowest();
  for (unsigned i = 0; i < ground.vertices.size(); i++) {
    float displacement = groundDisplacement(
        ground.waveParamsVec, {grid::vertexPos[i].x, grid::vertexPos[i].z});
    minYPos = std::min(minYPos, displacement);
    maxYPos = std::max(maxYPos, displacement);
  }

  // Transform the grid's vertices and add them to the flag with the correct
  // color.
  for (unsigned i = 0; i < ground.vertices.size(); i++) {
    Vector3D vertexPos = grid::vertexPos[i];
    float displacement =
        groundDisplacement(ground.waveParamsVec, {vertexPos.x, vertexPos.z});
    float displacementFactor = (displacement - minYPos) / (maxYPos - minYPos);
    Vector3D color = lowHeightColor * (1.0f - displacementFactor) +
                     highHeightColor * displacementFactor;

    ground.vertices[i] = {vertexPos + Vector3D{0.f, displacement, 0.f}, color};
  }
  ground.mesh = meshCreate(ground.vertices, grid::indices, GL_DYNAMIC_DRAW,
                           GL_STATIC_DRAW);

  return ground;
}

void groundDelete(Ground &ground) { meshDelete(ground.mesh); }
