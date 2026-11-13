#pragma once

#include "base.h"

struct ShaderProgram {
  GLuint id = 0;
  GLuint _vertexID = 0;
  GLuint _fragmentID = 0;
};

// Function to load vertex and fragment shader from file and compile and
// link them to create shader program.
//
// @param vertexPath Path to vertex shader file.
// @param fragmentPath Path to fragment shader file.
// @return Shader program.
ShaderProgram shaderLoad(const std::string &vertexPath,
                         const std::string &fragmentPath);

// Function to compile and link vertex and fragment source strings to
// create shader program.
//
// @param vertexSource Source string holding vertex shader code.
// @param fragmentSource Source string holding fragment shader code.
// @return Shader program.
ShaderProgram shaderCreate(const std::string &vertexSource,
                           const std::string &fragmentSource);

// Cleanup and delete all shaders of a shader program and the program
// itself. Has to be called for each shader program after it is not used
// anymore.
//
// @param program Shader program to delete.
void shaderDelete(const ShaderProgram &program);

// Function to set uniform in shader program.
//
// @param shader Shader program.
// @param name Uniform name.
// @param value Value to which the uniform should be set.
void shaderUniform(ShaderProgram &shader, const std::string &name,
                   const Matrix4D &value);

// Function to set uniform in shader program.
//
// @param shader Shader program.
// @param name Uniform name.
// @param value Value to which the uniform should be set.
void shaderUniform(ShaderProgram &shader, const std::string &name, int value);
