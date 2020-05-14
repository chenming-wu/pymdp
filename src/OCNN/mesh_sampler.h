#ifndef _MESH_SAMPLER_H
#define _MESH_SAMPLER_H

#include <ctime>
#include <fstream>
#include <math_functions.h>
#include <mesh.h>
#include <points.h>
#include <random>
#include <sstream>

namespace MeshSampler {
    using namespace std;
    void sample_points(vector<float>& pts, vector<float>& normals, const vector<float>& V, const vector<int>& F,
                       float area_unit);
    bool sample(const std::string& input, const std::string& outputFile);
}; // namespace MeshSampler

#endif