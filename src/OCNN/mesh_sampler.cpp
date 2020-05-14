#include <mesh_sampler.h>
#include <iostream>

namespace MeshSampler {

  void sample_points(vector<float>& pts, vector<float>& normals, const vector<float>& V, const vector<int>& F,
                   float area_unit) {
  std::default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  vector<float> face_normal, face_center, face_area;
  compute_face_normal(face_normal, face_area, V, F);
  compute_face_center(face_center, V, F);

  // float avg_area = 0;
  // for (auto& a : face_area) { avg_area += a;}
  // avg_area /= face_area.size();
  // area_unit *= avg_area;
  if (area_unit <= 0)
    area_unit = 1.0e-5f;

  int nf = F.size() / 3;
  vector<float> point_num(nf);
  int total_pt_num = 0;
  for (int i = 0; i < nf; ++i) {
    int n = static_cast<int>(face_area[i] / area_unit + 0.5f);
    if (n < 1)
      n = 1; // sample at least one point
    // if (n > 100) n = 100;
    point_num[i] = n;
    total_pt_num += n;
  }

  double sum_area = 0;
  for (auto i = 0; i < face_area.size(); ++i) {
    sum_area += face_area[i];
    face_area[i] = sum_area;
  }

  for (int i = 0; i < nf; ++i) {
    auto s = distribution(generator) * sum_area;
    auto k = std::upper_bound(face_area.begin(), face_area.end(), s);
  }

  pts.resize(3 * total_pt_num);
  normals.resize(3 * total_pt_num);

  for (int i = 0, id = 0; i < total_pt_num; ++i, ++id) {
    auto s = distribution(generator) * sum_area;
    auto k = std::upper_bound(face_area.begin(), face_area.end(), s);
    int ix3 = std::distance(face_area.begin(), k) * 3, idx3 = id * 3;

    float x = 0, y = 0, z = 0;
    while (z < 0.0001 || z > 0.9999) {
      x = distribution(generator);
      y = distribution(generator);
      z = 1.0 - x - y;
    }
    int f0x3 = F[ix3] * 3, f1x3 = F[ix3 + 1] * 3, f2x3 = F[ix3 + 2] * 3;
    for (int k = 0; k < 3; ++k) {
      pts[idx3 + k] = x * V[f0x3 + k] + y * V[f1x3 + k] + z * V[f2x3 + k];
      normals[idx3 + k] = face_normal[ix3 + k];
    }
  }
}

bool sample(const std::string& input, const std::string& outputFile) {
  auto FLAGS_scale = true;
  vector<float> V;
  vector<int> F;

  std::stringstream ss(input);
 
  read_off_from_stream(ss, V, F);

  // scale mesh
  float radius = 1.0, center[3];
  if (FLAGS_scale) {
    bounding_sphere(radius, center, V.data(), V.size() / 3);
    float scale = 1.0 / (2 * radius);
    for (auto& v : V) {
      v *= scale;
    }
  }

  // sample points
  vector<float> pts, normals;
  sample_points(pts, normals, V, F, 1.0);

  // save points
  Points point_cloud;
  point_cloud.set_points(pts, normals);
  point_cloud.write_points(outputFile + ".points");
    
  
  //std::ofstream oFile("o.xyz");
  //for (int j = 0; j < pts.size();) {
  //  oFile << pts[j] << " " << pts[j + 1] << " " << pts[j + 2] << std::endl;
  //  j += 3;
  //}
  //oFile.close();
}
}; // namespace MeshSampler