#include "mesh.h"

#include <cfloat>
#include <cmath>
#include <fstream>
#include <cstring>

#include <math_functions.h>

#include <algorithm>


string extract_suffix(string str) {
    string suffix;
    size_t pos = str.rfind('.');
    if (pos != string::npos) {
        suffix = str.substr(pos + 1);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
    }
    return suffix;
}


bool read_mesh(const string& filename, vector<float>& V, vector<int>& F) {
  bool succ = false;
  string suffix = extract_suffix(filename);
  if (suffix == "obj") {
    succ = read_obj(filename, V, F);
  } else if (suffix == "off") {
    succ = read_off(filename, V, F);
  } else {
    //cout << "Error : Unsupported file formate!" << endl;
  }
  return succ;
}


bool write_mesh(const string& filename, const vector<float>& V, const vector<int>& F) {
  bool succ = false;
  string suffix = extract_suffix(filename);
  if (suffix == "obj") {
    succ = write_obj(filename, V, F);
  } else if (suffix == "off") {
    //succ = write_off(filename, V, F); //todo
  } else {
    //cout << "Error : Unsupported file formate!" << endl;
  }
  return succ;
}


bool read_obj(const string& filename, vector<float>& V, vector<int>& F) {
  std::ifstream infile(filename, std::ifstream::binary);
  if (!infile) {
    //std::cout << "Open OBJ file error!" << std::endl;
    return false;
  }

  // get length of file
  infile.seekg(0, infile.end);
  int len = infile.tellg();
  infile.seekg(0, infile.beg);

  // load the file into memory
  char* buffer = new char[len + 1];
  infile.read(buffer, len);
  buffer[len] = 0;
  infile.close();

  // parse buffer data
  vector<char*> pVline, pFline;
  char* pch = strtok(buffer, "\n");
  while (pch != nullptr) {
    if (pch[0] == 'v' && pch[1] == ' ') {
      pVline.push_back(pch + 2);
    } else if (pch[0] == 'f' && pch[1] == ' ') {
      pFline.push_back(pch + 2);
    }

    pch = strtok(nullptr, "\n");
  }

  // load V
  V.resize(3 * pVline.size());
  //#pragma omp parallel for
  for (int i = 0; i < pVline.size(); i++) {
    //!!! strtok() is not thread safe in some platforms
    char* p = strtok(pVline[i], " ");
    for (int j = 0; j < 3; j++) {
      V[3 * i + j] = atof(p);
      p = strtok(nullptr, " ");
    }
  }

  // load F
  F.resize(3 * pFline.size());
  //#pragma omp parallel for
  for (int i = 0; i < pFline.size(); i++) {
    char* p = strtok(pFline[i], " ");
    for (int j = 0; j < 3; j++) {
      F[3 * i + j] = atoi(p) - 1;
      p = strtok(nullptr, " ");
    }
  }

  // release
  delete[] buffer;
  return true;
}

bool write_obj(const string& filename, const vector<float>& V, const vector<int>& F) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;

  int nv = V.size() / 3;
  int nf = F.size() / 3;
  if (V.size() % 3 != 0 || F.size() % 3 != 0) return false;
  const int len = 64;
  char* buffer = new char[(nv + nf) * len];

  // convert to string
  char* pV = buffer;
  #pragma omp parallel for
  for (int i = 0; i < nv; i++) {
    int ix3 = i * 3;
    sprintf(pV + i * len, "v %.6g %.6g %.6g\n", V[ix3], V[ix3 + 1], V[ix3 + 2]);
  }

  char* pF = buffer + nv * len;
  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    sprintf(pF + i * len, "f %d %d %d\n", F[ix3] + 1, F[ix3 + 1] + 1, F[ix3 + 2] + 1);
  }

  // shrink
  int k = 0;
  for (int i = 0; i < nv; i++) {
    for (int j = len * i; j < len * (i + 1); j++) {
      if (pV[j] == 0) break;
      buffer[k++] = pV[j];
    }
  }
  for (int i = 0; i < nf; i++) {
    for (int j = len * i; j < len * (i + 1); j++) {
      if (pF[j] == 0) break;
      buffer[k++] = pF[j];
    }
  }

  // write into file
  outfile.write(buffer, k);

  // close file
  outfile.close();
  delete[] buffer;
  return true;
}

bool read_off(const string& filename, vector<float>& V, vector<int>& F) {
  std::ifstream infile(filename, std::ios::binary);
  auto r = read_off_from_stream<std::ifstream>(infile, V, F);
  infile.close();
  return r;
}

void compute_face_center(vector<float>& Fc, const vector<float>& V,
    const vector<int>& F) {
  int nf = F.size() / 3;
  Fc.assign(3 * nf, 0);
  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    for (int j = 0; j < 3; j++) {
      int fx3 = F[ix3 + j] * 3;
      for (int k = 0; k < 3; ++k) {
        Fc[ix3 + k] += V[fx3 + k];
      }
    }
    for (int k = 0; k < 3; ++k) {
      Fc[ix3 + k] /= 3.0f;
    }
  }
}

void compute_face_normal(vector<float>& face_normal, vector<float>& face_area,
    const vector<float>& V, const vector<int>& F) {
  int nf = F.size() / 3;
  face_normal.resize(3 * nf);
  face_area.resize(nf);
  const float* pt = V.data();
  const float EPS = 1.0e-10;
  float* normal = face_normal.data();

  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    const float* v0 = pt + F[ix3] * 3;
    const float* v1 = pt + F[ix3 + 1] * 3;
    const float* v2 = pt + F[ix3 + 2] * 3;

    float p01[3], p02[3];
    for (int j = 0; j < 3; ++j) {
      p01[j] = v1[j] - v0[j];
      p02[j] = v2[j] - v0[j];
    }

    float* normal_i = normal + ix3;
    cross_prod(normal_i, p01, p02);
    float len = norm2(normal_i, 3);
    if (len < EPS) len = EPS;
    for (int j = 0; j < 3; ++j) {
      normal_i[j] /= len;
    }

    face_area[i] = len * 0.5;
  }
}