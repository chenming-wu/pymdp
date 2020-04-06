#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/exceptions.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "FillHoleCDT.h"
#include "PlaneCut.h"
#include "RoboFDM.h"

#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

std::default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
std::uniform_real_distribution<float> distribution(0.01, 0.99);

RoboFDM::RoboFDM() : newLoaded_(true), first_init_(false) { load_candidate_dirs(candDirs_); }

RoboFDM::~RoboFDM() {}

BSPNode RoboFDM::apply_action(double alpha, double beta, double gamma, Plane& pl) {
  // TODO: Apply this action to FineDecomposition
  const auto plt = MeshCutEval::get_platform_cross(rPlatform.first, rPlatform.second);
  pl = MeshCutEval::convert_abg_to_plane(alpha, beta, gamma, bsphere);
  auto res = MeshCutEval::apply_plane_cut(poly, pl, bbox_, plt);
  return res;
}

BSPNode RoboFDM::apply_action(double alpha, double beta, double gamma) {
  // TODO: Apply this action to FineDecomposition
  const auto plt = MeshCutEval::get_platform_cross(rPlatform.first, rPlatform.second);
  auto pl = MeshCutEval::convert_abg_to_plane(alpha, beta, gamma, bsphere);
  auto res = MeshCutEval::apply_plane_cut(poly, pl, bbox_, plt);
  return res;
}

py::tuple RoboFDM::step(py::array_t<double>& input) {
  py::tuple data(5);
  py::buffer_info buf = input.request();
  auto* ptr = (double*)buf.ptr;
  Plane pl;

  const auto plt = MeshCutEval::get_platform_cross(rPlatform.first, rPlatform.second);
  if (input.size() == 3)
    pl = MeshCutEval::convert_abg_to_plane(ptr[0], ptr[1], ptr[2], bsphere);
  else
    pl = Plane(ptr[0], ptr[1], ptr[2], ptr[3]);

  auto res = MeshCutEval::apply_plane_cut(poly, pl, bbox_, plt);

  data[0] = res.first;
  data[1] = res.second;
  data[2] = res.third;
  data[3] = res.fourth;
  data[4] = res.fifth;
  return data;
}

bool RoboFDM::plane_cut(py::array_t<double>& input) {
  auto* data = input.data();
  Plane pl;
  if (input.size() == 3) {
    pl = MeshCutEval::convert_abg_to_plane(data[0], data[1], data[2], bsphere);
  } else {
    pl = Plane(data[0], data[1], data[2], data[3]);
  }

  PlaneCutter pc;
  Polyhedron _;
  pc.cut_and_fill<FillHoleCDT>(poly, _, pl);
  return true;
}

bool RoboFDM::plane_cut_both(py::array_t<double>& input) {
  auto* data = input.data();
  Plane pl;
  if (input.size() == 3) {
    pl = MeshCutEval::convert_abg_to_plane(data[0], data[1], data[2], bsphere);
  } else {
    pl = Plane(data[0], data[1], data[2], data[3]);
  }

  poly_pos.clear();
  PlaneCutter pc;
  pc.cut_and_fill_both<FillHoleCDT>(poly, poly_pos, pl);
  return true;
}

py::array_t<double> RoboFDM::planes() {
  const auto dim = 3;

  // auto result = py::array_t<double>({Na, Nb, Nc, dim});
  auto result = py::array_t<double>({N3, dim});
  py::buffer_info buf_result = result.request();
  auto* arrEvalRes = (double*)buf_result.ptr;

#pragma omp parallel for
  for (auto i = 0; i < N3; i++) {
    const auto pos_dir = i % N2;
    const auto pos_off = static_cast<double>(i / N2);

    const auto gamma = pos_off / static_cast<double>(Nc);
    const auto alpha = static_cast<double>(pos_dir / Nb) / static_cast<double>(Na);
    const auto beta = static_cast<double>(pos_dir % Nb) / static_cast<double>(Nb);

    arrEvalRes[i * dim + 0] = alpha;
    arrEvalRes[i * dim + 1] = beta;
    arrEvalRes[i * dim + 2] = gamma;
  }
  return result;
}

double RoboFDM::get_far_risky_area() {
  double sumArea = 0.;
  for (auto& f : faces(poly)) {
    if (f->facetCentroid.y() > rPlatform.second + 1.4 && !f->supported()) {
      sumArea += f->area();
    }
  }

  return sumArea;
}

double RoboFDM::get_inrev_risky_area() {
  omp_set_num_threads(nThread); // Set Thread Number
  const auto plt = MeshCutEval::get_platform_cross(rPlatform.first, rPlatform.second);

  std::vector<std::vector<BSP_Result>> evalResults;
  evalResults.resize(nThread);

  std::vector<bool> isInrev(poly.size_of_facets(), false);

  int cnt = 0;
  for (auto f : faces(poly)) {
    f->set_patch_id(cnt++);
  }

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < candDirs_.size(); ++i) {
    int tid = omp_get_thread_num();
    const auto& curDir = candDirs_[i];
    MeshCutEval::get_inrev_risky_area(poly, curDir, bbox_, plt, isInrev);
  }

  double sumInrevArea = 0.;
  for (auto f : faces(poly)) {
    if (f->supported())
      continue;

    if (isInrev[f->patch_id()])
      sumInrevArea += f->area();
  }

  return sumInrevArea;
}

py::array_t<double> RoboFDM::reset(const std::string& meshfile) {
  poly.clear();
  std::ifstream f(meshfile);
  f >> poly;
  f.close();
  if (poly.is_valid()) {
    if (!poly.is_closed()) {
      unsigned int nb_holes = 0;
      for (Halfedge_handle h : halfedges(poly)) {
        if (h->is_border()) {
          std::vector<Facet_handle> patch_facets;
          CGAL::Polygon_mesh_processing::triangulate_hole(
              poly, h, std::back_inserter(patch_facets),
              CGAL::Polygon_mesh_processing::parameters::use_delaunay_triangulation(true));

          ++nb_holes;
        }
      }
    }
  }

  rPlatform = MeshCutEval::find_suitable_platform(poly);

  initResults = MeshCutEval::initialize_supports(poly);
  bbox_ = bbox_3(poly.points_begin(), poly.points_end());
  bsphere = MinSphere(poly.points_begin(), poly.points_end());

  initVolume = initResults[0];
  initArea = initResults[1];
  initRiskyArea = initResults[2];

  // std::cout << initResults[0] << " " << initResults[1] << " " << initResults[2] << std::endl;

  return py::array_t<double>({1});

  if (!first_init_) {
    init_features_ = render();
    first_init_ = true;
  }
  return init_features_;
}

py::array_t<double> RoboFDM::render() {
  omp_set_num_threads(nThread); // Set Thread Number
  // auto result = py::array_t<double>({Na, Nb, Nc, dim});

  const auto plt = MeshCutEval::get_platform_cross(rPlatform.first, rPlatform.second);

  std::vector<std::vector<BSP_Result>> evalResults;
  evalResults.resize(nThread);

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < candDirs_.size(); ++i) {
    int tid = omp_get_thread_num();
    const auto& curDir = candDirs_[i];
    MeshCutEval::evaluation_direction(poly, curDir, bbox_, plt, evalResults[tid]);
  }

  int numEvals = 0;
#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < evalResults.size(); ++i) {
#pragma omp critical
    numEvals += evalResults[i].size();
  }

  const auto dim = 9;
  auto result = py::array_t<double>({numEvals, dim});
  py::buffer_info buf_result = result.request();
  auto* arrEvalRes = (double*)buf_result.ptr;

  numEvals = 0;
  for (auto i = 0; i < evalResults.size(); ++i) {
#pragma omp parallel for schedule(dynamic)
    for (auto j = 0; j < evalResults[i].size(); ++j) {
      const auto curIdx = numEvals + j;
      const auto& tmpEval = evalResults[i][j];
      arrEvalRes[dim * curIdx + 0] = tmpEval.get<1>() / initVolume;
      arrEvalRes[dim * curIdx + 1] = tmpEval.get<2>() / initRiskyArea;
      arrEvalRes[dim * curIdx + 2] =
          std::max(0.0, std::min(tmpEval.get<3>() / std::sqrt(bsphere.squared_radius()), 1.0));
      arrEvalRes[dim * curIdx + 3] =
          std::max(0.0, std::min(tmpEval.get<4>() / std::sqrt(bsphere.squared_radius()), 1.0));
      arrEvalRes[dim * curIdx + 4] = tmpEval.get<5>() / initRiskyArea;
      const auto& pl = tmpEval.get<0>();
      arrEvalRes[dim * curIdx + 5] = pl.a();
      arrEvalRes[dim * curIdx + 6] = pl.b();
      arrEvalRes[dim * curIdx + 7] = pl.c();
      arrEvalRes[dim * curIdx + 8] = pl.d();
    }
    numEvals += evalResults[i].size();
  }
  return result;
}

std::string RoboFDM::get_poly() {
  std::stringstream ss;
  ss << poly;
  return ss.str();
}

std::string RoboFDM::get_positive_poly() {
  std::stringstream ss;
  ss << poly_pos;
  return ss.str();
}

bool RoboFDM::set_poly(const std::string& str) {
  std::stringstream ss(str);
  poly.clear();
  ss >> poly;
  if (poly.is_valid()) {
    if (!poly.is_closed()) {
      unsigned int nb_holes = 0;
      for (Halfedge_handle h : halfedges(poly)) {
        if (h->is_border()) {
          std::vector<Facet_handle> patch_facets;
          CGAL::Polygon_mesh_processing::triangulate_hole(
              poly, h, std::back_inserter(patch_facets),
              CGAL::Polygon_mesh_processing::parameters::use_delaunay_triangulation(true));

          ++nb_holes;
        }
      }
    }
    MeshCutEval::initialize_supports(poly);
    return true;
  } else {
    return false;
  }
}
