#pragma once

#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include "MeshCutEval.h"

namespace py = pybind11;
using namespace pybind11::literals;



class RoboFDM {
public:
    RoboFDM();

    ~RoboFDM();

    void load_tet_mesh(const std::string &file);

    void load_tri_mesh(const std::string &file);

    bool mesh_to_polyhedron(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Polyhedron &poly);

    py::array_t<double> reset(const std::string& meshfile);

    BSPNode apply_old_action(double x, double y, double z, double alpha, double beta);

    BSPNode apply_action(double alpha, double beta, double gamma, Plane& pl);

    BSPNode apply_action(double alpha, double beta, double gamma);

    py::tuple step(py::array_t<double>& input);

    py::array_t<double> render();

    std::string get_poly();

    std::string get_positive_poly();

    bool set_poly(const std::string& str);

    bool plane_cut(py::array_t<double>& input);

    bool plane_cut_both(py::array_t<double>& input);

    py::array_t<double> planes();

    int n_features();

    double get_far_risky_area();

    double get_inrev_risky_area();

    double get_risky_area() const { return initRiskyArea; };

    double get_volume() const { return initVolume; };

    double get_area() const { return initArea; };

private:
    static void load_candidate_dirs(std::vector<Vector3>& candidate_dirs_);

public:
    Polyhedron poly, poly_pos;
    std::vector<double> initResults;
    
    MinSphere bsphere;

    double initVolume;
    double initArea;
    double initRiskyArea;

private:
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;
    bool first_init_;
    py::array_t<double> init_features_;
    std::vector<Vector3> candDirs_;
    Bbox bbox_;
    const int Na = 64, Nb = 16, Nc = 64;
    const int N2 = Na * Nb;
    const int N3 = N2 * Nc;
	const int nThread = 4;
    std::pair<double, double> rPlatform = std::make_pair(20, 0);
	bool newLoaded_;
};
