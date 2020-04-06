#pragma once
//#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/bounding_box.h>
#include <CGAL/Bbox_3.h>
#include <vector>
#include <list>
#include <random>

#include "CustomisedPolyhedron.h"
#include "GeometryTools.h"

constexpr double alpha_max_ = 0.70711;
constexpr double FRAGILE_EPS = 0.9;
constexpr double CONNECT_EPS = 1;

class MeshCutEval {
public:
    typedef std::unordered_map<Halfedge_const_handle, Point3> MapEdgePoint;
    typedef boost::tuple<double, double> TupleRisky;

    // Functions
    MeshCutEval();

    ~MeshCutEval();

    static bool initialization(const Polyhedron &poly);

    static BSPNode
    apply_plane_cut(const Polyhedron &poly_, const Plane &pl, const Bbox &box, const std::vector<Point3> &plt);

    // Evaluate and cut
    static bool cut_with_plane(const Polyhedron &poly_, const Plane &plane, MapEdgePoint &mapEdgePoint,
                               std::set<Facet_const_iterator> &intFaceSet);

    static std::vector<double>
    euler_update_facets(const Polyhedron &poly_, const Plane &pl, const std::set<Facet_const_iterator> &intFaceSet);

    static BSPNode exact_cutting_volume_and_area(const Polyhedron &poly_, const Plane &pl,
                                                 const MapEdgePoint &map_edge_point,
                                                 std::vector<BSPNode> &vec_bsps);

    static BSPNode exact_cutting_volume_and_area(const Polyhedron &poly_, const Plane &pl,
                                                 const MapEdgePoint &map_edge_point,
                                                 std::vector<BSPNode> &vec_bsps, std::vector<bool> &inRev);


    static bool is_supported(Vector3 &f_normal, const Vector3 &dir);

    static std::vector<double> initialize_supports(Polyhedron &poly_);

    static std::vector<Point3> get_platform_cross(double radius, double platformZ);

    static Plane convert_xyzab_to_plane(double x, double y, double z, double alpha, double beta, Bbox &box_);

    static Plane convert_abg_to_plane(double alpha, double beta, double gamma, const MinSphere& sphere);

    static void evaluation_direction(const Polyhedron& poly, const Vector3& current_dir, const Bbox& box, const std::vector<Point3>& plt, std::vector<BSP_Result>& vec_tu);

    static std::pair<double, double> find_suitable_platform(const Polyhedron& poly);

    static void get_inrev_risky_area(const Polyhedron& poly, const Vector3& current_dir, const Bbox& box, const std::vector<Point3>& plt, std::vector<bool>& inRev);

private:
    static double compute_min_d(const Vector3& dir, const Bbox& box_);

    static double compute_max_d(const Vector3& dir, const Bbox& box_);
};
