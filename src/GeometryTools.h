#ifndef GEOMETRY_TOOL_HEADER
#define GEOMETRY_TOOL_HEADER

#include "CustomisedPolyhedron.h"

class Geotools {
public:
	Geotools();

	// Point conversions
	static Point3 point_to_3d(const Point2& p, Plane& pl);

	static Point2 point_to_2d(const Point3& p, Plane& pl);

	// Convex hull
	static void construct_CH(const std::vector<Point2>& pin, std::vector<Point2>& pout);

	// Position determinations
	static bool positive(const Vector3& p, const double c, const Point3& a);

	static bool negative(const Vector3& p, const double c, const Point3& a);

	static bool negative(const Plane& pl, const Point3& a);

	static bool positive(const Plane& pl, const Point3& a);

	static bool positive(const Plane& pl, const Facet_handle fh);

	static bool has_negative_vertex(Facet_iterator& t, const Plane& pl);

	static bool has_positive_vertex(Facet_iterator& t, const Plane& pl);

	static bool has_positive_vertex(Facet_const_iterator& t, const Plane& pl);

	static Plane plane_equation(Facet_iterator& f);

	static Point3 points_centroid(Point3& a, Point3& b, Point3& c, Point3& d);

    static double face_point_vol(const Facet_iterator &f, const Point3 &p);

    static double face_point_vol(const Point3 &p1, const Point3 &p2, const Point3 &p3, const Point3 &p);

    static double point_to_plane_dist(Point3 &p, Plane &pl);

    static Point3 *get_int_point(Point3 &p1, Point3 &p2, Point3 &p3, Point3 &p4);

    static double get_min_y(const Facet_handle &fh);

    static double get_min_y(const Facet_const_handle &fh);

    static double get_min_z(const Facet_handle &fh);

    static double get_max_y(const Facet_handle &fh);

    static Vector3 get_facet_centroid(const Facet_handle &fh);

    static Vector3 get_facet_nomal(const Facet_handle &fh, const Polyhedron &poly);

    static double get_facet_area(const Facet_handle &fh);

    static double triangle_area(Point3 &a, Point3 &b, Point3 &c);

	static Eigen::Matrix3d make_rotation(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

	static Eigen::Quaterniond euler2quaternion(const Eigen::Vector3d& euler);

	static Eigen::Matrix3d quaternion2mat(const Eigen::Quaterniond& q);

	static Eigen::Vector3d mat2euler(const Eigen::Matrix3d& m);

	static Eigen::Quaterniond mat2quaternion(const Eigen::Matrix3d& m);

	static Eigen::Matrix3d euler2mat(const Eigen::Vector3d& euler);

	static Eigen::Vector3d quaternion2euler(const Eigen::Quaterniond& q);

	static Point3 project_to_3d(const Point2& p, const Plane& pl);

	static Point2 project_to_2d(const Point3& p, const Plane& pl);
};

#endif
