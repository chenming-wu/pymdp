#include "GeometryTools.h"
#include <CGAL/convex_hull_2.h>


Geotools::Geotools() {
}

Point3 Geotools::point_to_3d(const Point2& p, Plane& pl) {
	Vector3 basis[2];
	auto pop = pl.point();
	const Vector3 vpop(pop.x(), pop.y(), pop.z());
	basis[0] = pl.base1() / CGAL::sqrt(pl.base1().squared_length());
	basis[1] = pl.base2() / CGAL::sqrt(pl.base2().squared_length());
	Vector3 nr(pl.a(), pl.b(), pl.c());
	const Point3 vi = pop + (p.x() * basis[0] + p.y() * basis[1]);
	return vi;
}

Point2 Geotools::point_to_2d(const Point3& p, Plane& pl) {
    Vector3 basis[2];
    auto pop = pl.point();
    const Vector3 vpop(pop.x(), pop.y(), pop.z());
    basis[0] = pl.base1() / CGAL::sqrt(pl.base1().squared_length());
    basis[1] = pl.base2() / CGAL::sqrt(pl.base2().squared_length());
    const Vector3 ter(pop, p);
    return {ter * basis[0], ter * basis[1]};
}


Point3 Geotools::project_to_3d(const Point2& p, const Plane& pl) {
    Vector3 basis[2];
    auto pop = pl.point();
    const Vector3 vpop(pop.x(), pop.y(), pop.z());
    basis[0] = pl.base1() / CGAL::sqrt(pl.base1().squared_length());
    basis[1] = pl.base2() / CGAL::sqrt(pl.base2().squared_length());
    Vector3 nr(pl.a(), pl.b(), pl.c());
    Vector3 vi = p.x() * basis[0] + p.y() * basis[1] + vpop;
    return {vi.x(), vi.y(), vi.z()};
}

Point2 Geotools::project_to_2d(const Point3& p, const Plane& pl) {
    Vector3 basis[2];
    auto pop = pl.point();
    const Vector3 vpop(pop.x(), pop.y(), pop.z());
    basis[0] = pl.base1() / CGAL::sqrt(pl.base1().squared_length());
    basis[1] = pl.base2() / CGAL::sqrt(pl.base2().squared_length());
    Vector3 ter(pop, p);
    return {ter * basis[0], ter * basis[1]};
}

void Geotools::construct_CH(const std::vector<Point2>& pin, std::vector<Point2>& pout) {
	CGAL::convex_hull_2(pin.begin(), pin.end(), std::back_inserter(pout));
}

bool Geotools::positive(const Vector3& p, const double c, const Point3& a) {
    const Vector3 probe(a.x(), a.y(), a.z());
    return probe * p + c > 1e-8;
}

bool Geotools::negative(const Vector3& p, const double c, const Point3& a) {
	const Vector3 probe(a.x(), a.y(), a.z());
	return probe * p + c < -1e-8;
}

bool Geotools::negative(const Plane& pl, const Point3& a) {
	const Vector3 p(pl.a(), pl.b(), pl.c());
	return negative(p, pl.d(), a);
}

bool Geotools::positive(const Plane& pl, const Point3& a) {
	const Vector3 p(pl.a(), pl.b(), pl.c());
	return positive(p, pl.d(), a);
}

bool Geotools::positive(const Plane& pl, const Facet_handle fh) {
	auto he = fh->halfedge();
	const auto nd = he;
	auto is_positive = false;
	do {
		if (positive(pl, he->vertex()->point())) {
			is_positive = true;
			break;
		}
		he = he->next();
	} while (he != nd);
	return is_positive;
}

bool Geotools::has_negative_vertex(Facet_iterator& t, const Plane& pl) {
	return (negative(pl, t->halfedge()->vertex()->point()) ||
		negative(pl, t->halfedge()->prev()->vertex()->point()) ||
		negative(pl, t->halfedge()->next()->vertex()->point()));
}

bool Geotools::has_positive_vertex(Facet_iterator& t, const Plane& pl) {
	return (positive(pl, t->halfedge()->vertex()->point()) ||
		positive(pl, t->halfedge()->prev()->vertex()->point()) ||
		positive(pl, t->halfedge()->next()->vertex()->point()));
}

bool Geotools::has_positive_vertex(Facet_const_iterator& t, const Plane& pl) {
	return (positive(pl, t->halfedge()->vertex()->point()) ||
		positive(pl, t->halfedge()->prev()->vertex()->point()) ||
		positive(pl, t->halfedge()->next()->vertex()->point()));
}

Plane Geotools::plane_equation(Facet_iterator& f) {
    auto h = f->halfedge();
    return {h->vertex()->point(),
            h->next()->vertex()->point(),
            h->next()->next()->vertex()->point()};
}

Point3 Geotools::points_centroid(Point3& a, Point3& b, Point3& c, Point3& d) {
    return {(a.x() + b.x() + c.x() + d.x()) / 4.0, (a.y() + b.y() + c.y() + d.y()) / 4.0,
            (a.z() + b.z() + c.z() + d.z()) / 4.0};
}

double Geotools::face_point_vol(const Facet_iterator& f, const Point3& p) {
	Halfedge_handle h = f->halfedge();
	Tetra t(h->next()->next()->vertex()->point(), h->next()->vertex()->point(), h->vertex()->point(), p);
	if (t.is_degenerate()) return 0.;
	else return t.volume();
}

double Geotools::face_point_vol(const Point3& p1, const Point3& p2, const Point3& p3, const Point3& p) {
	Tetra t(p3, p2, p1, p);
	if (t.is_degenerate()) return 0.;
	else return t.volume();
}

double Geotools::point_to_plane_dist(Point3& p, Plane& pl) {
	Vector3 cut(pl.a(), pl.b(), pl.c());
	Vector3 vp(p.x(), p.y(), p.z());
	double C = pl.d();
	cut = cut / CGAL::sqrt(cut.squared_length());
	C = C / CGAL::sqrt(cut.squared_length());
	return (cut * vp + C);
}

Point3* Geotools::get_int_point(Point3& p1, Point3& p2, Point3& p3, Point3& p4) {
	const Segment3 s1(p1, p2);
	const Segment3 s2(p3, p4);
	CGAL::cpp11::result_of<K::Intersect_3(Segment3, Segment3)>::type resi = CGAL::intersection(s1, s2);
	if (resi) {
		if (Point3* l = boost::get<Point3>(&*resi)) {
			return l;
		}
	}
	return nullptr;
}

double Geotools::get_min_y(const Facet_handle &fh) {
    Halfedge_handle he = fh->halfedge();
    Point3 &p1 = he->prev()->vertex()->point();
    Point3 &p2 = he->vertex()->point();
    Point3 &p3 = he->next()->vertex()->point();
    return std::min(p1.y(), std::min(p2.y(), p3.y()));
}

double Geotools::get_min_y(const Facet_const_handle &fh) {
    Halfedge_const_handle he = fh->halfedge();
    const Point3 &p1 = he->prev()->vertex()->point();
    const Point3 &p2 = he->vertex()->point();
    const Point3 &p3 = he->next()->vertex()->point();
    return std::min(p1.y(), std::min(p2.y(), p3.y()));
}

double Geotools::get_max_y(const Facet_handle &fh) {
    Halfedge_handle he = fh->halfedge();
    Point3 &p1 = he->prev()->vertex()->point();
    Point3 &p2 = he->vertex()->point();
    Point3 &p3 = he->next()->vertex()->point();
    return std::max(p1.y(), std::max(p2.y(), p3.y()));
}

double Geotools::get_min_z(const Facet_handle &fh) {
    Halfedge_handle he = fh->halfedge();
    Point3 &p1 = he->prev()->vertex()->point();
    Point3 &p2 = he->vertex()->point();
    Point3 &p3 = he->next()->vertex()->point();
    return std::min(p1.z(), std::min(p2.z(), p3.z()));
}


Vector3 Geotools::get_facet_centroid(const Facet_handle& fh) {
    Halfedge_handle he = fh->halfedge();
    Point3 &p1 = he->prev()->vertex()->point();
    Point3 &p2 = he->vertex()->point();
    Point3 &p3 = he->next()->vertex()->point();
    return {(p1.x() + p2.x() + p3.x()) / 3.0, (p1.y() + p2.y() + p3.y()) / 3.0,
            (p1.z() + p2.z() + p3.z()) / 3.0};
}

Vector3 Geotools::get_facet_nomal(const Facet_handle& fh, const Polyhedron& poly) {
	return CGAL::Polygon_mesh_processing::compute_face_normal(fh, poly);
}

double Geotools::triangle_area(Point3& a, Point3& b, Point3& c) {
	return CGAL::sqrt(CGAL::squared_area(a, b, c));
}

double Geotools::get_facet_area(const Facet_handle& fh) {
	Halfedge_handle he = fh->halfedge();
	Point3& p1 = he->prev()->vertex()->point();
	Point3& p2 = he->vertex()->point();
	Point3& p3 = he->next()->vertex()->point();
	return triangle_area(p1, p2, p3);
}

Eigen::Matrix3d Geotools::make_rotation(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
	double angle = std::atan2(a.cross(b).norm(), a.dot(b)) / 2;
	Eigen::Vector3d r_axis = a.cross(b);
	if (std::abs(angle) < 1e-8) {
		r_axis = a;
	}
	r_axis.normalize();
	r_axis = r_axis * std::sin(angle);
	Eigen::Quaterniond q(std::cos(angle), r_axis.x(), r_axis.y(), r_axis.z());
	return quaternion2mat(q);
}

Eigen::Quaterniond Geotools::euler2quaternion(const Eigen::Vector3d& euler) {
	double cr = cos(euler(0) / 2);
	double sr = sin(euler(0) / 2);
	double cp = cos(euler(1) / 2);
	double sp = sin(euler(1) / 2);
	double cy = cos(euler(2) / 2);
	double sy = sin(euler(2) / 2);
	Eigen::Quaterniond q;
	q.w() = cr * cp * cy + sr * sp * sy;
	q.x() = sr * cp * cy - cr * sp * sy;
	q.y() = cr * sp * cy + sr * cp * sy;
	q.z() = cr * cp * sy - sr * sp * cy;
	return q;
}

Eigen::Matrix3d Geotools::quaternion2mat(const Eigen::Quaterniond& q) {
	Eigen::Matrix3d m;
	double a = q.w(), b = q.x(), c = q.y(), d = q.z();
	m << a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
		2 * (b * c + a * d), a* a - b * b + c * c - d * d, 2 * (c * d - a * b),
		2 * (b * d - a * c), 2 * (c * d + a * b), a* a - b * b - c * c + d * d;
	return m;
}

Eigen::Vector3d Geotools::mat2euler(const Eigen::Matrix3d& m) {
	double r = atan2(m(2, 1), m(2, 2));
	double p = asin(-m(2, 0));
	double y = atan2(m(1, 0), m(0, 0));
	Eigen::Vector3d rpy(r, p, y);
	return rpy;
}

Eigen::Quaterniond Geotools::mat2quaternion(const Eigen::Matrix3d& m) {
	//return euler2quaternion(mat2euler(m));
	Eigen::Quaterniond q;
	double a, b, c, d;
	a = sqrt(1 + m(0, 0) + m(1, 1) + m(2, 2)) / 2;
	b = (m(2, 1) - m(1, 2)) / (4 * a);
	c = (m(0, 2) - m(2, 0)) / (4 * a);
	d = (m(1, 0) - m(0, 1)) / (4 * a);
	q.w() = a;
	q.x() = b;
	q.y() = c;
	q.z() = d;
	return q;
}

Eigen::Matrix3d Geotools::euler2mat(const Eigen::Vector3d& euler) {
	double cr = cos(euler(0));
	double sr = sin(euler(0));
	double cp = cos(euler(1));
	double sp = sin(euler(1));
	double cy = cos(euler(2));
	double sy = sin(euler(2));
	Eigen::Matrix3d m;
	m << cp * cy, -cr * sy + sr * sp * cy, sr* sy + cr * sp * cy,
		cp* sy, cr* cy + sr * sp * sy, -sr * cy + cr * sp * sy,
		-sp, sr* cp, cr* cp;
	return m;
}

Eigen::Vector3d Geotools::quaternion2euler(const Eigen::Quaterniond& q) {
	return mat2euler(quaternion2mat(q));
}



//gdiam_bbox Searcher::get_mvbb(gdiam_real *points, int num) {
//	gdiam_point *pnt_arr;
//	gdiam_bbox bb;
//	pnt_arr = gdiam_convert(points, num);
//	bb = gdiam_approx_mvbb_grid_sample(pnt_arr, num, 5, 400);
//
//	return bb;
//}
