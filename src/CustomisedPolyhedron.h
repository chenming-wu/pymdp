#ifndef CUSTOMISED_POLYHEDRON_H
#define CUSTOMISED_POLYHEDRON_H
#include <Eigen/Dense>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Random.h>
#include <CGAL/AABB_halfedge_graph_segment_primitive.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Min_sphere_d.h>
#include <CGAL/Min_sphere_annulus_d_traits_3.h>

#include <CGAL/Polygon_mesh_slicer.h>
//#define USEDEBUG
#ifdef USEDEBUG
#define Debug(x) std::cout << x
#else
#define Debug(x)
#endif 

#define NDebug(x) std::cerr << x

// Microsoft Visual Studio 2019 doesn't have M_PI anymore
#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

template <typename Refs, typename Tag, typename Point, typename Patch_id, class Vector_>
class Polyhedron_demo_vertex :
	public CGAL::HalfedgeDS_vertex_base<Refs, Tag, Point>
{
public:
	typedef std::set<Patch_id> Set_of_indices;
	Vector_ vertexNormal;
private:
	typedef CGAL::HalfedgeDS_vertex_base<Refs, Tag, Point> Pdv_base;

	Set_of_indices indices;
	std::size_t mID;
	std::size_t time_stamp_;

public:
	int nb_of_feature_edges;

	bool is_corner() const
	{
		return nb_of_feature_edges > 2;
	}

	bool is_feature_vertex() const
	{
		return nb_of_feature_edges != 0;
	}

	void add_incident_patch(const Patch_id i)
	{
		indices.insert(i);
	}

	/// For the determinism of Compact_container iterators
	///@{
	typedef CGAL::Tag_true Has_timestamp;

	std::size_t time_stamp() const
	{
		return time_stamp_;
	}

	void set_time_stamp(const std::size_t& ts)
	{
		time_stamp_ = ts;
	}

	///}@

	const Set_of_indices&
		incident_patches_ids_set() const
	{
		return indices;
	}

	std::size_t& id() { return mID; }
	std::size_t id() const { return mID; }

	Polyhedron_demo_vertex() : Pdv_base(), mID(-1), nb_of_feature_edges(0)
	{
	}

	Polyhedron_demo_vertex(const Point& p) : Pdv_base(p), mID(-1), nb_of_feature_edges(0)
	{
	}
};


template <class Refs, class Tprev, class Tvertex, class Tface>
class Polyhedron_demo_halfedge :
	public CGAL::HalfedgeDS_halfedge_base<Refs, Tprev, Tvertex, Tface>
{
private:
	bool feature_edge;
	std::size_t time_stamp_;
	std::size_t mask_;

public:

	Polyhedron_demo_halfedge()
		: feature_edge(false), mask_(0)
	{
	};

	bool is_feature_edge() const
	{
		return feature_edge;
	}

	void set_feature_edge(const bool b)
	{
		feature_edge = b;
		this->opposite()->feature_edge = b;
	}

	std::size_t& mask() { return mask_; }
	std::size_t mask() const { return mask_; }

	void set_mask(std::size_t m) { mask_ = m; }

	/// For the determinism of Compact_container iterators
	///@{
	typedef CGAL::Tag_true Has_timestamp;

	std::size_t time_stamp() const
	{
		return time_stamp_;
	}

	void set_time_stamp(const std::size_t& ts)
	{
		time_stamp_ = ts;
	}

	///@}
};

// Defined facet base
// auxID is used in customised_mesh_slicer, to identify the id of cross-section's faces
template <class Refs, class T_, class Pln_, class Patch_id_, class Vector_>
class Polyhedron_demo_face :
	public CGAL::HalfedgeDS_face_base<Refs, T_, Pln_>
{
public:
	Patch_id_ patch_id_;

	int auxID;
	std::size_t time_stamp_;
	bool isVisited;
	Vector_ facetNormal;
	Vector_ facetCentroid;

	bool isSupported;
	double area_;
	//std::vector<int> vecIDs;
public:
	typedef Patch_id_ Patch_id;

	Polyhedron_demo_face()
		: patch_id_(1), isVisited(false), isSupported(true), auxID(-1),
		area_(0)
	{
		//vecIDs.reserve(1);
	}

	int patch_id() const
	{
		return patch_id_;
	}

	void set_patch_id(const int i)
	{
		patch_id_ = i;
	}

	void set_face_area(const double a)
	{
		area_ = a;
	}

	const double& area() { return area_; }
	const double area() const { return area_; }

	bool& visited() { return isVisited; }
	bool visited() const { return isVisited; }

	bool& supported() { return isSupported; }
	bool supported() const { return isSupported; }

	/// For the determinism of Compact_container iterators
	///@{
	typedef CGAL::Tag_true Has_timestamp;

	std::size_t time_stamp() const
	{
		return time_stamp_;
	}

	void set_time_stamp(const std::size_t& ts)
	{
		time_stamp_ = ts;
	}

	///@}
};

template <typename Patch_id>
class Polyhedron_demo_items : public CGAL::Polyhedron_items_3
{
public:
	// wrap vertex
	template <class Refs, class Traits>
	struct Vertex_wrapper
	{
		typedef typename Traits::Point_3 Point;
		typedef Polyhedron_demo_vertex<Refs,
			CGAL::Tag_true,
			Point,
			Patch_id,
			typename Traits::Vector_3> Vertex;
	};

	// wrap face
	template <class Refs, class Traits>
	struct Face_wrapper
	{
		typedef Polyhedron_demo_face<Refs,
			CGAL::Tag_true,
			typename Traits::Plane_3,
			Patch_id,
			typename Traits::Vector_3> Face;
	};

	// wrap halfedge
	template <class Refs, class Traits>
	struct Halfedge_wrapper
	{
		typedef Polyhedron_demo_halfedge<Refs,
			CGAL::Tag_true,
			CGAL::Tag_true,
			CGAL::Tag_true> Halfedge;
	};
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Tetrahedron_3<K> Tetra;
typedef CGAL::Polyhedron_3<K, Polyhedron_demo_items<int>> Polyhedron;
typedef Polyhedron::Vertex_handle Vertex_handle;
typedef Polyhedron::Facet_handle Facet_handle;
typedef Polyhedron::Halfedge_handle Halfedge_handle;
typedef Polyhedron::Edge_iterator Edge_iterator;
typedef Polyhedron::Facet_iterator Facet_iterator;
typedef Polyhedron::Halfedge_const_iterator Halfedge_iterator;
typedef Polyhedron::Facet::Halfedge_around_facet_const_circulator HF_circulator;

// constant typedefs
typedef Polyhedron::Edge_const_iterator Edge_const_iterator;
typedef Polyhedron::Halfedge_const_handle Halfedge_const_handle;
typedef Polyhedron::Halfedge_const_iterator Halfedge_const_iterator;
typedef Polyhedron::Vertex_const_handle Vertex_const_handle;
typedef Polyhedron::Vertex_const_iterator Vertex_const_iterator;
typedef Polyhedron::Facet_const_handle Facet_const_handle;
typedef Polyhedron::Facet_const_iterator Facet_const_iterator;

typedef K::Point_3 Point3;
typedef K::Plane_3 Plane;
typedef K::Vector_3 Vector3;
typedef K::Line_3 Line3;
typedef K::Triangle_3 Triangle3;
typedef K::Segment_3 Segment3;
typedef K::Ray_3 Ray3;
typedef K::Point_2 Point2;
typedef K::Vector_2 Vector2;
typedef CGAL::Polygon_2<K> Polygon2;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> FGTP;
typedef CGAL::AABB_traits<K, FGTP> AABB_traits_FGTP;
typedef CGAL::AABB_tree<AABB_traits_FGTP> AABB_tree_FGTP;
typedef AABB_tree_FGTP::Primitive_id Primitive_id;
typedef boost::optional<AABB_tree_FGTP::Intersection_and_primitive_id<Ray3>::Type> Ray_intersection;
typedef CGAL::Polygon_mesh_slicer<Polyhedron, K> Slicer;
typedef std::vector<std::vector<Point3>> Polylines;
typedef CGAL::Bbox_3 Bbox;
typedef CGAL::Min_sphere_annulus_d_traits_3<K> Traits;
typedef CGAL::Min_sphere_d<Traits>             MinSphere;

typedef struct bsp {
	bsp()
	{
		initialize();
	}

	void initialize()
	{
		first = 0.0;
		second = 0.0;
		third = 0.0;
		fourth = 0.0;
		fifth = 0.0;
		collided = false;
		support_free = true;
		done = false;
	}
	double first;
	double second;
	double third;
	double fourth;
	double fifth;
	bool collided;
	bool support_free;
	bool done;
} BSPNode;

typedef boost::tuple<Plane, double, double, double, double, double, bool> BSP_Result;

#endif //ICRA17_BJUT_CUSTOMISED_POLYHEDRON_H