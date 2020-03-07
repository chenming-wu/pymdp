#include "FillHoleCDT.h"
#include <CGAL/Modifier_base.h>
#include <CGAL/HalfedgeDS_decorator.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Triangulation_2_projection_traits_3.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Polygon_mesh_processing/refine.h>

template <class Polyhedron, class Kernel >
class Triangulate_modifier
	: public CGAL::Modifier_base<typename Polyhedron::HalfedgeDS>
{
	typedef typename Polyhedron::HalfedgeDS HDS;
	typedef typename Polyhedron::Traits Traits;
	typedef typename Polyhedron::Halfedge_iterator Halfedge_iterator;
	typedef typename Polyhedron::Halfedge_handle Halfedge_handle;
	typedef typename Polyhedron::Facet Facet;
	typedef typename Polyhedron::Facet_iterator Facet_iterator;
	typedef typename Polyhedron::Facet_handle Facet_handle;
	typedef typename Kernel::Plane_3 Plane;
	typedef typename Kernel::Vector_3 Vector;
	typedef typename Kernel::Point_3 Point;
	typedef CGAL::Triangulation_2_projection_traits_3<Traits>   P_traits;
	typedef CGAL::Triangulation_vertex_base_with_info_2<Halfedge_handle,
		P_traits>        Vb;

	// 定义Constrained Delaunay Triangulation(CDT)中的Face_Information
	struct Face_info {
		typename Polyhedron::Halfedge_handle e[3];
		int is_external;
		void set_in_domain(const bool f)
		{
			is_external = f;
		}

		void is_in_domain()
		{
			return is_external;
		}
	};

	typedef CGAL::Triangulation_face_base_with_info_2<Face_info,
		P_traits>          Fb1;
	typedef CGAL::Constrained_triangulation_face_base_2<P_traits, Fb1>   Fb2;
	typedef CGAL::Delaunay_mesh_face_base_2<P_traits, Fb2> Fb;
	typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                  TDS;
	typedef CGAL::Exact_predicates_tag	                               Itag;
	typedef CGAL::Constrained_Delaunay_triangulation_2<P_traits,
		TDS,
		Itag>             CDTbase;
	typedef CGAL::Constrained_triangulation_plus_2<CDTbase>              CDT;
	typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

public:
	Triangulate_modifier(std::vector<Facet_iterator>& list_faces, Vector3& nr) : list_faces_(list_faces), normal(nr) {}
	std::vector<Facet_iterator>& list_faces_;
	Vector3 normal;

	bool is_external(typename CDT::Face_handle fh) const {
		return (fh->info().is_external % 2 != 1);
	}

	void discoverComponent(const CDT& ct,
		typename CDT::Face_handle start, int index,
		typename std::list<typename CDT::Edge>& border)
	{
		if (start->info().is_external != -1)
			return;
		typename std::list<typename CDT::Face_handle> queue;
		queue.push_back(start);
		while (!queue.empty())
		{
			typename CDT::Face_handle fh = queue.front();
			queue.pop_front();
			if (fh->info().is_external == -1)
			{
				fh->info().is_external = index;
				fh->info().is_external = (index % 2 == 1) ? 1 : 0;
				for (int i = 0; i < 3; i++)
				{
					typename CDT::Edge e(fh, i);
					typename CDT::Face_handle n = fh->neighbor(i);
					if (n->info().is_external == -1)
					{
						if (ct.is_constrained(e))
						{
							border.push_back(e);
						}
						else
						{
							queue.push_back(n);
						}
					}
				}
			}
		}
	}

	void operator()(HDS& hds) {
		CGAL::HalfedgeDS_decorator<HDS> decorator(hds);
		typedef typename HDS::Halfedge Halfedge;

		// 存储多面体所有的facets
		// 在modifier的过程中，facets list 会被修改，所以必须先存储到vector里
		std::vector<Facet_handle> facets;
		facets.reserve(hds.size_of_faces());
		for (Facet_iterator
			fit = hds.faces_begin(),
			end = hds.faces_end();
			fit != end; ++fit) {
			facets.push_back(fit);
		}

		hds.normalize_border();
		if (hds.border_halfedges_begin() == hds.halfedges_end())
		{
			//printf("No need to fill hole\n");
			return;
		}

		// Clear status bit
		for (auto eit = hds.halfedges_begin(); eit != hds.halfedges_end(); ++eit)
			eit->set_mask(0);

		// Hole list (ordered by halfedges)
		typename std::vector<Halfedge_iterator> halfedges;
		int noc = 0;
		for (Halfedge_iterator it = hds.border_halfedges_begin();;)
		{
			it++;
			if (it == hds.halfedges_end()) { break; }

			Halfedge_handle h = (it);
			if (h->mask() == 0)
			{
				Halfedge_handle g = h;
				halfedges.push_back(h);
				noc++;
				do
				{
					h->set_mask(noc);
					h = h->next();
				} while (h != g);
			}
			it++;
			if (it == hds.halfedges_end()) { break; }
		}

		P_traits cdt_traits(normal);
		CDT cdt(cdt_traits);

		// 			int vert_num = 0 ;
		// 			for(typename std::vector<Halfedge_iterator>::iterator
		// 				eit = halfedges.begin(); eit != halfedges.end(); eit++)
		// 			{
		// 				Halfedge_iterator h , g;
		// 				h = g = *eit;
		// 				do 
		// 				{
		// 					++ vert_num;
		// 					h = h->next();
		// 				} while ( h!=g );
		// 			}
		// 			printf("%d\n",vert_num);

		for (typename std::vector<Halfedge_iterator>::iterator
			eit = halfedges.begin(); eit != halfedges.end(); eit++)
		{
			//printf("in\n");
			Halfedge_iterator h = *eit;
			Halfedge_iterator g = h;
			typename CDT::Vertex_handle previous = 0, first = 0;
			do
			{
				typename CDT::Vertex_handle vh = cdt.insert(h->vertex()->point());
				if (first == 0) {
					first = vh;
				}
				vh->info() = h;
				if (previous != 0 && previous != vh) {
					cdt.insert_constraint(previous, vh);
				}
				previous = vh;
				h = h->next();
			} while (h != g);
			cdt.insert_constraint(previous, first);
		}

		// 			std::queue<typename CDT::Face_handle> face_queue;
		// 			face_queue.push(cdt.infinite_vertex()->face());
		// 			while(! face_queue.empty() ) {
		// 				typename CDT::Face_handle fh = face_queue.front();
		// 				face_queue.pop();
		// 				if(fh->info().is_external) continue;
		// 				fh->info().is_external = true;
		// 				for(int i = 0; i <3; ++i) {
		// 					if(!cdt.is_constrained(std::make_pair(fh, i)))
		// 					{
		// 						face_queue.push(fh->neighbor(i));
		// 					}
		// 				}
		// 			}
		// 			typename std::cout << cdt.number_of_vertices() << typename std::endl;
		for (typename CDT::All_faces_iterator
			fit = cdt.all_faces_begin(); fit != cdt.all_faces_end();
			fit++)
		{
			fit->info().is_external = -1;
		}

		int index = 0;
		typename std::list<typename CDT::Edge> border;
		typename CDT::Face_handle start = cdt.infinite_face();
		discoverComponent(cdt, cdt.infinite_face(), index++, border);
		while (!border.empty()) {
			typename CDT::Edge e = border.front();
			border.pop_front();
			typename CDT::Face_handle n = e.first->neighbor(e.second);
			if (n->info().is_external == -1) {
				discoverComponent(cdt, n, e.first->info().is_external + 1, border);
			}
		}

		/*
		// test module
		int external = 0;
		for (typename CDT::Finite_faces_iterator
			fit = cdt.finite_faces_begin(),
			end = cdt.finite_faces_end();
			fit != end; ++fit)
		{
			if (is_external(fit))
			{
				++external;
			}
		}
		printf("External faces = %d\n", external);
		// test end
		*/

		// 开始对多面体进行三角化( 循环到的这个facet )
		for (typename CDT::Finite_edges_iterator
			eit = cdt.finite_edges_begin(),
			end = cdt.finite_edges_end();
			eit != end; ++eit)
		{
			typename CDT::Face_handle fh = eit->first;
			const int index = eit->second;
			typename CDT::Face_handle opposite_fh = fh->neighbor(eit->second);
			const int opposite_index = opposite_fh->index(fh);
			const typename CDT::Vertex_handle va = fh->vertex(cdt.cw(index));
			const typename CDT::Vertex_handle vb = fh->vertex(cdt.ccw(index));

			if (!(is_external(fh) && is_external(opposite_fh)) &&
				!cdt.is_constrained(*eit))
			{
				// strictly internal edge
				Halfedge_handle h = hds.edges_push_back(Halfedge(),
					Halfedge());
				fh->info().e[index] = h;
				opposite_fh->info().e[opposite_index] = h->opposite();

				decorator.set_vertex(h, va->info()->vertex());
				decorator.set_vertex(h->opposite(), vb->info()->vertex());
			}
			if (cdt.is_constrained(*eit))
			{
				if (!is_external(fh)) {
					fh->info().e[index] = va->info();
				}
				if (!is_external(opposite_fh)) {
					opposite_fh->info().e[opposite_index] = vb->info();
				}
			}
		}

		//std::cout << "Before " << cdt.number_of_vertices() << std::endl;
		//CGAL::refine_Delaunay_mesh_2(cdt, Critria(0.0, 0.2, cdt_traits));
		//std::cout << "After " << cdt.number_of_vertices() << std::endl;
		for (typename CDT::Finite_faces_iterator
			fit = cdt.finite_faces_begin(),
			end = cdt.finite_faces_end();
			fit != end; ++fit)
		{
			if (!is_external(fit))
			{
				Halfedge_handle h0 = fit->info().e[0];
				Halfedge_handle h1 = fit->info().e[1];
				Halfedge_handle h2 = fit->info().e[2];
				CGAL_assertion(h0 != Halfedge_handle());
				CGAL_assertion(h1 != Halfedge_handle());
				CGAL_assertion(h2 != Halfedge_handle());

				typedef typename Halfedge::Base HBase;
				h0->HBase::set_next(h1);
				decorator.set_prev(h1, h0);
				h1->HBase::set_next(h2);
				decorator.set_prev(h2, h1);
				h2->HBase::set_next(h0);
				decorator.set_prev(h0, h2);

				decorator.fill_hole(h0);
				list_faces_.push_back(h0->facet());
			}
		}
	}
}; // end class Triangulate_modifier

void FillHoleCDT::fill_hole(Polyhedron& poly, Vector3& nr, const double density) {
	std::vector<Facet_iterator> list_faces;
	Triangulate_modifier<Polyhedron, K> modifier(list_faces, nr);
	poly.delegate(modifier);
	CGAL::Polygon_mesh_processing::refine(poly, list_faces,
		CGAL::Emptyset_iterator(),
		CGAL::Emptyset_iterator(),
		CGAL::Polygon_mesh_processing::parameters::density_control_factor(density));
}