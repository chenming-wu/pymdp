#include "PlaneCut.h"
#include <CGAL/Polygon_mesh_processing/repair.h>

constexpr double eps = 1e-8;
constexpr double eps10 = 1e-7;

inline bool negative(Vector3& p, double C, Point3& a)
{
	Vector3 probe(a.x(), a.y(), a.z());
	return probe * p + C > eps;
}

inline bool positive(Vector3& p, double C, Point3& a)
{
	Vector3 probe(a.x(), a.y(), a.z());
	return probe * p + C < -eps;
}

bool PlaneCutter::cut(Polyhedron& poly_left, Polyhedron& poly_right, const Plane& pl) {

	int IntrCnt = 0;
	std::vector<Edge_iterator> Edges;
	std::vector<Halfedge_handle> ModifiedEdges; //Side to be modified
	Edges.reserve(poly_left.size_of_halfedges() / 2);

	// clear degenerate edges
	for (const auto& e : edges(poly_left))
	{
		if (CGAL::Polygon_mesh_processing::is_degenerate_edge(e, poly_left))
		{
			poly_left.join_vertex(e.halfedge());
		}
	}

	for (Polyhedron::Edge_iterator it = poly_left.edges_begin();
		it != poly_left.edges_end(); ++it)
	{
		Edges.push_back(it);
	}

	Vector3 cut(pl.a(), pl.b(), pl.c());
	double C = pl.d();
	cut = cut / CGAL::sqrt(cut.squared_length());
	C = C / CGAL::sqrt(cut.squared_length());

	double d0, d1;
	for (std::vector<Polyhedron::Edge_iterator>::iterator it = Edges.begin();
		it != Edges.end(); ++it)
	{
		Halfedge_handle h = *it;
		Vector3 p1(h->prev()->vertex()->point().x(), h->prev()->vertex()->point().y(), h->prev()->vertex()->point().z());
		Vector3 p2(h->vertex()->point().x(), h->vertex()->point().y(), h->vertex()->point().z());
		d0 = cut * p1 + C;
		d1 = cut * p2 + C;
		Vector3 Q;
		if ((d0 >= 0 && d1 < 0) || (d0 < 0 && d1 >= 0))
		{
			Q = p1 + ((d0 / (d0 - d1)) * (p2 - p1));
			Point3 newPnt(Q.x(), Q.y(), Q.z());
			if (std::abs(d0) < eps10)
			{
				h->prev()->vertex()->point() = newPnt;
			}
			else if (std::abs(d1) < eps10)
			{
				h->vertex()->point() = newPnt;
			}
			else
			{
				IntrCnt++;
				Halfedge_handle t = poly_left.split_edge(h);
				t->vertex()->point() = newPnt;
				ModifiedEdges.push_back(t);
			}
		}
	}
	//std::cout << "Modified edges = " << ModifiedEdges.size() << std::endl;

	for (std::vector<Halfedge_handle>::iterator it = ModifiedEdges.begin();
		it != ModifiedEdges.end(); it++)
	{
		Halfedge_handle h = *it;
		Halfedge_handle g = h->opposite()->prev();
		Facet_handle f_h = h->facet();
		Facet_handle g_h = g->facet();
		Halfedge_handle tmp_he;
		if (f_h != nullptr && !f_h->is_triangle())
		{
			if (f_h->is_quad())
			{
				tmp_he = poly_left.split_facet(h, h->next()->next());
			}
			else
			{
				tmp_he = poly_left.split_facet(h, h->next()->next());
				poly_left.split_facet(h, h->next()->next());
			}
		}

		if (g_h != nullptr && !g_h->is_triangle())
		{
			if (g_h->is_quad())
			{
				tmp_he = poly_left.split_facet(g, g->next()->next());
			}
			else
			{
				tmp_he = poly_left.split_facet(g, g->next()->next());
				poly_left.split_facet(g, g->next()->next());
			}
		}
	}

	poly_right = poly_left;

	for (Facet_iterator it = poly_left.facets_begin(), nd = poly_left.facets_end();
		it != nd;)
	{
		Facet_iterator itNext = it;
		++itNext;
		Halfedge_handle h = it->halfedge();
		if (h == NULL)
		{
			it = itNext;
			continue;
		}
		Halfedge_handle e = h;
		do
		{
			if (negative(cut, C, h->vertex()->point()))
			{
				poly_left.erase_facet(e);
				break;
			}
			h = h->next();
		} while (h != e);
		it = itNext;
	}

	for (Facet_iterator it = poly_right.facets_begin(), nd = poly_right.facets_end();
		it != nd;)
	{
		Facet_iterator itNext = it;
		++itNext;
		Halfedge_handle h = it->halfedge();
		if (h == NULL)
		{
			it = itNext;
			continue;
		}
		Halfedge_handle e = h;
		do
		{
			if (positive(cut, C, h->vertex()->point()))
			{
				poly_right.erase_facet(e);
				break;
			}
			h = h->next();
		} while (h != e);
		it = itNext;
	}

	return true;
}

std::pair<Polyhedron, Polyhedron> PlaneCutter::cut(const Polyhedron& poly, const Plane& pl) {
	Polyhedron o1;
	Polyhedron o2;
	o1 = poly;
	cut(o1, o2, pl);
	return std::make_pair(o1, o2);
}