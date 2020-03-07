#include "FillHole.h"

#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

void FillHoleCGAL::fill_hole(Polyhedron& poly, Vector3& nr, const double density)
{
	//#define FILL_AND_REFINE
	double alpha = 0.8;
	bool use_DT = true;
	unsigned int nb_holes = 0;
	BOOST_FOREACH(Halfedge_handle h, halfedges(poly))
	{
		if (h->is_border())
		{
			std::vector<Facet_handle> patch_facets;
#ifdef FILL_AND_REFINE
			CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(poly,
				h, std::back_inserter(patch_facets),
				CGAL::Emptyset_iterator(),
				CGAL::Polygon_mesh_processing::parameters::
				density_control_factor(alpha).
				use_delaunay_triangulation(use_DT));
#else
			CGAL::Polygon_mesh_processing::triangulate_hole(poly,
				h, std::back_inserter(patch_facets),
				CGAL::Polygon_mesh_processing::parameters::use_delaunay_triangulation(use_DT));
#endif
			++nb_holes;
		}
	}
}