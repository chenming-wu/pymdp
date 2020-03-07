#include "MeshCutEval.h"

BSPNode MeshCutEval::exact_cutting_volume_and_area(const Polyhedron& poly_, const Plane& pl,
	const MapEdgePoint& map_edge_point,
	std::vector<BSPNode>& vec_bsps, std::vector<bool>& inRev)
{
	BSPNode VA;
	VA.initialize();

	const Point3& pop = pl.point();
	Vector3 cut(pl.a(), pl.b(), pl.c());
	double C = pl.d();
	cut = cut / CGAL::sqrt(cut.squared_length()); // Compute unit vector
	C = C / CGAL::sqrt(cut.squared_length());

	std::vector<Point3> cpnts3;
	std::vector<std::vector<Point2>> cpnts(vec_bsps.size());

	// Evaluate area
	const auto rotMat = Geotools::make_rotation(Eigen::Vector3d(cut.x(), cut.y(), cut.z()),
		Eigen::Vector3d(0, 1, 0));
	auto sumArea = 0.0;
	int nIntersect = 0;

	for (const auto fit : faces(poly_)) {
		Halfedge_const_handle he = fit->halfedge();
		std::pair<Halfedge_const_handle, bool> p[3];
		p[0].first = he->prev();
		p[1].first = he;
		p[2].first = he->next();
		p[0].second = Geotools::negative(cut, C, p[0].first->vertex()->point());
		p[1].second = Geotools::negative(cut, C, p[1].first->vertex()->point());
		p[2].second = Geotools::negative(cut, C, p[2].first->vertex()->point());

		double triArea = fit->area();
		double triVol = 0;

		// from small to great
		std::sort(std::begin(p), std::end(p),
			[](const std::pair<Halfedge_const_handle, bool>& a, const std::pair<Halfedge_const_handle, bool>& b) {
			return a.second < b.second;
		});

		int sum = 0;
		for (auto it = std::begin(p); it != std::end(p); ++it) {
			sum += it->second;
		}

		if (sum == 0) {
			triVol = Geotools::face_point_vol(fit, pop);
		}
		else if (sum == 1) // two points are at positive side
		{
			const Halfedge_const_handle e1 = p[2].first, e2 = e1->next();
			const auto& e1_in_map = map_edge_point.find(e1);
			const auto& e2_in_map = map_edge_point.find(e2);
			if (e1_in_map != map_edge_point.end() && e2_in_map != map_edge_point.end()) {
				const double sArea = CGAL::sqrt(CGAL::squared_area(e1_in_map->second,
					p[2].first->vertex()->point(),
					e2_in_map->second));
				triVol = (Geotools::face_point_vol(fit, pop) - Geotools::face_point_vol(
					e1_in_map->second, p[2].first->vertex()->point(), e2_in_map->second, pop));
				triArea = triArea - sArea;
			}
			else {
				std::cout << "sum == 1 error" << std::endl;
				const Point3& p0 = p[0].first->vertex()->point();
				const Point3& p1 = p[1].first->vertex()->point();
				const double dis0 = CGAL::abs(cut * Vector3(p0.x(), p0.y(), p0.z()) + C);
				const double dis1 = CGAL::abs(cut * Vector3(p1.x(), p1.y(), p1.z()) + C);
				int nOp = 0;
				if (dis0 < 1e-10) ++nOp;
				if (dis1 < 1e-10) ++nOp;
				if (2 == nOp) triVol = Geotools::face_point_vol(fit, pop);
				else triArea = 0;
			}
		}
		else if (sum == 2) // one point is at positive side
		{
			const Halfedge_const_handle e1 = p[0].first, e2 = e1->next();
			const auto& e1_in_map = map_edge_point.find(e1);
			const auto& e2_in_map = map_edge_point.find(e2);
			if (map_edge_point.find(e1) != map_edge_point.end() && map_edge_point.find(e2) != map_edge_point.end()) {
				triArea = CGAL::sqrt(CGAL::squared_area(e1_in_map->second,
					p[0].first->vertex()->point(),
					e2_in_map->second));
				triVol = Geotools::face_point_vol(e1_in_map->second,
					p[0].first->vertex()->point(),
					e2_in_map->second,
					pop);
			}
			else {
				std::cout << "sum == 2 error" << std::endl;
				const Point3& p1 = p[1].first->vertex()->point();
				const Point3& p2 = p[2].first->vertex()->point();
				const double dis1 = CGAL::abs(cut * Vector3(p1.x(), p1.y(), p1.z()) + C);
				const double dis2 = CGAL::abs(cut * Vector3(p2.x(), p2.y(), p2.z()) + C);
				int nOp = 0;
				if (dis1 < 1e-10) ++nOp;
				if (dis2 < 1e-10) ++nOp;
				if (2 == nOp) triVol = Geotools::face_point_vol(fit, pop);
				else triArea = 0;
			}
		}

		// Computation
		if (sum != 3) {
			const bool isSupported = is_supported(fit->facetNormal, cut);

			if (!isSupported) VA.support_free = false;

			// insert rotated centroid points of faces to cpnts(2D) and cpnts3(3D)
			auto& tmp_p = fit->facetCentroid;
			Eigen::Vector3d tmp_pe(tmp_p.x(), tmp_p.y(), tmp_p.z());
			auto rot_p = rotMat * tmp_pe;
			//cpnts[fit->id() - 2].emplace_back(rot_p.x(), rot_p.z());
			cpnts3.emplace_back(rot_p.x(), rot_p.y(), rot_p.z());
			//std::cout << rot_p.transpose() << std::endl;

			// update volume
			VA.first += triVol;
			//vec_bsps[fit->id() - 2].first += triVol;

			// update area
			if (fit->supported() && !isSupported) // was safe before but now is risky
			{
				VA.second += triArea;
			}

			if (!fit->supported() && isSupported) // was risky before but now is safe
			{
				//vec_bsps[fit->id() - 2].second -= projArea;
				VA.second -= triArea;

				inRev[fit->patch_id()] = true;
			}

			if (!fit->supported() && !isSupported)    // was risky before and now is also risky
			{
				//vec_bsps[fit->id() - 2].second -= (fit->projArea() - projArea);
				//VA.second -= (fit->projArea() - projArea);
				//VA.second -= 0;
				// Absolute decrease
				//VA.fourth += (fit->projArea() - projArea);
			}

			if (!isSupported) VA.fifth += triArea;

			sumArea += triArea;
		}

		++nIntersect;
	}

	// compute the volume of bounding box
	if (nIntersect <= 3) {
		VA.first = 0.;
		VA.second = 0.;
	}

	//VA.third /= boxVol_;
	VA.first = std::abs(VA.first);
	//VA.fifth /= sumArea;
	//VA.fifth = 1.0 - VA.fifth;
	return VA;
}

void MeshCutEval::get_inrev_risky_area(const Polyhedron& poly, const Vector3& current_dir, const Bbox& box, const std::vector<Point3>& plt, std::vector<bool>& inRev)
{
    std::vector<BSP_Result> vec_tu;

    Plane min_plane;
    const double a = current_dir.x(), b = current_dir.y(), c = current_dir.z();
    double dmin = compute_min_d(current_dir, box);
    double dmax = compute_max_d(current_dir, box);
    double cdmax = -DBL_MAX;

    for (auto &p : plt) {
        Vector3 vp(p.x(), p.y(), p.z());
        const double tmp = vp * current_dir;
        if (tmp > cdmax) cdmax = tmp;
    }
    const int n_int = static_cast<int>(dmax - dmin + 1);

    // Find possible fragile points
    std::vector<Vertex_const_handle> vecPosPnts, vecNegPnts;
    for (auto vit = poly.vertices_begin(); vit != poly.vertices_end(); ++vit) {
        const double faceDot = vit->vertexNormal * Vector3(a, b, c);
        if (faceDot > 0.92) vecPosPnts.emplace_back(vit);
        else if (faceDot < -0.92) vecNegPnts.emplace_back(vit);
    }

    Debug("dmin = " << dmin << " dmax = " << dmax << std::endl);
    Debug("cdmax " << cdmax << std::endl);
    Debug("interval = " << n_int << std::endl);

    for (auto j = 0; j < n_int; ++j) {
        double d = dmin + static_cast<double>(j);
        if (d < cdmax) continue;
        Plane plane(a, b, c, -d);

		bool isFragile = false;
		double minFragile = DBL_MAX;
		for (auto v : vecPosPnts)
		{
			auto& pnt = v->point();
			auto  dis = pnt.x() * a + pnt.y() * b + pnt.z() * c - d;
			if (dis > 0 && dis < 1)
			{
				isFragile = true;
				break;
			}
			if (dis > 0 && dis < minFragile) minFragile = dis;
		}

		if (isFragile) continue;

		for (auto v : vecNegPnts) {
            auto &pnt = v->point();
            auto dis = pnt.x() * a + pnt.y() * b + pnt.z() * c - d;
            if (dis < 0 && dis > -1) {
                isFragile = true;
                break;
            }
            if (dis < 0 && -dis < minFragile) minFragile = -dis;
        }

        if (isFragile) continue;

        std::unordered_map<Halfedge_const_handle, Point3> mapEdgePoint;
        std::set<Facet_const_iterator> intersectedFaces;
        if (!cut_with_plane(poly, plane, mapEdgePoint, intersectedFaces)) continue;

        // Label intersected faces (euler update)
        Plane plane_neg(plane.a(), plane.b(), plane.c(), plane.d());
        auto nbc = euler_update_facets(poly, plane_neg, intersectedFaces);

        // if it has multiple connected component after cutting, then check if they are connected with
        // the physical platform, this is done by approximating its minimal y values
        if (nbc.size() > 1) {
            auto maxMinValueY = *(std::max_element(nbc.begin(), nbc.end()));
            if (maxMinValueY > plt[0].y() + CONNECT_EPS) continue;
        }
        std::vector<BSPNode> vecBsps;
        auto nRetRes = exact_cutting_volume_and_area(poly, plane, mapEdgePoint, vecBsps, inRev);

        BSP_Result tu;
        boost::get<0>(tu) = plane;
        boost::get<1>(tu) = nRetRes.first;
        boost::get<2>(tu) = -nRetRes.second;
        boost::get<3>(tu) = d - cdmax;
        boost::get<4>(tu) = minFragile;
        boost::get<5>(tu) = nRetRes.fifth;
        boost::get<6>(tu) = nRetRes.support_free; // Support-Free
        vec_tu.push_back(tu);
    }
}