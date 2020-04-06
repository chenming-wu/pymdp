#include "MeshCutEval.h"
#include <stack>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <CGAL/Polygon_mesh_processing/measure.h>


inline double string_to_double(std::string &str) {
    std::istringstream iss(str);
    double x;
    if (iss >> x)
        return x;
    return 0.0;
}

MeshCutEval::MeshCutEval() = default;

MeshCutEval::~MeshCutEval() = default;

bool MeshCutEval::initialization(const Polyhedron &poly_) {
    Debug(ori_mesh_file_ << std::endl << ori_simp_mesh_file_
                         << std::endl
                         << "Number of int_planes: " << intPlanes_.size() << std::endl
                         << "Number of cross sections " << intCrossSection_.size() << std::endl);
    Debug(poly_.size_of_facets() << std::endl);

    // Construct a bounding box
    //box_ = bbox_3(poly_.points_begin(), poly_.points_end());
    //boxVol_ = (box_.xmax() - box_.xmin()) * (box_.ymax() - box_.ymin()) * (box_.zmax() - box_.zmin());
    Debug(box_ << std::endl);
    Debug("Vol: " << boxVol_ << std::endl);
   

    return true;
}

std::vector<Point3> MeshCutEval::get_platform_cross(double radius, double platformZ) {
    std::vector<Point3> intCross;
    auto interRad = 2.0 * M_PI / 60.0;
    for (int i = 0; i < 60; ++i) {
        auto x = radius * std::cos(i * interRad);
        auto z = radius * std::sin(i * interRad);
        auto y = platformZ;
        intCross.emplace_back(x, y, z);
    }
    return intCross;
}

Plane MeshCutEval::convert_xyzab_to_plane(double x, double y, double z, double alpha, double beta, Bbox& box_)
{
	const auto ax = x * (box_.xmax() - box_.xmin());
	const auto ay = y * (box_.ymax() - box_.ymin());
	const auto az = z * (box_.zmax() - box_.zmin());
	const auto alphaRad = alpha * 2.0 * M_PI, betaRad = beta * 2.0 * M_PI;
	const auto a = std::cos(alphaRad) * std::cos(betaRad);
	const auto b = std::sin(alphaRad) * std::cos(betaRad);
	const auto c = std::sin(betaRad);
	const auto d = a * ax + b * ay + c * az;
    return {a, b, c, d};
}

Plane MeshCutEval::convert_abg_to_plane(double alpha, double beta, double gamma, const MinSphere& sphere) {
    const auto alphaRad = alpha * 2.0 * M_PI, betaRad = beta * M_PI /2.0;
    const auto a = std::cos(alphaRad) * std::cos(betaRad);
    const auto c = std::sin(alphaRad) * std::cos(betaRad);
    const auto b = std::sin(betaRad);
    const auto d = 2.0 * (gamma - 0.5) * CGAL::sqrt(sphere.squared_radius()) ;
    const Vector3 planeDir(a, b, c);
    const Point3 crossPnt = sphere.center() + d * planeDir;
    return {crossPnt, planeDir};
}


bool MeshCutEval::cut_with_plane(const Polyhedron& poly_, const Plane& plane, MapEdgePoint& mapEdgePoint,
    std::set<Facet_const_iterator>& intFaceSet) {
    Vector3 cut(plane.a(), plane.b(), plane.c());
    double C = plane.d();
    cut = cut / CGAL::sqrt(cut.squared_length()); // Normal
    C = C / CGAL::sqrt(cut.squared_length());

    int n_cut = 0;
    // TODO: Test if ``edges'' is equivalent to ``edge_begin()''
    for (auto it : edges(poly_))
    {
        const auto h = it.halfedge();
        Vector3 p1(h->prev()->vertex()->point().x(), h->prev()->vertex()->point().y(),
                   h->prev()->vertex()->point().z());
        Vector3 p2(h->vertex()->point().x(), h->vertex()->point().y(), h->vertex()->point().z());
        const double d0 = cut * p1 + C;
        double d1 = cut * p2 + C;
        Vector3 Q;
        if ((d0 >= 0 && d1 < 0) || (d0 < 0 && d1 >= 0)) {
            // new position
            Q = p1 + ((d0 / (d0 - d1)) * (p2 - p1));

            mapEdgePoint[h] = Point3(Q.x(), Q.y(), Q.z());
            mapEdgePoint[h->opposite()] = mapEdgePoint[h];
            ++n_cut;

            // label facet with id
            // h->facet()->id() = 1;
            //h->opposite()->facet()->id() = 1;

            // TODO: insert intersected faces to the set container
            intFaceSet.insert(h->facet());
            intFaceSet.insert(h->opposite()->facet());
        }
    }

    return n_cut > 3;
}

/**
 * \brief Euler update faces on the surface mesh
 * \param pl 
 * \return number of components, and its minimal y value
 */
std::vector<double> MeshCutEval::euler_update_facets(const Polyhedron &poly_, const Plane &pl,
                                                     const std::set<Facet_const_iterator> &intFaceSet) {
    std::stack<Facet_const_iterator> trilist;
    std::unordered_set<Facet_const_iterator> already_processed;
    while (!trilist.empty()) { trilist.pop(); }
    int noc = 1;
    Facet_const_handle t, s;
    std::vector<double> vecMinY;

    for (auto fit : faces(poly_)) {
        if (intFaceSet.find(fit) != intFaceSet.end()) {
            if (already_processed.find(fit) != already_processed.end())
                continue;
            ++noc;
            trilist.push(fit);
            double tempMinY = DBL_MAX;
            while (!trilist.empty()) {
                t = trilist.top();
                trilist.pop();
                if (!already_processed.insert(t).second) continue;
                Halfedge_const_iterator he = t->halfedge(), hd = he;
                do
                {
                    auto neighbor = hd->opposite()->face();
                    if ( neighbor != boost::graph_traits<Polyhedron>::null_face() ) {
                        if ( (!Geotools::has_positive_vertex(neighbor, pl) || intFaceSet.find(neighbor) != intFaceSet.end())
                            && already_processed.find(neighbor) == already_processed.end()) {
                            trilist.push(neighbor);
                            auto tempY = Geotools::get_min_y(neighbor);
                            if (tempY < tempMinY) tempMinY = tempY;
                        }
                    }
					hd = hd->next();
                } while (hd != he);
            }
            vecMinY.push_back(tempMinY);
        }
    }
    return vecMinY;
}

/**
 * \brief Get the decrease value of the risky faces
 * \param pl 
 * \param map_edge_point
 * \param vec_bsps
 * \return 
 */
BSPNode MeshCutEval::exact_cutting_volume_and_area(const Polyhedron& poly_,
	const Plane& pl,
	const MapEdgePoint& map_edge_point,
	std::vector<BSPNode>& vec_bsps) {
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
                  [](const std::pair<Halfedge_const_handle, bool> &a, const std::pair<Halfedge_const_handle, bool> &b) {
                      return a.second < b.second;
                  });

        int sum = 0;
        for (auto it = std::begin(p); it != std::end(p); ++it) {
            sum += it->second;
        }

        if (sum == 0) {
            triVol = Geotools::face_point_vol(fit, pop);
        } else if (sum == 1) // two points are at positive side
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
            } else {
                const Point3 &p0 = p[0].first->vertex()->point();
                const Point3 &p1 = p[1].first->vertex()->point();
                const double dis0 = CGAL::abs(cut * Vector3(p0.x(), p0.y(), p0.z()) + C);
                const double dis1 = CGAL::abs(cut * Vector3(p1.x(), p1.y(), p1.z()) + C);
                int nOp = 0;
                if (dis0 < 1e-10) ++nOp;
                if (dis1 < 1e-10) ++nOp;
                if (2 == nOp) triVol = Geotools::face_point_vol(fit, pop);
                else triArea = 0;
            }
        } else if (sum == 2) // one point is at positive side
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
            } else {
                const Point3 &p1 = p[1].first->vertex()->point();
                const Point3 &p2 = p[2].first->vertex()->point();
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
            auto &tmp_p = fit->facetCentroid;
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

double MeshCutEval::compute_min_d(const Vector3 &dir, const Bbox& box_) {
	auto d = DBL_MAX; 
    std::vector<Vector3> bottomLayer_;
    bottomLayer_.reserve(4);
    bottomLayer_.emplace_back(box_.xmin(), box_.ymin(), box_.zmin());
	bottomLayer_.emplace_back(box_.xmin(), box_.ymin(), box_.zmax());
	bottomLayer_.emplace_back(box_.xmax(), box_.ymin(), box_.zmax());
	bottomLayer_.emplace_back(box_.xmax(), box_.ymin(), box_.zmin());
    
    for (auto &p : bottomLayer_) {
        const double tmp = dir * p;
        if (tmp < d) d = tmp;
    }
    return d;
}

double MeshCutEval::compute_max_d(const Vector3 &dir, const Bbox& box_) {
	double d = -DBL_MAX;
	std::vector<Vector3> topLayer_;
    topLayer_.reserve(4);
	topLayer_.emplace_back(box_.xmin(), box_.ymax(), box_.zmin());
	topLayer_.emplace_back(box_.xmin(), box_.ymax(), box_.zmax());
	topLayer_.emplace_back(box_.xmax(), box_.ymax(), box_.zmax());
	topLayer_.emplace_back(box_.xmax(), box_.ymax(), box_.zmin());

    for (auto &p : topLayer_) {
        const double tmp = dir * p;
        if (tmp > d) d = tmp;
    }
    return d;
}

/**
 * \brief This function is used to verify whether a face needs support structure or not
 * under the constraint of maximal supporting angle.
 * \param f_normal Face normal.
 * \param dir Printing direction.
 * \return true or false
 */
bool MeshCutEval::is_supported(Vector3 &f_normal, const Vector3 &dir) {
    return !(f_normal * dir + alpha_max_ < 0);
}

/**
 * \brief This function will initialize the normals, areas and whether needs support of a face
 * \param
 */
std::vector<double> MeshCutEval::initialize_supports(Polyhedron& poly_) {
    std::vector<double> init_results;
    double sumRiskyArea = 0.0, sumVolume = 0.0, sumArea = 0.0;
    sumVolume = CGAL::Polygon_mesh_processing::volume(poly_);

    // TODO: make this function static
    Vector3 cut(0, 1, 0);

    Debug("base plane eps " << base_plane_eps << std::endl);

    // Initialize normals, supported and area
    for (auto fit = poly_.facets_begin(); fit != poly_.facets_end(); ++fit) 
    {
        fit->facetCentroid = Geotools::get_facet_centroid(fit);
        fit->facetNormal = CGAL::Polygon_mesh_processing::compute_face_normal(fit, poly_);
        fit->supported() = is_supported(fit->facetNormal, cut);

        // Compute the area of face
        Halfedge_handle he = fit->halfedge();
        const auto a = CGAL::sqrt(CGAL::squared_area(he->prev()->vertex()->point(),
                                                    he->vertex()->point(),
                                                    he->next()->vertex()->point()));
        fit->set_face_area(a);
    }

    for (auto vit = poly_.vertices_begin(); vit != poly_.vertices_end(); ++vit) {
        vit->vertexNormal = CGAL::Polygon_mesh_processing::compute_vertex_normal(vit, poly_);
    }

    // For sup volume and area
    for (auto &fit : faces(poly_)) {
        // Step 0: Compute sum area
        sumArea += fit->area();

        // Step1: Virtual supported Case I: Lower than the base plane
        Point3 cp(fit->facetCentroid.x(), fit->facetCentroid.y(), fit->facetCentroid.z());

        // TODO: how to handle virtual supported
        //        if (base_plane_eps.has_on_negative_side(cp)) {
        //            fit->virtual_supported() = true;
        //            continue;
        //        }

        // Step 2: Compute risky area and volume
        if (!fit->supported()) {
            sumRiskyArea += fit->area();
            Debug(fit->area() << " " << fit->proj_area() << std::endl);
        }
    }

    //std::cout << "Sum area = " << sumArea  << " or " << CGAL::Polygon_mesh_processing::area(poly_) << std::endl;
    //std::cout <<"Sum risky area = " << sumRiskyArea << std::endl;

    init_results.push_back(sumVolume);
    init_results.push_back(sumArea);
    init_results.push_back(sumRiskyArea);
    return init_results;
}

// This function is currently used in greedy random search
BSPNode MeshCutEval::apply_plane_cut(const Polyhedron& poly_, const Plane& pl, const Bbox& box, const std::vector<Point3>& plt) {
    BSPNode retRes;
    retRes.done = true;
    Plane min_plane;
    //	candidate_dirs_.clear();
    //	candidate_dirs_.emplace_back(0, 1, 0);

    // Pass 1: Test if there is an intersection with the platform
    const auto a = pl.a();
    const auto b = pl.b();
    const auto c = pl.c();
    const auto d = pl.d();

    Vector3 currentDir(a, b, c);
    //auto minD = compute_min_d(currentDir, box);
    //auto maxD = compute_max_d(currentDir, box);
    auto cdmax = -DBL_MAX, cdmin = DBL_MAX;

    for (const auto& p : plt) {
        Vector3 vp(p.x(), p.y(), p.z());
        const double tmp = vp * currentDir;
        if (tmp > cdmax) cdmax = tmp;
        if (tmp < cdmin) cdmin = tmp;
    }
    //retRes.third = std::max(0.0, d-cdmax);
    const auto neg_d = -d;
    if (neg_d > cdmin&& neg_d < cdmax) {
        retRes.third = 0.0;
        return retRes;
    }
    else
    {
        retRes.third = neg_d - cdmax;
    }
    //if (d < cdmax) return retRes;

    // Pass 2: Test if fragile is detected
    //std::cout << "Pass 2: Test if fragile is detected" << std::endl;
    std::vector<Polyhedron::Vertex_const_iterator> pos_pnts, neg_pnts;
    for (auto vit = poly_.vertices_begin(); vit != poly_.vertices_end(); ++vit) {
        const double faceDot = vit->vertexNormal * Vector3(a, b, c);
        if (faceDot > FRAGILE_EPS) pos_pnts.emplace_back(vit);
        else if (faceDot < -FRAGILE_EPS) neg_pnts.emplace_back(vit);
    }

    double minFragile = DBL_MAX;
    for (auto v : pos_pnts) {
        auto& pnt = v->point();
        auto dis = pnt.x() * a + pnt.y() * b + pnt.z() * c + d;

        if (dis > 0 && dis < minFragile)
        {
            minFragile = dis;
        }
    }

    for (auto v : neg_pnts) {
        auto& pnt = v->point();
        auto dis = pnt.x() * a + pnt.y() * b + pnt.z() * c + d;
        if (dis < 0 && -dis < minFragile)
        {
            minFragile = -dis;
        }
    }
    retRes.fourth = minFragile;

    // Pass 3: feed it into evaluation and return a reward
    //std::cout << "Pass 3: feed it into evaluation and return a reward" << std::endl;
    Plane plane(a, b, c, d);
    //std::cout << plane << std::endl;

    std::unordered_map<Halfedge_const_handle, Point3> mapEdgePoint;
    std::set<Facet_const_iterator> intersectedFaces;
    if (!cut_with_plane(poly_, plane, mapEdgePoint, intersectedFaces)) {
        return retRes;
    }

    Plane plane_neg(plane.a(), plane.b(), plane.c(), plane.d());
    auto nbc = euler_update_facets(poly_, plane_neg, intersectedFaces);

    // if it has multiple connected component after cutting, then check if they are connected with
    // the physical platform, this is done by approximating its minimal y values
    if (nbc.size() > 1) {
        auto maxMinValueY = *(std::max_element(nbc.begin(), nbc.end()));
        if (maxMinValueY > plt[0].y() + CONNECT_EPS) return retRes;
    }

    std::vector<BSPNode> vecBsps;
    auto nRetRes = exact_cutting_volume_and_area(poly_, plane, mapEdgePoint, vecBsps);

    retRes.first = nRetRes.first;
    retRes.second = nRetRes.second;
    retRes.fifth = nRetRes.fifth;

    Debug("Volume = " << va.first << "; R_face = " << va.second << "; R_vol = " << va.third << std::endl;);

    /* //-----------------------------------------------//
    // Objective function
    const double a1 = 0.8;
    const double a2 = 1 - a1;
    boost::get<1>(tu) = a1 * va.second / intArea_ + a2 * va.third / intVol_;
    //-----------------------------------------------// */

    // consider: decrease of volume, decrease of risky area, thickness, distance to the platform


    /*
    va.first;    // Approximate segmented volume
    va.second;    // Decreased risky area (incl. both 1. from risky to safe (-) and 2. from safe to risky (+)
    va.third;    // Bounding volume (not use)
    va.fourth;    // Only counts decreased risky area
    va.fifth;    // Risky area (residual)
    va.support_free; // Support-Free
    // One should note that (second = fourth - fifth)
    */
    //std::cout << va.first << " " << va.second << " " << va.third <<  " " << va.fourth << " " << va.fifth << " " << va.support_free << std::endl;

    //std::cout << va.second << " = " << va.fourth << " - " << va.fifth << " = (" << va.fourth -va.fifth << ")" << std::endl;

    return retRes;
}

void MeshCutEval::evaluation_direction(const Polyhedron& poly, const Vector3& current_dir, const Bbox& box, const std::vector<Point3>& plt, std::vector<BSP_Result>& vec_tu)
{
    double interval = 0.1;
    Plane min_plane;
    const double a = current_dir.x(), b = current_dir.y(), c = current_dir.z();
    double dmin = compute_min_d(current_dir, box);
    double dmax = compute_max_d(current_dir, box);
    double cdmax = -DBL_MAX;

    for (auto& p : plt)
    {
        Vector3 vp(p.x(), p.y(), p.z());
        const double tmp = vp * current_dir;
        if (tmp > cdmax) cdmax = tmp;
    }
    const int n_int = static_cast<int>(dmax - dmin + 1);

    // Find possible fragile points
    std::vector<Vertex_const_handle> vecPosPnts, vecNegPnts;
    for (auto vit = poly.vertices_begin(); vit != poly.vertices_end(); ++vit)
    {
        const double faceDot = vit->vertexNormal * Vector3(a, b, c);
        if (faceDot > 0.92)	vecPosPnts.emplace_back(vit);
        else if (faceDot < -0.92) vecNegPnts.emplace_back(vit);
    }

    Debug("dmin = " << dmin << " dmax = " << dmax << std::endl);
    Debug("cdmax " << cdmax << std::endl);
    Debug("interval = " << n_int << std::endl);

    for (auto j = 0; j < n_int; ++j)
    {
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

        for (auto v : vecNegPnts)
        {
            auto& pnt = v->point();
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
        auto nRetRes = exact_cutting_volume_and_area(poly, plane, mapEdgePoint, vecBsps);

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

std::pair<double, double> MeshCutEval::find_suitable_platform(const Polyhedron& poly)
{
	typedef std::vector<K::Point_3> Polyline_type;
	typedef std::list< Polyline_type > Polylines;

    double min_y = DBL_MAX;
    for (const auto v : vertices(poly)) {
        if (v->point().y() < min_y)
        {
            min_y = v->point().y();
        }
    }

    CGAL::Polygon_mesh_slicer<Polyhedron, K> slicer(poly);

    Polylines polylines;
    for (auto i = 0; i < 10; ++i)
    {
        slicer(K::Plane_3(0, 1, 0, - min_y - 0.1*i), std::back_inserter(polylines));
    }

    double radius = 0;

    for (const auto& poly : polylines)
    {
        for (const auto& p : poly)
        {
            Vector3 v(p.x(), 0.0, p.z());
            const double tmpD = CGAL::sqrt(v.squared_length());
            if (tmpD > radius) 
            {
                radius = tmpD;
            }
        }
    }
    return std::make_pair(radius, min_y);
}