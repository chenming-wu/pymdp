#pragma once

#include "CustomisedPolyhedron.h"
#include "FillHoleCDT.h"

class PlaneCutter
{
public:
	PlaneCutter() = default;
	~PlaneCutter() = default;

	// Approach 1: this would modify original polyhedron
	bool cut(Polyhedron& poly_left, Polyhedron& poly_right, const Plane& pl);

	template<typename FH>
	bool cut_and_fill(Polyhedron& poly_left, Polyhedron& poly_right, const Plane& pl) {
		bool res = cut(poly_left, poly_right, pl);
		if (!res) return res;
		Vector3 planeDir(pl.a(), pl.b(), pl.c());
		FH fh;
		fh.fill_hole(poly_left, planeDir);
		//fh.fill_hole(poly_right, -planeDir);
		return res;
	}

	// Approach 2: this won't affect original polyhedron
	std::pair<Polyhedron, Polyhedron> cut(const Polyhedron& poly, const Plane& pl);

	template<typename FH>
	std::pair<Polyhedron, Polyhedron> cut_and_fill(const Polyhedron& poly, const Plane& pl) {
		Polyhedron o1 = poly, o2;
		cut(o1, o2, pl);
		Vector3 planeDir(pl.a(), pl.b(), pl.c());
		FH fh;
		fh.fill_hole(o1, -planeDir);
		fh.fill_hole(o2, planeDir);
		return std::make_pair(o1, o2);
	}

};
