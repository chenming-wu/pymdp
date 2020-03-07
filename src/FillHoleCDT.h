#pragma once
#include "FillHole.h"

class FillHoleCDT : public FillHole {
public:
	FillHoleCDT() = default;
	~FillHoleCDT() = default;

	void fill_hole(Polyhedron& poly, Vector3& nr, const double density = 0.4);
};