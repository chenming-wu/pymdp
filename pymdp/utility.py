import RoboFDM
from multiprocessing import Process, Manager,TimeoutError

def apply_cut(poly, plane, return_dict):
    try:
        ra = RoboFDM.init()
        ra.reset("bunny.off")
        ra.set_poly(poly)
        #print('--> Actual plane: ', plane)
        #print('--> Actual evaluation: ', ra.step(plane))
        ra.plane_cut(plane)
        poly=ra.get_poly()
        return_dict[0] = poly
    except Exception:
        pass

def apply_cut_both(poly, plane, return_dict):
    try:
        ra = RoboFDM.init()
        ra.reset("bunny.off")
        ra.set_poly(poly)
        #print('--> Actual plane: ', plane)
        #print('--> Actual evaluation: ', ra.step(plane))
        ra.plane_cut_both(plane)
        poly=ra.get_poly()
        poly_pos = ra.get_positive_poly()
        return_dict[0] = poly
        return_dict[1] = poly_pos
    except Exception:
        pass

def run_cut_process(poly, plane, export=False):
    manager = Manager()
    return_dict = manager.dict()
    if export == False:
        t = Process(target=apply_cut,args=(poly, plane, return_dict))
    else:
        t = Process(target=apply_cut_both, args=(poly, plane, return_dict))

    t.start()
    t.join(timeout=5.0)
    ret = return_dict.values()
    t.terminate()
    if len(ret) == 0:
        return None
    else:
        if export:
            return (ret[0], ret[1])
        else:
            return ret[0]

def write_mesh(mesh_str, filename):
    with open(filename, "w") as f:
        f.write(mesh_str)

def sample_poly(poly, outfile):
    ra = RoboFDM.init()
    result = ra.sample_mesh(poly, outfile)
