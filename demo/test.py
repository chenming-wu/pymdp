from pymdp.beam_guided import BGS

def test_inv(proc):
    inrev_risky = proc.env.get_inrev_risky_area()
    print(inrev_risky)

def test_risky(proc):
    risky_area = proc.env.get_risky_area()
    print(risky_area)

def test_cut(proc):
    plane = [0.0502082,    0.71893      ,0.693267 ,  -18.21249238]
    res = proc.env.step(plane)
    print(res)

if __name__ == "__main__":
    proc = BGS(filename='449906_sf_repaired.off', export=True)
    # test_cut(proc)
    test_inv(proc)

    # test_cut(proc)
    # test_cut(proc)
    # quit(0)
    
    test_risky(proc)
    #test_inv(proc)
    quit(0)
    proc.set_beam_width(2)
    proc.set_output_folder('kitten')
    proc.start_search()
    