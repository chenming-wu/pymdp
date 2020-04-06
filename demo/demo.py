from pymdp.beam_guided import BGS

if __name__ == "__main__":
    proc = BGS(filename='kitten.off', export=True)
    proc.set_beam_width(10)
    proc.set_output_folder('kitten')
    proc.start_search()