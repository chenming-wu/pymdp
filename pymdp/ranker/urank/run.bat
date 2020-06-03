cd E:\RAL2020\
python "generate_dataset - 0529.py"
cd E:\RAL2020\ranker\src
python prepare_data.py
python evaluate.py --loss_fn grank --data_dir ../data/RAL-I/1 --tfrecords_filename RAL-I.tfrecords