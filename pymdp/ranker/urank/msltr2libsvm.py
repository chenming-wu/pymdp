import os

RAW_RANK_DATA = os.environ.get('RAW_RANK_DATA')
LIGHTGBM_DATA = os.environ.get('LIGHTGBM_DATA')

def get_OHSUMED_data_path(tfrecords_folder, fold_str, file_type):
    OHSUMED_data_folder = os.path.join('OHSUMED', 'Feature-min', 'Fold{}'.format(fold_str))
    # OHSUMED
    # print('file_type', file_type)
    full_file_name = os.path.join(RAW_RANK_DATA, OHSUMED_data_folder, file_type)
    if file_type == 'train':
        full_file_name += 'ing'    
    if file_type == 'vali':
        full_file_name += 'dation'
    full_file_name += 'set'
    data_path = full_file_name + '.txt'
    return data_path

def get_data_path(tfrecords_folder, fold_str, file_type):
    data_path = ''
    # OHSUMED
    if tfrecords_folder == 'OHSUMED':
        data_path = get_OHSUMED_data_path(tfrecords_folder, fold_str, file_type)
    else:
        # MQ2007_data
        MS_data_folder = os.path.join(tfrecords_folder, 'Fold{}'.format(fold_str))
        data_path = os.path.join(RAW_RANK_DATA, MS_data_folder, file_type + ".txt")
    return data_path

def convert(write2folder, lightgbm_folder, file_type, \
	fold, out_data_filename, out_query_filename):
	group_features = {}
	group_labels = {}
	fold = str(fold)
	data_path = get_data_path(lightgbm_folder, fold, file_type)
	print('data_path', data_path)

	raw_rank_data_input = open(data_path,"r")
	output_feature = open(os.path.join(write2folder, out_data_filename), "w")
	output_query = open(os.path.join(write2folder, out_query_filename), "w")
	cur_cnt = 0
	cur_doc_cnt = 0
	last_qid = -1
	while True:
		line = raw_rank_data_input.readline()
		if not line:
			break
		line = line.split(' #')[0]
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		label = tokens[0]
		qid = int(tokens[1].split(':')[1])
		if qid != last_qid:
			if cur_doc_cnt > 0:
				output_query.write(str(cur_doc_cnt) + '\n')
				cur_cnt += 1
			cur_doc_cnt = 0
			last_qid = qid
		cur_doc_cnt += 1
		output_feature.write(label+' ')
		output_feature.write(' '.join(tokens[2:]) + '\n')
	output_query.write(str(cur_doc_cnt) + '\n')
	raw_rank_data_input.close()
	output_query.close()
	output_feature.close()


def main():
    lightgbm_folders = ['OHSUMED', 'MQ2007']# 'OHSUMED', 'MQ2007', 'MSLR-WEB10K', 'MSLR-WEB30K'
    folds = 5
    for lightgbm_folder in lightgbm_folders:
        for fold in range(1, folds + 1):
            write2folder = os.path.join(LIGHTGBM_DATA, lightgbm_folder, str(fold))
            print(write2folder)
            if not os.path.exists(write2folder):
                os.makedirs(write2folder)
            convert(write2folder, lightgbm_folder, "train", fold, "rank.train", "rank.train.query")
            convert(write2folder, lightgbm_folder, "test", fold, "rank.test", "rank.test.query")
            convert(write2folder, lightgbm_folder, "vali", fold, "rank.vali", "rank.vali.query")

if __name__ == '__main__':
	main()
	print('Done')