import os

RAW_RANK_DATA = os.environ.get('RAW_RANK_DATA')
LIGHTGBM_DATA = os.environ.get('LIGHTGBM_DATA')
PREDICTION_RESULT_FILE = 'LightGBM_predict_result'

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

def run_pl(lightgbm_folder, fold):
    fold = str(fold)
    data_path = get_data_path(lightgbm_folder, fold, 'test')
    print('data_path', data_path)
    fold_result_file = '{}_{}_ndcg'.format(lightgbm_folder, fold)
    os.system('perl mslr-eval-score-mslr.pl {} {} {} 0'.format(data_path, \
        PREDICTION_RESULT_FILE, fold_result_file))
    complete_result_file = '{}_ndcg.txt'.format(lightgbm_folder)
    os.system('cat "{}" >> "{}"'.format(fold_result_file, complete_result_file))
    os.system('echo "\n" >> "{}"'.format(complete_result_file))

def main():
    lightgbm_folders = ['MSLR-WEB30K']# 'OHSUMED', 'MQ2007', 'MSLR-WEB10K', 'MSLR-WEB30K'
    folds = 5
    for lightgbm_folder in lightgbm_folders:
        complete_result_file = '{}_ndcg.txt'.format(lightgbm_folder)
        if os.path.isfile(complete_result_file):
            os.system('rm {}'.format(complete_result_file))
        for fold in range(1, folds + 1):
            input_data_folder = os.path.join(LIGHTGBM_DATA, lightgbm_folder, str(fold))
            # print(input_data_folder)
            os.system('cp template_train.conf train.conf')
            os.system('cp template_predict.conf predict.conf')
            data_path = os.path.join(input_data_folder, 'rank.train')
            valid_data_path = os.path.join(input_data_folder, 'rank.vali')
            test_data_path = os.path.join(input_data_folder, 'rank.test')
            # 'data = {}'.format(data_path)
            # 'valid_data = {}'.format(valid_data_path)
            # 'data = {}'.format(test_data_path)
            os.system('echo "data = {}\n" >> train.conf'.format(data_path))
            os.system('echo "valid_data = {}\n" >> train.conf'.format(valid_data_path))
            os.system('echo "output_model = LightGBM_model\n" >> train.conf'.format(valid_data_path))
            os.system('./lightgbm config=train.conf')

            os.system('echo "input_model = LightGBM_model\n" >> predict.conf')
            os.system('echo "data = {}\n" >> predict.conf'.format(test_data_path))
            os.system('echo "output_result = LightGBM_predict_result\n" >> predict.conf')
            os.system('./lightgbm config=predict.conf')
            # prediction scores in LightGBM_predict_result.txt
            run_pl(lightgbm_folder, fold)


if __name__ == '__main__':
	main()
	print('Done')
