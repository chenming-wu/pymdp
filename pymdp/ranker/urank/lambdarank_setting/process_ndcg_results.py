import glob
import pandas as pd

def getPaths():
	txt_files = glob.glob('*.txt')
	return txt_files
	# for file_path in txt_files:
	# 	print(file_path)

# e.g., file_path = 'MQ2007_glrank.txt'
def get_data_model(file_path):
	fields = file_path.split('.txt')[0]
	dataset_model = fields.split('_')	
	return dataset_model[0], dataset_model[1]


def get_ndcgs(file_path):
	with open(file_path, 'r') as f:
		ndcg_1s = []
		ndcg_3s = []
		ndcg_5s = []
		ndcg_10s = []

		lines = f.readlines()
		for line in lines:
			if line.startswith('- Eval metrics:'):
				# print(line)
				fields = line.split(' ')
				one_fold = []
				ndcg_1 = float(fields[4])
				ndcg_3 = float(fields[7])
				ndcg_5 = float(fields[10])
				ndcg_10 = float(fields[13])

				ndcg_1s.append(ndcg_1)
				ndcg_3s.append(ndcg_3)
				ndcg_5s.append(ndcg_5)
				ndcg_10s.append(ndcg_10)
		if (len(ndcg_1s) != 5 or len(ndcg_3s) != 5 or len(ndcg_5s) != 5 or len(ndcg_10s) != 5):
			print('ERROR! in ', file_path)
		a_ndcg_1 = sum(ndcg_1s) / float(len(ndcg_1s))
		a_ndcg_3 = sum(ndcg_3s) / float(len(ndcg_3s))
		a_ndcg_5 = sum(ndcg_5s) / float(len(ndcg_5s))
		a_ndcg_10 = sum(ndcg_10s) / float(len(ndcg_10s))
		# print('a_ndcg_1 : ', a_ndcg_1)
		# print('a_ndcg_3 : ', a_ndcg_3)
		# print('a_ndcg_5 : ', a_ndcg_5)
		# print('a_ndcg_10 : ', a_ndcg_10)
		return [a_ndcg_1, a_ndcg_3, a_ndcg_5, a_ndcg_10]

txt_files = getPaths()
txt_files.sort()
for file_path in txt_files:
	dataset, model = get_data_model(file_path)
	ncdgs = get_ndcgs(file_path)
	# df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
	# df.to_csv('reult.csv',header=True, index=False)
	print(dataset, model, ncdgs)