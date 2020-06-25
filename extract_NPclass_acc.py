import json
import pickle
import numpy as np

'''
The file first reads answer in file new_topNP_ans.json. Then it checks NP 
annotation file (validation data) and picks question ids whose 
most_frequent_answer lies in the new_topNP_ans.json file. It then extracts 
accuracy of these ids from tr_nonyn_yn_model_preds.pkl. 
'''

top_NP_answers = json.load(open('data/top_np_ans.json','r')) #List of answers

top_NP_answers_first352 = top_NP_answers
top_NP_answers_first352 = top_NP_answers[: 352]
top_NP_answers_last = top_NP_answers[352 :]

validation_file = json.load(open('data/val_ann_np_ans.json','r'))['annotations'] #133540
preds = pickle.load(open('tr_nonyn_yn_model_preds.pkl','rb')) #list of tuples

topNP_ans_qids352 = []
topNP_ans_qidslast = []

for annos in validation_file:

	if annos['multiple_choice_answer'] in top_NP_answers_first352:
		topNP_ans_qids352.append(annos['question_id'])
	if annos['multiple_choice_answer'] in top_NP_answers_last:
		topNP_ans_qidslast.append(annos['question_id'])

qid2score = {}
for tup in preds:
	qid2score[tup[0]] = tup[1]

topNP_ans_score352 = []
topNP_ans_scorelast = []

for qid in topNP_ans_qids352:
	if qid in qid2score:
		topNP_ans_score352.append(qid2score[qid])

for qid in topNP_ans_qidslast:
	if qid in qid2score:
		topNP_ans_scorelast.append(qid2score[qid])

print("Score for top NP classes 352: ", np.mean(topNP_ans_score352))
print("Score for top NP classes last: ", np.mean(topNP_ans_scorelast))




