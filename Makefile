
train:
	python train.py

predict:
	python predict.py \
		./data/rawdata/testA_datasets_document_detection/images \
		./results

final_predict:
	python predict.py \
		./data/rawdata/document_detection_testB_dataset \
		./final_results

submit:
	zip submit.zip model.pdparams predict.py
