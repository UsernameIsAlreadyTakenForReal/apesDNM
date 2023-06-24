mkdir /ebs_data/project_datasets
mkdir /ebs_data/project_datasets/d_ekg1
mkdir /ebs_data/project_datasets/d_ekg2
mkdir /ebs_data/project_datasets/d_img1

#aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/ECG5000.txt /ebs_data/d_ekg1/ECG5000.txt
aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/d_ekg1_description.json /ebs_data/project_datasets/d_ekg1/d_ekg1_description.json
aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/ECG5000_TEST.arff /ebs_data/project_datasets/d_ekg1/ECG5000_TEST.arff
aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/ECG5000_TRAIN.arff /ebs_data/project_datasets/d_ekg1/ECG5000_TRAIN.arff
#aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/ECG5000_TEST.txt /ebs_data/d_ekg1/ECG5000_TEST.txt
#aws s3 cp s3://apesdnm-s3/datasets/d_ekg1/ECG5000_TRAIN.txt /ebs_data/d_ekg1/ECG5000_TRAIN.txt

aws s3 cp 's3://apesdnm-s3/datasets/d_ekg2/d_ekg2_description.json' '/ebs_data/project_datasets/d_ekg2/d_ekg2_description.json'
aws s3 cp 's3://apesdnm-s3/datasets/d_ekg2/mitbih_test - test.csv' '/ebs_data/project_datasets/d_ekg2/mitbih_test - test.csv'
aws s3 cp s3://apesdnm-s3/datasets/d_ekg2/mitbih_test.csv /ebs_data/project_datasets/d_ekg2/mitbih_test.csv
aws s3 cp s3://apesdnm-s3/datasets/d_ekg2/mitbih_train.csv /ebs_data/project_datasets/d_ekg2/mitbih_train.csv
aws s3 cp s3://apesdnm-s3/datasets/d_ekg2/ptbdb_abnormal.csv /ebs_data/project_datasets/d_ekg2/ptbdb_abnormal.csv
aws s3 cp s3://apesdnm-s3/datasets/d_ekg2/ptbdb_normal.csv /ebs_data/project_datasets/d_ekg2/ptbdb_normal.csv