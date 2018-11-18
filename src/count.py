import pandas as pd

def count():

    landmarks_path = '/vagrant/imgs/list_attr_celeba.csv'
    df = pd.read_csv(landmarks_path)
    headers = list(df)
    for header in headers:
    	print(header)
    	print(df[header].value_counts()[-1])
    	print(df[header].value_counts()[1])
    	print()

if __name__ == "__main__":
	count()
