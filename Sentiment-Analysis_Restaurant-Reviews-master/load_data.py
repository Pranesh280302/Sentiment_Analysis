import pandas as pd

def load_data():
    downloaded_file_path = "/Users/bharatkammakatla/Downloads/yelp_dataset/yelp_academic_dataset_review.json"

    # read the entire file into python array
    with open(downloaded_file_path, 'r') as f:
        data = f.readlines()

    # remove the trailing '\n' from each line
    data = map(lambda x : x.rstrip(), data)

    data_json_format = "[" + ','.join(data) + "]"

    # now load it into pandas dataframe
    data_df = pd.read_json(data_json_format)
    print("loaded into dataframe")

    # insert first 100000 rows into csv file and save it
    data_df.head(100000).to_csv('reviews_100000.csv')

if __name__ == '__main__':
    load_data()