import pymongo
import pandas as pd

def connect(df):
    
    data=df.to_dict(orient='records')
    
    DB_NAME = "Proj1"
    COLLECTION_NAME = "Proj1-Data"
    CONNECTION_URL = f"mongodb+srv://haseebpro343:9GU3C6OGNhnLuIJg@cluster0.xg6z3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    
    client = pymongo.MongoClient(CONNECTION_URL)
    data_base = client[DB_NAME]
    collection = data_base[COLLECTION_NAME]
    
    collection.insert_many(data)
    print('Data inserted successfully')
    
def main():
    df = pd.read_csv('data.csv')
    connect(df)
    
if __name__ == '__main__':
    main()
    