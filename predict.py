from utils import preprocess_input, recommend_products
import pickle
import pandas as pd

nn_model = pickle.load(open('feature_store/model.pkl', 'rb'))
vectorizer = pickle.load(open('feature_store/vectorizer.pkl', 'rb'))
main_df = pd.read_csv('../data/mast_product_level_information.csv')
query = input('Search: ')

res = recommend_products(query, vectorizer, nn_model, main_df, n=10)

print(res.keys())
print(res['Product_ID'])
print(pd.DataFrame(res))