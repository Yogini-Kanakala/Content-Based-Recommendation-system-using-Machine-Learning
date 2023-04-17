from utils import preprocess_data, generate_vectors
import pandas as pd
import pickle


main_df = pd.read_csv("../data/mast_product_level_information.csv")
df = preprocess_data(main_df)

print("data loaded..")

vectorizer, model = generate_vectors(df)

print('vectors generated')

with open('feature_store/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_store/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print('models saved..')