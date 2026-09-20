
RAW_PATH = "SuperKart_project/data/SuperKart.csv"
df = pd.read_csv(RAW_PATH)

df.drop(columns=["Product_Id"], inplace=True)
df.drop(columns=["Store_Id"], inplace=True)
#removing identity column

#seperated target from features
target = "Product_Store_Sales_Total"
X = df.drop(columns=[target])
y = df[target]

#test train split with 20% test size
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Product_Store_Sales_Total distribution in train:")
print(ytrain.value_counts())
