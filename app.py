import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm 
from sklearn.preprocessing import LabelEncoder

st.title('Prédiction prix voiture')

st.write("""Ce projet contient un formulaire permettant d'estimer la valeur d'un véhicule en prenant en considération plusieurs éléments qui sont les suivants: 
    Marque du véhicule, Gamme du véhicule, Type du véhicule, carburant, nombre de portes, kilometrage parcouru par an, nombre de cheveaux 
    ainsi que le choix entre deux types d'algorithmes différents (DecisionTreeRegressor et RandomForestRegressor)""")

df = pd.read_csv('./CarPrice_Assignment.csv', sep=',')

# Create LabelEncoder
le = LabelEncoder()

# Create new column brand
df['brand'] = df.CarName.str.split(' ').str.get(0).str.upper()

# Replace
df['brand'] = df['brand'].replace(['VW', 'VOKSWAGEN'], 'VOLKSWAGEN')
df['brand'] = df['brand'].replace(['MAXDA'], 'MAZDA')
df['brand'] = df['brand'].replace(['PORCSHCE'], 'PORSCHE')
df['brand'] = df['brand'].replace(['TOYOUTA'], 'TOYOTA')

df_comp_avg_price = df[['brand','price']].groupby("brand", as_index = False).mean().rename(columns={'price':'brand_avg_price'})
df = df.merge(df_comp_avg_price, on = 'brand')
df['brandcategory'] = df['brand_avg_price'].apply(lambda x : "Bas_gamme" if x < 10000 else ("Moyenne_gamme" if 10000 <= x < 20000 else "Haut_gamme"))

# Use LabelEncoder on brand, fueltype, carbody, doornumber
df['brand_encoded'] = le.fit_transform(df.brand.values)
df['fuelType_encoded'] = le.fit_transform(df.fueltype.values)
df['carBody_encoded'] = le.fit_transform(df.carbody.values)
df['doorNumber_encoded'] = le.fit_transform(df.doornumber.values)
df['brandCategory_encoded'] = le.fit_transform(df.brandcategory.values)
df['brandName_encoded'] = le.fit_transform(df.CarName.values)

# Create variable conversion
brandConversion = {
    'ALFA-ROMERO': 0,
    'AUDI': 1,
    'BMW': 2,
    'BUICK': 3,
    'CHEVROLET': 4,
    'DODGE': 5,
    'HONDA': 6,
    'ISUZU': 7,
    'JAGUAR': 8,
    'MAZDA': 9,
    'MERCURY': 10,
    'MITSUBISHI': 11,
    'NISSAN': 12,
    'PEUGEOT': 13,
    'PLYMOUTH': 14,
    'PORSCHE': 15,
    'RENAULT': 16,
    'SAAB': 17,
    'SUBARU': 18,
    'TOYOTA': 19,
    'VOLKSWAGEN': 20,
    'VOLVO': 21,
}

carBodyConversion = {
    'Cabriolet': 0,
    'Pick-up': 1,
    'Citadine': 2,
    'Berline': 3,
    'Break': 4,
}

fuelTypeConversion = {
    'Essence': 1,
    'Diesel': 0
}

doorNumberConversion = {
    'Deux': 1,
    'Quatre': 0,
}

brandCategoryConversion = {
    'Bas_gamme': 0,
    'Haut_gamme': 1,
    'Moyenne_gamme': 2
}


# Create new column "mileage" in one year
df['mileage'] = (df['citympg'] * 0.55 + df['highwaympg'] * 0.45) * 365

auto = df[['brand_encoded', 'brandCategory_encoded', 'carBody_encoded', 'fuelType_encoded', 'doorNumber_encoded', 'mileage', 'horsepower', 'price']]

dfCarbody = df['carbody'].replace(['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'], ['Cabriolet', 'Citadine', 'Berline', 'Break', 'Pick-up'])
dfFuelType = df['fueltype'].replace(['gas', 'diesel'], ['Essence', 'Diesel'])
dfDoorNumber = df['doornumber'].replace(['two', 'four'], ['Deux', 'Quatre'])

brand = st.selectbox('Marque du véhicule', df.brand.unique())
brandCategory = st.selectbox('Gamme du véhicule', ['Bas_gamme', 'Moyenne_gamme', 'Haut_gamme'])
carBody = st.selectbox('Type de véhicule', dfCarbody.unique())
fuelType = st.selectbox('Carburant', dfFuelType.unique())
doorNumber = st.selectbox('Nombre de porte', dfDoorNumber.unique())
mileage = st.text_input("Kilométrage parcouru par an", '')
horsePower = st.text_input("Puissance (Ch)", '')
choiceAlgo = st.selectbox('Choix Algo prédiction', ['RandomForestRegressor', 'DecisionTreeRegressor'])

button = st.button('Estimation')

if button:
    mileage = int(mileage)
    horsePower = int(horsePower)
    model = RandomForestRegressor(random_state=42)

    if choiceAlgo == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=42)

    y_test = auto.pop('price')
    X_test = auto
    model.fit(X_test, y_test)
    prediction = model.predict([[brandConversion[brand], brandCategoryConversion[brandCategory], carBodyConversion[carBody], fuelTypeConversion[fuelType], doorNumberConversion[doorNumber], mileage, horsePower]])
    price = round(prediction[0],2)
    
    st.write("L'estimation est de : ", price, " $")

    # Prediction algo
    y_pred = model.predict(X_test)
    st.write("Le rendement d'efficacité de l'algorithme : ", r2_score(y_test, y_pred))
