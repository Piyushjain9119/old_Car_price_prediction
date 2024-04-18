import pandas as pd
import numpy as np 
import streamlit as st
import pickle
# from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import warnings 









model = pickle.load(open('Model1.pkl','rb'))

page_bg_img = """
<style>

[data-testid="stAppViewContainer"]{
background-image :url('https://www.hdcarwallpapers.com/walls/lamborghini_gallardo_spyder_adv1-wide.jpg');

background-size :cover;
}

</style>

"""




st.markdown(page_bg_img,unsafe_allow_html=True)

st.title('CAR PRICE PREDICTION')
st.sidebar.header("Car Data")
# # image = Image.open('ggs.jpg')
# # st.image(image,'')




# Name = st.text_input(
#     label = "Enter the name of the car",
#     max_chars= 15,
#     placeholder="NAME"
# )

model1 = ['Mitsubishi Montero', 'Ford Freestyle', 'Renault Pulse', 'Maruti Omni', 'Renault Duster', 'Nissan Terrano', 'Mitsubishi Cedia', 'Nissan X-Trail', 'Mahindra Jeep', 'Toyota Etios', 'Chevrolet Captiva', 'Fiat Punto', 'Tata Indica', 'Maruti 1000', 'Audi A6', 'Volkswagen Passat', 'Maruti Vitara', 'Nissan Sunny', 'BMW 3', 'Tata Bolt', 'Hyundai EON', 'Mahindra Renault', 'Toyota Fortuner', 'Ford Classic', 'Toyota Qualis', 'Mahindra TUV', 'Mercedes-Benz A', 'Mercedes-Benz M-Class', 'Chevrolet Optra', 'Maruti Dzire', 'Renault Koleos', 'Mitsubishi Pajero', 'Hyundai Santro', 'Ford Fusion', 'Honda Accord', 'Maruti Versa', 'Maruti Swift', 'Nissan Evalia', 'Renault Fluence', 'Maruti Ritz', 'Hyundai Accent', 'Ford Figo', 'Honda CR-V', 'Hyundai Verna', 'Mercedes-Benz New', 'Audi Q5', 'Honda Amaze', 'Mahindra Ssangyong', 'Toyota Platinum', 'Mercedes-Benz E-Class', 'Hyundai Elite', 'Fiat Siena', 'Maruti S', 'Mitsubishi Lancer', 'Audi Q7', 'Chevrolet Tavera', 'Honda BR-V', 'Honda Brio', 'Nissan Teana', 'Maruti Esteem', 'Maruti Celerio', 'Chevrolet Enjoy', 'Mahindra Quanto', 'Maruti 800', 'Mercedes-Benz S-Class', 'Honda BRV', 'Toyota Camry', 'Skoda Octavia', 'Mercedes-Benz S', 'Maruti Ignis', 'Maruti Grand', 'Datsun redi-GO', 'Skoda Superb', 'Hyundai Elantra', 'Tata Safari', 'Hyundai Creta', 'Audi A4', 'Maruti Eeco', 'Ford EcoSport', 'Hyundai i20', 'Hyundai Sonata', 'Tata Manza', 'Maruti S-Cross', 'Hyundai Xcent', 'Ford Ikon', 'Volkswagen CrossPolo', 'Maruti A-Star', 'Datsun Redi', 'Skoda Fabia', 'Fiat Grande', 'Chevrolet Cruze', 'Volkswagen Ameo', 'Maruti Alto', 'Volkswagen Polo', 'Maruti SX4', 'Tata Tigor', 'Honda Civic', 'Skoda Rapid', 'Tata Zest', 'Chevrolet Beat', 'BMW X1', 'Renault Lodgy', 'Land Rover', 'Ambassador Classic', 'Tata Venture', 'BMW 5', 'Volvo S80', 'Datsun GO', 'Fiat Avventura', 'BMW 7', 'Ford Aspire', 'Honda City', 'Honda WR-V', 'Tata Tiago', 'Mahindra Thar', 'Mahindra KUV', 'Maruti Ertiga', 'Mahindra NuvoSport', 'Volkswagen Jetta', 'Hyundai Grand', 'Toyota Corolla', 'Mahindra Verito', 'Maruti Zen', 'Tata Nexon', 'Fiat Linea', 'Fiat Petra', 'Mahindra Logan', 'Hyundai i10', 'Ford Ecosport', 'Maruti Wagon', 'Tata Nano', 'Maruti Estilo', 'Mahindra Bolero', 'Porsche Cayenne', 'Hyundai Santa', 'Mahindra Xylo', 'ISUZU D-MAX', 'Tata Xenon', 'Mercedes-Benz B', 'Tata Sumo', 'Honda WRV', 'Honda Jazz', 'Ford Endeavour', 'Mahindra XUV500', 'Toyota Innova', 'Tata New', 'Nissan Micra', 'Maruti Baleno', 'Hyundai Getz', 'Chevrolet Spark', 'Chevrolet Aveo', 'Chevrolet Sail', 'Tata Indigo', 'Mitsubishi Outlander', 'Maruti Ciaz', 'Mahindra XUV300', 'Mahindra Scorpio', 'Skoda Yeti', 'Smart Fortwo', 'Volkswagen Vento', 'Skoda Laura', 'Force One', 'Honda Mobilio', 'Renault Scala', 'Hyundai Tucson', 'Ford Fiesta', 'Renault KWID','Jaguar XF', 'Hyundai Sonata', 'Ford EcoSport', 'Volvo V40', 'Honda CR-V', 'Hyundai Creta', 'Audi A8', 'Toyota Prius', 'Renault Duster', 'Hyundai Verna', 'Audi Q5', 'Mercedes-Benz New', 'Audi Q3', 'Isuzu MUX', 'Volkswagen Beetle', 'Mini Cooper', 'Audi A3', 'Mercedes-Benz GLA', 'BMW X6', 'Porsche Cayenne', 'Mahindra Ssangyong', 'Hyundai Santa', 'BMW X3', 'Mercedes-Benz E-Class', 'Volvo XC60', 'ISUZU D-MAX', 'Audi Q7', 'Jeep Compass', 'Mercedes-Benz B', 'Audi A6', 'BMW X5', 'Tata Hexa', 'Ford Endeavour', 'Mahindra XUV500', 'Toyota Innova', 'BMW 3', 'Toyota Fortuner', 'BMW X1', 'Hyundai Elantra', 'Mercedes-Benz A', 'Mercedes-Benz M-Class', 'Land Rover', 'Mercedes-Benz S-Class', 'Renault Captur', 'Mahindra Scorpio', 'BMW 5', 'BMW 6', 'Toyota Camry', 'Volvo S60', 'BMW 7', 'Mercedes-Benz R-Class', 'Mitsubishi Pajero', 'Honda City', 'Mahindra E', 'Mercedes-Benz S', 'Skoda Octavia', 'Volkswagen Vento', 'Force One', 'Skoda Superb', 'BMW 1', 'Hyundai Tucson', 'Audi A4', 'Volkswagen Jetta', 'Toyota Corolla','Mercedes-Benz GL-Class', 'Jaguar XF', 'Volvo V40', 'Audi A8', 'Audi A7', 'Bentley Continental', 'Ford Mustang', 'Audi Q5', 'Mercedes-Benz New', 'Audi Q3', 'Mercedes-Benz SL-Class', 'Mini Cooper', 'Mercedes-Benz GLS', 'Audi A3', 'Mercedes-Benz GLA', 'BMW X6', 'Porsche Cayenne', 'Lamborghini Gallardo', 'Mercedes-Benz SLK-Class', 'Jaguar XJ', 'BMW Z4', 'Mercedes-Benz C-Class', 'Hyundai Santa', 'BMW X3', 'Mercedes-Benz E-Class', 'Volvo XC60', 'Mercedes-Benz SLC', 'Volvo XC90', 'Volkswagen Tiguan', 'Audi Q7', 'Audi TT', 'Jeep Compass', 'Audi A6', 'BMW X5', 'Mercedes-Benz GLE', 'Ford Endeavour', 'Toyota Innova', 'BMW 3', 'Porsche Boxster', 'Mercedes-Benz GLC', 'Toyota Fortuner', 'Porsche Panamera', 'BMW X1', 'Mercedes-Benz M-Class', 'Land Rover', 'Audi RS5', 'Mercedes-Benz S-Class', 'BMW 5', 'BMW 6', 'Toyota Camry', 'Jaguar F', 'BMW 7', 'Jaguar XE', 'Mercedes-Benz R-Class', 'Mitsubishi Pajero', 'Volvo S60', 'Mercedes-Benz S', 'Mini Clubman', 'Mercedes-Benz CLS-Class', 'Skoda Superb', 'Mercedes-Benz CLA', 'Audi A4', 'Porsche Cayman', 'Mini Countryman']
# model2 = []
# # model3 = []
Name = st.selectbox("Enter the name of the car", placeholder="Name",model1)


# if Name in model1:
#     Name = 'Tier1'
# elif Name in model2:
#     Name = 'Tier2'
# else:
#     Name = 'Tier3'

# st.write(Name)



loc = ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur','Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad']

Location= st.selectbox("SELECT LOCATION" , loc )




Year= st.number_input(
    label = "Enter the Model Year",
    min_value=2000,
    max_value=2019,
    value = 2015,
    step = 1
)



Kilometers_Driven = st.slider(
    label = "Enter the approx range of Kilometers_Driven",
    min_value=1000,
    max_value=100000,
    value = 10000,
    step = 1000
)



ft = ['CNG', 'Diesel', 'Petrol', 'LPG' ,'Electric']
Fuel_Type = st.selectbox("Select fuel type" , ft)


Trans = ['Manual', 'Automatic']
Transmission = st.selectbox("Select Type",Trans)


Ot= ['First', 'Second' ,'Fourth & Above' ,'Third']
Owner_Type = st.selectbox("Select Owner Type" , Ot)



user_given_data = {
    'Name' :Name,
    'Location':Location,
    'Year' :Year,
    'Kilometers_Driven' :Kilometers_Driven,
    'Fuel_Type' :Fuel_Type,
    'Transmission' :Transmission,
    'Owner_Type' :Owner_Type

}
# user_given_data = {
#     'Name' :'Tier3',
#     'Location':'Mumbai',
#     'Year' :2015,
#     'Kilometers_Driven' :73000,
#     'Fuel_Type' :'Petrol',
#     'Transmission' :'Automatic',
#     'Owner_Type' :'Second'

# }

given_data = pd.DataFrame(user_given_data,index = [0])


st.header("Given Information")
st.write(given_data)


price = model.predict(given_data)
st.header("Car Price",divider = 'rainbow')
st.header(np.round(np.abs(price)[0]),'lac' )#st.header("lac")
st.header(':rainbow[THE ABOVE VALUE IS IN LAKH, THANKS FOR USING] :sunglasses:')

