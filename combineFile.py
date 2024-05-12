import streamlit as st
import pickle
import tensorflow
import cv2
import os
from PIL import Image
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process


df = pd.read_csv("data.csv")

#removing the empty column
df1 = df[df.columns[:-1]]

#dropping year 
df1 = df1.drop(columns='year')






def text_input_from_ui(user_input_text):

    # Text vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df1['Search'].fillna(''))

    # Build NearestNeighbors model
    neighbors_model = NearestNeighbors(metric='cosine')
    neighbors_model.fit(tfidf_matrix)

    # Function to recommend items based on product display name
    def recommend_items(product_display_name, n_neighbors=10):
        query_vector = tfidf.transform([product_display_name])
        distances, indices = neighbors_model.kneighbors(query_vector, n_neighbors=n_neighbors + 1)
        return indices

    # product_to_search = "red check shirt"




    valid_categories = ['Navy Blue', 'Blue', 'Silver', 'Black', 'Grey', 'Green', 'Purple',
        'White', 'Beige', 'Brown', 'Bronze', 'Teal', 'Copper', 'Pink',
        'Off White', 'Maroon', 'Red', 'Khaki', 'Orange', 'Coffee Brown',
        'Yellow', 'Gold', 'Steel', 'Tan', 'Multi', 'Magenta',
        'Lavender', 'Sea Green', 'Cream', 'Peach', 'Olive', 'Skin',
        'Burgundy', 'Grey Melange', 'Rust', 'Rose', 'Lime Green', 'Mauve',
        'Turquoise Blue', 'Metallic', 'Taupe', 'Nude',
        'Mushroom Brown', 'Fluorescent Green','Topwear', 'Bottomwear', 'Watches', 'Socks', 'Shoes', 'Belts',
        'Flip Flops', 'Bags', 'Innerwear', 'Sandal', 'Shoe Accessories',
        'Fragrance', 'Jewellery', 'Lips', 'Saree', 'Eyewear', 'Nails',
        'Scarves', 'Dress', 'Loungewear and Nightwear', 'Wallets',
        'Apparel Set', 'Headwear', 'Mufflers', 'Skin Care', 'Makeup',
        'Free Gifts', 'Ties', 'Accessories', 'Skin', 'Beauty Accessories',
        'Water Bottle', 'Eyes', 'Bath and Body', 'Gloves',
        'Sports Accessories', 'Cufflinks', 'Sports Equipment', 'Stoles',
        'Hair', 'Perfumes', 'Home Furnishing', 'Umbrellas', 'Wristbands',
        'Vouchers','Men', 'Women', 'Boys', 'Girls', 'Unisex','Apparel', 'Accessories', 'Footwear', 'Personal Care',
        'Free Items', 'Sporting Goods', 'Home','Fall', 'Summer', 'Winter', 'Spring','Casual', 'Ethnic', 'Formal', 'Sports', 'Smart Casual',
        'Travel', 'Party', 'Home','Shirts', 'Jeans', 'Watches', 'Track Pants', 'Tshirts', 'Socks',
        'Casual Shoes', 'Belts', 'Flip Flops', 'Handbags', 'Tops', 'Bra',
        'Sandals', 'Shoe Accessories', 'Sweatshirts', 'Deodorant',
        'Formal Shoes', 'Bracelet', 'Lipstick', 'Flats', 'Kurtas',
        'Waistcoat', 'Sports Shoes', 'Shorts', 'Briefs', 'Sarees',
        'Perfume and Body Mist', 'Heels', 'Sunglasses', 'Innerwear Vests',
        'Pendant', 'Nail Polish', 'Laptop Bag', 'Scarves', 'Rain Jacket',
        'Dresses', 'Night suits', 'Skirts', 'Wallets', 'Blazers', 'Ring',
        'Kurta Sets', 'Clutches', 'Shrug', 'Backpacks', 'Caps', 'Trousers',
        'Earrings', 'Camisoles', 'Boxers', 'Jewellery Set', 'Dupatta',
        'Capris', 'Lip Gloss', 'Bath Robe', 'Mufflers', 'Tunics',
        'Jackets', 'Trunk', 'Lounge Pants', 'Face Wash and Cleanser',
        'Necklace and Chains', 'Duffel Bag', 'Sports Sandals',
        'Foundation and Primer', 'Sweaters', 'Free Gifts', 'Trolley Bag',
        'Tracksuits', 'Swimwear', 'Shoe Laces', 'Fragrance Gift Set',
        'Bangle', 'Nightdress', 'Ties', 'Baby Dolls', 'Leggings',
        'Highlighter and Blush', 'Travel Accessory', 'Kurtis',
        'Mobile Pouch', 'Messenger Bag', 'Lip Care', 'Face Moisturisers',
        'Compact', 'Eye Cream', 'Accessory Gift Set', 'Beauty Accessory',
        'Jumpsuit', 'Kajal and Eyeliner', 'Water Bottle', 'Suspenders',
        'Lip Liner', 'Robe', 'Salwar and Dupatta', 'Patiala', 'Stockings',
        'Eyeshadow', 'Headband', 'Tights', 'Nail Essentials', 'Churidar',
        'Lounge Tshirts', 'Face Scrub and Exfoliator', 'Lounge Shorts',
        'Gloves', 'Mask and Peel', 'Wristbands', 'Tablet Sleeve',
        'Ties and Cufflinks', 'Footballs', 'Stoles', 'Shapewear',
        'Nehru Jackets', 'Salwar', 'Cufflinks', 'Jeggings', 'Hair Colour',
        'Concealer', 'Rompers', 'Body Lotion', 'Sunscreen', 'Booties',
        'Waist Pouch', 'Hair Accessory', 'Rucksacks', 'Basketballs',
        'Lehenga Choli', 'Clothing Set', 'Mascara', 'Toner',
        'Cushion Covers', 'Key chain', 'Makeup Remover', 'Lip Plumper',
        'Umbrellas', 'Face Serum and Gel', 'Hat', 'Mens Grooming Kit',
        'Rain Trousers', 'Body Wash and Scrub', 'Suits', 'Ipad']

    user_input = user_input_text
    list_formate = user_input.split()



    #correcting the user input spelling 

    threshold = 50  
    correct_input_formate = []
    for i in list_formate:
        closest_match, confidence = process.extractOne(i, valid_categories)
        if confidence >= threshold:
            corrected_category = closest_match
            # print(corrected_category)
            correct_input_formate.append(corrected_category)
        else:
            print('No suitable match found.')
    product_to_search = str(correct_input_formate)
    # print(product_to_search)






    recommended_items = recommend_items(product_to_search)
    # print("Recommended items for '{}':".format(product_to_search))
    # print(recommended_items)
    na = list(recommended_items)
    return na[0]














def image_input_from_ui(user_input_image):
    feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
    filenames = pickle.load(open('filenames.pkl','rb'))


    model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
    model.trainable = False

    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])



    img = image.load_img(user_input_image,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distances,indices = neighbors.kneighbors([normalized_result])


    # return indices
    # print("Recommended items for '{}':".format(product_to_search))
    # print(recommended_items)
    na = list(indices)
    return na[0]








    # for file in indices[0][1:6]:
    #     temp_img = cv2.imread(filenames[file])
    #     cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    #     cv2.waitKey()




st.title("Outfit Recommendation System")
search_text = st.text_input("Search for your outfit")
st.subheader("Or")
search_image = st.file_uploader("Upload a photo")

if search_image:
    st.image(search_image, width=100)
    image_indexs = image_input_from_ui(search_image)
elif search_text:
    text_indexs = text_input_from_ui(search_text)
    








