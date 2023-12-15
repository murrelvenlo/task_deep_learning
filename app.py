import streamlit as st
import os
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
import requests
import time
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 


# Streamlit App Header
st.title("Image Classification Task Introduction")

# Introduction Text
intro_text = """
In my deep learning exploration, crafting an effective Convolutional Neural Network (CNN) for image classification is an art. I've learned to weave together 2D convolutions, pooling layers, activation functions like ReLU, and regularization techniques such as Dropout and Batch Normalization. This process, coupled with image augmentation, leads to a flattened layer for categorical classification.

This challenge invites me to design a CNN from scratch, testing its performance on a self-collected dataset. I'll compare my results with Google's Teachable Machines for insights.

For my classifier, I've chosen five passion-driven flower categories: "roses," "sunflowers," "orchids," "tulips," and "daisies." Let the exploration of my CNN's efficacy begin.
"""

# Display the Introduction
st.write(intro_text)

# Sidebar for selecting the step
step = st.sidebar.radio("Select Step", ["Image Scraping", "Dataset Preparation","CNN Network", "Model Training", "Model Comparison"])

if step == "Image Scraping":
    st.header("1. Image Scraping")
    st.markdown("**Dataset Collection Overview:**")
    st.write("""
    I curated a dataset for image classification by selecting five categories: "roses," "sunflowers," "orchids," "tulips," and "daisies." Using Selenium, I crafted a Python scraper to automate image collection from [flickr](https://www.flickr.com/), ensuring code understanding and customization. The dataset includes a minimum of 100 images per category, sourced through simulated browser interactions. Challenges were addressed by adapting and extending code snippets, and the process details, including category choices and challenges, were documented for clarity.
    """)

    code = '''import os
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
import requests
import time



def download_images(search_query, category, num_images=100):
    # Create a directory for the category
    category_dir = f"./resources/images/flowers/{category}"
    os.makedirs(category_dir, exist_ok=True)

    # Set up Firefox options
    firefox_options = Options()
    firefox_options.binary_location = 'C:/Program Files/Mozilla Firefox/firefox.exe'  # Replace with the actual path to your firefox.exe
    
    # Set up the service
    geckodriver_path = './geckodriver-v0.33.0-win32/geckodriver.exe'
    ser = Service(geckodriver_path)

    # Create a Firefox webdriver instance
    driver = webdriver.Firefox(service=ser, options=firefox_options)

    # Construct the search URL
    search_url = f"https://www.flickr.com/search/?text={search_query}"

    # Open the search URL in the browser
    driver.get(search_url)

    # Scroll down to load more images (simulate scrolling in the browser)
    for _ in range(num_images // 25):  # Flickr typically loads 25 images at a time
        driver.execute_script(f"window.scrollBy(0, 1000);")
        time.sleep(2)
        
    # Extract image URLs based on the HTML structure of the search results
    img_tags = driver.find_elements(by=By.CSS_SELECTOR, value='.photo-list-photo-container img')
    img_urls = [img.get_attribute('src') for img in img_tags]

    # Download the images (limit to num_images)
    for i, img_url in enumerate(img_urls[:num_images]):
        full_img_url = urljoin(search_url, img_url)  # Join base URL with relative image URL
        img_data = requests.get(full_img_url).content
        img_path = os.path.join(category_dir, f"{category}_{i + 1}.jpg")
        with open(img_path, 'wb') as f:
            f.write(img_data)

    print(f"Downloaded {min(len(img_urls), num_images)} images for {category}.")

    # Close the browser
    driver.quit()

# Specify the categories
categories = ["roses", "sunflowers", "orchids", "tulips", "daisy"]
num_images_per_category = 200

# Loop through categories and download images
for category in categories:
    print(f"Downloading images for {category}...")
    download_images(category, category, num_images=num_images_per_category)
'''

    st.code(code, language='python')

    # Function to download images
    def download_images(search_query, category, num_images=100):
        # Create a directory for the category
        category_dir = f"./images/flowers/{category}"
        os.makedirs(category_dir, exist_ok=True)

        # Set up Firefox options
        firefox_options = Options()
        firefox_options.binary_location = 'C:/Program Files/Mozilla Firefox/firefox.exe'  # Replace with the actual path to your firefox.exe

        # Set up the service
        geckodriver_path = './geckodriver-v0.33.0-win32/geckodriver.exe'
        ser = Service(geckodriver_path)

        # Create a Firefox webdriver instance
        driver = webdriver.Firefox(service=ser, options=firefox_options)

        # Construct the search URL
        search_url = f"https://www.flickr.com/search/?text={search_query}"

        # Open the search URL in the browser
        driver.get(search_url)

        # Scroll down to load more images (simulate scrolling in the browser)
        for _ in range(num_images // 25):  # Flickr typically loads 25 images at a time
            driver.execute_script(f"window.scrollBy(0, 1000);")
            time.sleep(1)

        # Extract image URLs based on the HTML structure of the search results
        img_tags = driver.find_elements(by=By.CSS_SELECTOR, value='.photo-list-photo-container img')
        img_urls = [img.get_attribute('src') for img in img_tags]

        # Download the images (limit to num_images)
        for i, img_url in enumerate(img_urls[:num_images]):
            full_img_url = urljoin(search_url, img_url)  # Join base URL with relative image URL
            img_data = requests.get(full_img_url).content
            img_path = os.path.join(category_dir, f"{category}_{i + 1}.jpg")
            with open(img_path, 'wb') as f:
                f.write(img_data)

        st.success(f"Downloaded {min(len(img_urls), num_images)} images for {category}.")

        # Close the browser
        driver.quit()

    # Download Button
    if st.button("Download Images"):
        # Specify the categories
        categories = ["roses", "sunflowers", "orchids", "tulips", "daisy"]
        num_images_per_category = 200

        # Loop through categories and download images
        for category in categories:
            st.write(f"Downloading images for {category}...")
            download_images(category, category, num_images=num_images_per_category)

    st.write("""
    Below, you can download the images for each category.
    """)
elif step == "Dataset Preparation":
    st.header("2. EDA & Data preparation")
    st.write("""
             After successfully collecting the dataset through web scraping, the next step involves Exploratory Data Analysis (EDA) and preparing the data for training the Convolutional Neural Network (CNN).
             """)
    st.subheader("""
                 Exploratory Data Analysis (EDA)
                 """)
    st.markdown("Class Distribution")
    st.write("""
             I will conduct a brief Exploratory Data Analysis (EDA) to assess the distribution of images across each class, namely "roses," "sunflowers," "orchids," "tulips," and "daisies."
I will employ visualization techniques to observe and comprehend the number of images within each class, providing valuable insights into the overall dataset balance.
             """)
    st.text("Before I started, I imported all packages needed for the model")
    code1 = '''import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay'''

    st.code(code1, language='python')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_packages():
    # Loading packages here
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras import optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing import image
    from keras.utils import image_dataset_from_directory
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create a button to load packages
load_button = st.button("Load Packages")

# Check if the button is pressed
if load_button:
    load_packages()
    st.success("Packages loaded successfully!")

st.write("""In this step, I am preparing the dataset for the model by applying necessary preprocessing techniques and augmenting the images using the ImageDataGenerator from Keras.
Data preprocessing involves standardizing the pixel values, resizing images to a consistent shape, and dividing the dataset into training, validation, and test sets.
Augmentation with ImageDataGenerator includes applying transformations like rotation, zooming, and flipping to artificially increase the diversity of the training dataset. This enhances the model's ability to generalize well to unseen data and improves overall performance.""")

code2 = '''# Create ImageDataGenerator for training and validation sets
train_val_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and validation sets
training_set = train_val_datagen.flow_from_directory(
    './images/flowers/training_set',
    subset='training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # I use Categorical, because I have more than one class classes
)
validation_set = train_val_datagen.flow_from_directory(
    './images/flowers/training_set',
    subset='validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # I use Categorical, because I have more than one class
)

# Create generator for the test set
test_set = train_val_datagen.flow_from_directory(
    './images/flowers/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # I use Categorical, because I have more than one class
)
'''

st.code(code2, language='python')

import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data_generators():
    # Create ImageDataGenerator for training and validation sets
    train_val_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Create ImageDataGenerator for the test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators for training and validation sets
    training_set = train_val_datagen.flow_from_directory(
        './images/flowers/training_set',
        subset='training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_set = train_val_datagen.flow_from_directory(
        './images/flowers/training_set',
        subset='validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    # Create generator for the test set
    test_set = test_datagen.flow_from_directory(
        './images/flowers/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    return training_set, validation_set, test_set

# Create a button to load data generators
load_button = st.button("Load Data Generators")

# Check if the button is pressed
if load_button:
    training_set, validation_set, test_set = load_data_generators()
    st.success("Data generators loaded successfully!")


elif step == "CNN Network":
    st.header("3. CNN network with regularisation options")
    st.write("""
             In this step, I am creating a Convolutional Neural Network (CNN) from scratch to serve as the image classifier for the flower dataset.
The design involves defining convolutional layers, max-pooling layers, and fully connected layers. The architecture is inspired by best practices and knowledge gained from class notebooks and online resources.
Regularization techniques, such as dropout and batch normalization, are incorporated to prevent overfitting and enhance the model's generalization capability.
The network is compiled with appropriate loss function, optimizer, and evaluation metric, setting the stage for training and evaluation.
             """)
    st.subheader("Initialization of the CNN")
    st.code('''# initialising the CNN
model = Sequential()''', language='python')
    st.subheader("Step 1 - Convolution")
    st.write("""I will now code the Convolution step.""")
    st.code('''model.add(Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation = 'relu'))''', language='python')
    st.subheader("Step 2 - Pooling")
    st.write("""I will now perform a pooling operation on the resultant feature maps obtained after the convolution operation on an image. My primary aim is to reduce the size of the images as much as possible. I will start by taking the classifier object and adding the pooling layer. I will use Max Pooling on 2x2 matrices.""")
    st.code('''model.add(MaxPooling2D(pool_size = (2, 2)))''', language='python')
    st.write("""Now I will make a dropout layer to prevent overfitting, which functions by randomly eliminating some of the connections between the layers (0.2 means it drops 20% of the existing connections).""")
    st.code('''model.add(Dropout(0.2))''', language='python')
    st.subheader("Repeat step 1 and 2")
    st.write("""Here I will build a second Convolution, Max Pooling and Dropout layer with the same parameters.""")
    st.code('''model.add(Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))''', language='python')
    st.subheader("Step 3 - Flattening & Full connection")
    st.write("""I will now convert all the pooled images into a continuous vector through Flattening. In the last step, I will create a fully connected layer.""")
    st.code('''model.add(Flatten())
model.add(Dense(activation="relu", units=128))''', language='python')
    st.subheader("Output Layer")
    st.write("""Now, it's time for me to initialize the output layer, which should contain only one node, as it is binary classification. This single node will give me a binary output of either a car, bus, train, airplane, or a ship. I will be using a sigmoid activation function for the final layer.

Remember: if I have more than one output neuron, and I want a clear winner among these output neurons, I would have to choose the 'softmax' activation function. Using a sigmoid activation for multiple output neurons would give a probability score for each of these outputs, not a clear winner (nor a '100%' total, for instance output1=0.95, output2=0.75, output3=0.2: so the total sum is not 100% when using sigmoids as the final output layer activation functions...)""")
    st.code('''model.add(Dense(activation="softmax", units=5))''', language='python')
    st.subheader("Compiling the model")
    st.write("""Now that I have completed building my CNN model, it's time to compile it.

Note: I use categorical crossentropy because I have a softmax activation function with multiple neurons in the output layer. I'm comparing the entropy scores between these neurons, and categorical crossentropy is suitable for this scenario.""")
    st.code('''# compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])''', language='python')
 

    st.write("""I can print out the model summary to see what the whole model looks like.""")
    st.code('''print(model.summary())''', language='python')

# Function to initialize the CNN model
def initialize_cnn_model():
    # initializing the CNN
    model = Sequential()

    model.add(Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation='relu'))  # convolution
    model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(activation="relu", units=128))
    model.add(Dense(activation="softmax", units=5))

    # compiling the CNN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Create a button to initialize the CNN model
init_button = st.button("Initialize CNN Model")

# Check if the button is pressed
if init_button:
    # Initialize the CNN model
    cnn_model = initialize_cnn_model()
    
    # Display a success message
    st.success("CNN Model Initialized Successfully!")
    
    # Print the model summary in a more readable format
    st.subheader("CNN Model Summary")
    
    # Display model architecture
    st.text("Model: \"sequential\"")
    st.text("___________________________________________________________________________")
    
    # Display layer details
    for layer in cnn_model.layers:
        st.text(f"Layer (type)                Output Shape              Param #   ")
        st.text("=" * 69)
        st.text(f" {layer.__class__.__name__.lower()} ({', '.join([f'{key}={value}' for key, value in layer.get_config().items()])})")
        st.text(" " * 69)
    
    # Display trainable and non-trainable parameters
    st.text("=" * 69)
    st.text("Trainable params: 710981 (2.71 MB)")
    st.text("Non-trainable params: 0 (0.00 Byte)")
    st.text("___________________________________________________________________________")

    st.code('''NUM_CLASSES = 5

# Create a sequential model with a list of layers
model = tf.keras.Sequential([
  layers.Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (4, 4), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile and train your model as usual
model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

print(model.summary())''', language='python')
    
# Function to create and compile the model
def create_and_compile_model():
    NUM_CLASSES = 5

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
        layers.Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create a button to create and compile the model
create_button = st.button("Create and Compile Model")

# Check if the button is pressed
if create_button:
    # Create and compile the model
    model = create_and_compile_model()

    # Display a success message
    st.success("Model Created and Compiled Successfully!")

    # Print the model summary in a more readable format
    st.subheader("Model Summary")

    # Display model architecture
    st.text("Model: \"sequential\"")
    st.text("_________________________________________________________________")

    # Display layer details
    for layer in model.layers:
        st.text(f"Layer (type)                Output Shape              Param #   ")
        st.text("=" * 69)
        st.text(f" {layer.__class__.__name__.lower()} ({', '.join([f'{key}={value}' for key, value in layer.get_config().items()])})")
        st.text(" " * 69)

    # Display trainable and non-trainable parameters
    st.text("=" * 69)
    st.text(f"Trainable params: {model.count_params()} ({model.count_params() * 4 / (1024 ** 2):.2f} MB)")
    st.text("Non-trainable params: 0 (0.00 Byte)")
    st.text("_________________________________________________________________")

    st.code('''NUM_CLASSES = 5
IMG_SIZE = 64
# There is no shearing option anymore, but there is a translation option
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Create a sequential model with a list of layers
model_new = tf.keras.Sequential([
  # Add a resizing layer to resize the images to a consistent shape
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  # Add a rescaling layer to rescale the pixel values to the [0, 1] range
  layers.Rescaling(1./255),
  # Add some data augmentation layers to apply random transformations during training
  layers.RandomFlip("horizontal"),
  layers.RandomTranslation(HEIGTH_FACTOR,WIDTH_FACTOR),
  layers.RandomZoom(0.2),



  layers.Conv2D(32, (4, 4), input_shape = (64, 64, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (4, 4), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(), # Or, layers.GlobalAveragePooling2D()
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile and train your model as usual
model_new.compile(optimizer = optimizers.Adam(learning_rate=0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
''', language='python')
    
# Global variables
NUM_CLASSES = 5
IMG_SIZE = 64
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Function to load the new model
def load_new_model():
    global NUM_CLASSES, IMG_SIZE, HEIGTH_FACTOR, WIDTH_FACTOR

    # Create a sequential model with a list of layers
    model_new = tf.keras.Sequential([
        # Add a resizing layer to resize the images to a consistent shape
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        # Add a rescaling layer to rescale the pixel values to the [0, 1] range
        layers.Rescaling(1./255),
        # Add some data augmentation layers to apply random transformations during training
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
        layers.RandomZoom(0.2),
        layers.Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),  # Or, layers.GlobalAveragePooling2D()
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile the model
    model_new.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # Build the model
    model_new.build((None, 64, 64, 3))

    return model_new


# Create a button to load the new model
load_model_button = st.button("Load New Model")

# Check if the button is pressed
if load_model_button:
    # Load the new model
    model_new = load_new_model()

    # Display a success message
    st.success("New Model Loaded Successfully!")

    # Display the model summary
    st.subheader("New Model Summary")
    with st.echo():
        # Display model architecture
        st.text("Model: \"sequential\"")
        st.text("_" * 69)

        # Display layer details
        for layer in model_new.layers:
            st.text(f" Layer (type)                Output Shape              Param #   ")
            st.text("_" * 69)
            st.text(f" {layer.__class__.__name__.lower()} ({', '.join([f'{key}={value}' for key, value in layer.get_config().items()])})")
            st.text(" " * 69)

        # Display trainable and non-trainable parameters
        st.text("_" * 69)
        st.text(f" Trainable params: {model_new.count_params()} ({model_new.count_params() * 4 / (1024 ** 2):.2f} MB)")
        st.text(" Non-trainable params: 0 (0.00 Byte)")
        st.text("_" * 69)
    
    code4 = '''# Set the parameters for your data
batch_size = 32
image_size = (64, 64)
validation_split = 0.2

# Create the training dataset with error handling
train_ds = image_dataset_from_directory(
    directory='./images/flowers/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

validation_ds = image_dataset_from_directory(
    directory='./images/flowers/training_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Create the testing dataset from the 'test' directory
test_ds = image_dataset_from_directory(
    directory='./images/flowers/test_set',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size
)'''

    st.code(code4, language='python')

# parameters for the data
batch_size = 32
image_size = (64, 64)
validation_split = 0.2

# Function to count the number of files in a directory
def count_files(directory):
    return sum([len(files) for root, dirs, files in os.walk(directory)])

# Function to load datasets
def load_datasets():
    global batch_size, image_size  # Access the global variables

    # Create the training dataset with error handling
    train_ds = image_dataset_from_directory(
        directory='./images/flowers/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='training',
        seed=123
    )

    validation_ds = image_dataset_from_directory(
        directory='./images/flowers/training_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='validation',
        seed=123
    )

    # Create the testing dataset from the 'test' directory
    test_ds = image_dataset_from_directory(
        directory='./images/flowers/test_set',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    # Get the number of classes by counting subdirectories in the training set directory
    num_classes = len([name for name in os.listdir('./images/flowers/training_set') if os.path.isdir(os.path.join('./images/flowers/training_set', name))])

    # Count the number of files for each dataset
    train_samples = count_files('./images/flowers/training_set')
    validation_samples = count_files('./images/flowers/training_set') * validation_split
    test_samples = count_files('./images/flowers/test_set')

    return train_ds, validation_ds, test_ds, num_classes, train_samples, validation_samples, test_samples


# Create a button to load datasets
load_button = st.button("Load Datasets")

# Check if the button is pressed
if load_button:
    # Load datasets
    train_ds, validation_ds, test_ds, num_classes, train_samples, validation_samples, test_samples = load_datasets()

    # Display a success message
    st.success("Datasets Loaded Successfully!")

    # Display information about the datasets
    st.subheader("Dataset Information")

    st.write("**Training Dataset:**")
    st.write(f"Found {train_samples} files belonging to {num_classes} classes.")
    st.write(f"Using {train_samples} files for training.")

    st.write("**Validation Dataset:**")
    st.write(f"Found {validation_samples} files belonging to {num_classes} classes.")
    st.write(f"Using {validation_samples} files for validation.")

    st.write("**Testing Dataset:**")
    st.write(f"Found {test_samples} files belonging to {num_classes} classes.")

elif step == "Model Training":
    st.header("4. Model Training")
    st.write("""I will proceed to train the designed CNN model, keeping a close eye on the validation error to prevent overfitting. During training, I will monitor both training and validation errors and visualize their curves to assess the model's performance. The objective is to achieve a balance where the model generalizes well to unseen data without fitting too closely to the training set.""")
    st.write("""Once I'm satisfied with the model's performance, I will compute the confusion matrix on the test set. This matrix will provide a detailed breakdown of the model's predictions, allowing for an in-depth analysis of its classification accuracy for each class. The insights gained from the confusion matrix will be valuable in understanding the strengths and weaknesses of the model in the context of the specific image classification task.""")
    st.text("""Now I'm going to train my network using the 'older' version of our datasets, via the ImageDataGenerators: training_set, validation_set, test_set.
            Later, I'll use the 'newer' version of my model, model_new, and train_ds/valid_ds/test_ds.""")
    
    st.code('''history = model.fit(training_set,
                        validation_data=validation_set,
                        steps_per_epoch=10,
                        epochs=20)''', language='python')
    
# Initialize model variable in the global scope
model = None

# Function to create and compile the model
def create_and_compile_model():
    NUM_CLASSES = 5

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
        layers.Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, training_set, validation_set):
    history = model.fit(training_set,
                        validation_data=validation_set,
                        steps_per_epoch=10,
                        epochs=20)
    return history

# Create a button to create and compile the model
create_button = st.button("Create and Compile Model", key=hash("create_button"))

# Check if the button is pressed
if create_button:
    # Create and compile the model
    model = create_and_compile_model()

    # Display a success message
    st.success("Model Created and Compiled Successfully!")

    # Print the model summary in a more readable format
    st.subheader("Model Summary")

    # Display model architecture
    st.text("Model: \"sequential\"")
    st.text("_________________________________________________________________")

    # Display layer details
    for layer in model.layers:
        st.text(f"Layer (type)                Output Shape              Param #   ")
        st.text("=" * 69)
        st.text(f" {layer.__class__.__name__.lower()} ({', '.join([f'{key}={value}' for key, value in layer.get_config().items()])})")
        st.text(" " * 69)

    # Display trainable and non-trainable parameters
    st.text("=" * 69)
    st.text(f"Trainable params: {model.count_params()} ({model.count_params() * 4 / (1024 ** 2):.2f} MB)")
    st.text("Non-trainable params: 0 (0.00 Byte)")
    st.text("_________________________________________________________________")

# Simulate your data loading process
# You need to replace this with your actual data loading logic
(x_train, y_train), (x_val, y_val) = mnist.load_data()
training_set = (x_train, y_train)
validation_set = (x_val, y_val)

# Create a button to train the model
train_button = st.button("Train Model", key=hash("train_button"))

# Check if the button is pressed
if train_button:
    if model is not None:  # Check if the model is created
        # Train the model
        history = train_model(model, training_set, validation_set)

        # Display a success message
        st.success("Model Trained Successfully!")

        # Plot the loss and accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ... (the rest of the code remains the same)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Model not created. Please create and compile the model first.")