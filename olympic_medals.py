import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk

# Load the dataset
file_path = r"C:\Users\User\Documents\Python\olympic medals economic data\olympics-economics.csv"
data = pd.read_csv(file_path)

# Create a new column for GDP per capita by dividing GDP by population
data['gdp_per_capita'] = data['gdp'] / data['population']

# Set the weight for GDP per capita to be 5x more relevant
gdp_weight = 5

# Apply the weight to GDP per capita
data['weighted_gdp_per_capita'] = data['gdp_per_capita'] * gdp_weight

# Weigh down the population by a factor of 10
population_weight = 10
data['weighted_population'] = data['population'] / population_weight

# Selecting features (weighted_gdp_per_capita, weighted_population, gold, silver, bronze)
X = data[['weighted_gdp_per_capita', 'weighted_population', 'gold', 'silver', 'bronze']].values
y = data[['gold', 'silver', 'bronze']].values  # Target remains the same: future gold, silver, and bronze

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_scaled.shape[1],)),  # Adjust input layer size
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')  # 3 outputs: gold, silver, bronze
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Split the data into training and testing sets for training purposes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Predict medals for all countries based on their features
predictions = model.predict(X_scaled)

# Create a function to display predictions in a Tkinter window
def display_predictions():
    # Create Tkinter window
    window = tk.Tk()
    window.title("Olympics Medal Predictions")

    # Create a frame to hold the table and the scrollbar
    frame = ttk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a Treeview widget
    tree = ttk.Treeview(frame, columns=('Country', 'Predicted Gold', 'Predicted Silver', 'Predicted Bronze'), show='headings')

    # Define column headings
    tree.heading('Country', text='Country', anchor=tk.CENTER)
    tree.heading('Predicted Gold', text='Predicted Gold', anchor=tk.CENTER)
    tree.heading('Predicted Silver', text='Predicted Silver', anchor=tk.CENTER)
    tree.heading('Predicted Bronze', text='Predicted Bronze', anchor=tk.CENTER)

    # Set column width and alignment to center
    tree.column('Country', anchor=tk.CENTER, width=150)
    tree.column('Predicted Gold', anchor=tk.CENTER, width=120)
    tree.column('Predicted Silver', anchor=tk.CENTER, width=120)
    tree.column('Predicted Bronze', anchor=tk.CENTER, width=120)

    # Add rows of predictions to the table
    for i, country in enumerate(data['country']):
        tree.insert('', 'end', values=(country, 
                                       round(predictions[i][0], 1), 
                                       round(predictions[i][1], 1), 
                                       round(predictions[i][2], 1)))

    # Create a scrollbar for the Treeview
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Position the scrollbar
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Pack the Treeview into the frame
    tree.pack(fill=tk.BOTH, expand=True)

    # Run the Tkinter event loop
    window.mainloop()

# Call the function to display the Tkinter window with predictions
display_predictions()
