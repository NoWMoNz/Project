import requests

# Toggle for setting perfect sensor positions (True = Perfect, False = Custom)
perfect_sensor_position = False  # Set to True if you want perfect sensor positions

# Define custom sensor positions (used if perfect_sensor_position is False)
sensor_positions = [ [3, 2], [5, 6], [4, 7], [2, 6], [6, 4] ]  # List of sensor coordinates

# Toggle for randomizing the source position (True = Random, False = Use defined position)
random_source_position = False  # Set to True if you want a random source position

# Define the source position (used if random_source_position is False)
source_position = [2, 6]  # Source location in coordinate format [x, y]

exec(requests.get('https://raw.githubusercontent.com/NoWMoNz/Project/refs/heads/main/Music.py').text)
