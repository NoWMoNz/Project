import requests

perfect_sensor_position = False  # Set to True if you want perfect sensor positions
sensor_positions = [ [3, 2], [5, 6], [4, 7], [2, 6], [6, 4] ]  # List of sensor coordinates
random_source_position = False  # Set to True if you want a random source position
source_position = [2, 6]  # Source location in coordinate format [x, y]

exec(requests.get('https://raw.githubusercontent.com/NoWMoNz/Project/refs/heads/main/Music.py').text)
