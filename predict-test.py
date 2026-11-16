import requests

url = 'http://localhost:9696/predict'

stellar_object_id = 'TON-35523d'
stellar_object_data = {
    "redshift": 0.5,
    "u": 19.2,
    "g": 18.1,
    "r": 17.9,
    "i": 17.7,
    "z": 17.5,
    "alpha": 135.689,
    "delta": 32.494,
}

response = requests.post(url, json=stellar_object_data).json()

print(response)

print('%s is a %s' % (stellar_object_id, response['predicted_class']))
