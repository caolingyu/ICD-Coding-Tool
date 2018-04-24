# coding=utf-8

import requests
import json

# Set server address
addr = 'http://localhost:5000'
test_url = addr + '/api/test'
# Set post header
content_type = 'application/json'
headers = {'content-type': content_type}

text = "生儿高胆红素血症"

# wrap them into json
json_f1 = json.dumps({'data': text})
# post request
response1 = requests.post(test_url, json=json_f1, headers=headers)
print(response1.text)