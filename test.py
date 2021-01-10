# importing requests
import requests
# Defining a BASE request
Base = "http://127.0.0.1:5000/"
# Getting the input review from the user in the command line
name = str(input("Enter your review: "))
# Sending a get request to the specific URL
response = requests.post(Base+"/helloworld/"+name)
# Getting a JSON response
print(response.json())