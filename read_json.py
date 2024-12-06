import json

# Open the JSON file for reading
with open('benchmark_1_0_instr.json', 'r') as file:
    # Load the JSON data from the file
    data = json.load(file)

# Now `data` is a Python dictionary containing the JSON data
print(data)