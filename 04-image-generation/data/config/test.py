import os

expected_path = os.path.join("..","..", "data", "config")
print("Arquivos em", expected_path)
print(os.listdir(expected_path))
