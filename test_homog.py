import pickle

with open("cache/homogs/H_map1_20260427_201937.pkl", "rb") as f:
    h_light = pickle.load(f)
print(h_light)

with open("cache/homogs/H_map1_20260427_202441.pkl", "rb") as f:
    h_roma = pickle.load(f)
print(h_roma)