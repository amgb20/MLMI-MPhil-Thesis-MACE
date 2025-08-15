import pickle

data = pickle.load(open('Experiments/numerical_stability/pickle/dump_snapshot.pickle', 'rb'))

print(type(data))

print(data.keys() if hasattr(data, 'keys') else data)