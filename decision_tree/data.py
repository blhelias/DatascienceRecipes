from collections import namedtuple

# Fruit = namedtuple('Fruit', ["color", "diametere", "target"])
# fruit = [Fruit("Green", 3, "Apple"), Fruit("Yellow", 3, "Apple"), Fruit("Red", 2, "Grape"), Fruit("Red", 1, "Grape"), Fruit("Yellow", 3, "Lemon")]

Data = namedtuple('Data', ["outlook", "temperature", "humidity", "wind", "target"])

data = [Data("sunny", "hot", "high", "false", "no"), 
        Data("sunny", "hot", "high", "true", "no"),
        Data("overcast", "hot", "high", "false", "yes"),
        Data("rain", "mild", "high", "false", "yes"),
        Data("rain", "cool", "normal", "false", "yes"),
        Data("rain", "cool", "normal", "true", "no"),
        Data("overcast", "cool", "normal", "true", "yes"),
        Data("sunny", "mild", "high", "false", "no"),
        Data("sunny", "cool", "normal", "false", "yes"),
        Data("rain", "mild", "normal", "false", "yes"),
        Data("sunny", "mild", "normal", "true", "yes"),
        Data("overcast", "mild", "high", "true", "yes"),
        Data("overcast", "hot", "normal", "false", "yes"),
        Data("rain", "mild", "high", "true", "no")
        ] 

