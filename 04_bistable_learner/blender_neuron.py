class Neuron():
    radius = .75
    c_to_c_dist = 2.25 # center to center distance
    strength_increment = 2. 
    strength_decay = .9
    default_color = (.75, .75, .75, 1.)

    def __init__(self, position=(0,0,0), color=default_color):
        self.position = position
        self.color = color
        self.strength = 2. #TODO: just for debug
