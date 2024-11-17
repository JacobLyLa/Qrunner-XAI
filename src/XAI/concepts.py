import math
import random
import numpy as np

from src.XAI.concept import Concept

concept_instances = [
	Concept(name="random binary", binary=True,
	concept_function=lambda state_variables:
	random.random() > 0.5),
	
	Concept(name="random continuous", binary=False,
	concept_function=lambda state_variables:
	np.random.normal(0, 1)),
 
	Concept(name="player on grass", binary=True,
	concept_function=lambda state_variables:
	state_variables['player on grass']),

	Concept(name="player on wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['player on wall']),
 
	Concept(name="player in air", binary=True,
	concept_function=lambda state_variables:
	state_variables['player falling']),
 
 	Concept(name="player dodging", binary=True,
	concept_function=lambda state_variables:
	state_variables['player dodging']),

	Concept(name="player dodging on wall", binary=True,
	concept_function=lambda state_variables:
        state_variables['player on wall']
	and state_variables['player dodging']),
 
 	Concept(name="bullet aligned with player", binary=True,
	concept_function=lambda state_variables:
	state_variables['bullet aligned with player']),
 
	Concept(name="good coin left of player", binary=True,
	concept_function=lambda state_variables:
	state_variables['good coin left of player']),
 
	Concept(name="total wall area", binary=False,
	concept_function=lambda state_variables:
	state_variables['total wall area']),
 
	Concept(name="events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible events']),

	Concept(name="good events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible good events']),

	Concept(name="bad events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible bad events']),
 
	Concept(name="visible air wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible air walls'] > 0),
 
	Concept(name="visible wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible walls'] > 0),
 
	Concept(name="visible bullet", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible bullets'] > 0),
 
	Concept(name="visible lava", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible lava'] > 0),

	Concept(name="visible blue coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible blue coins'] > 0),
 
	Concept(name="visible gold coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible gold coins'] > 0),
 
	Concept(name="visible red coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible red coins'] > 0),
 
  	Concept(name="visible high coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible high coin']),
  
   	Concept(name="visible good low coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible good low coin']),
    
    Concept(name="visible ghost", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible ghost'])]

concept_instances = {concept.name: concept for concept in concept_instances}

# Some modifcations for plotting but avoiding to recalcuate stuff:
# Rename keys with 'events' in name to 'event'
for key in list(concept_instances.keys()):
    if 'events' in key:
        concept_instances[key.replace('events', 'event')] = concept_instances.pop(key)
        
# First letter in key to uppercase
concept_instances = {key.capitalize(): value for key, value in concept_instances.items()}
# Set plot name to be the same as the key
for key in concept_instances.keys():
    concept_instances[key].plot_name = key

# Rewrite plot_names manually for better readability
concept_instances['Bullet aligned with player'].plot_name = 'Player bullet aligned'
concept_instances['Good coin left of player'].plot_name = 'Player right good coin'
