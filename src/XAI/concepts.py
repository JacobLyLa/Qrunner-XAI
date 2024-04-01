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
 
	Concept(name="total wall area", binary=False,
	concept_function=lambda state_variables:
	state_variables['total wall area']),
 
	Concept(name="player on grass", binary=True,
	concept_function=lambda state_variables:
	state_variables['player on grass']),

	Concept(name="player in air", binary=True,
	concept_function=lambda state_variables:
	state_variables['player falling']),
 
 	Concept(name="player dodging", binary=True,
	concept_function=lambda state_variables:
	state_variables['player dodging']),

	Concept(name="player on wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['player on wall']),

	Concept(name="player dodging on wall", binary=True,
	concept_function=lambda state_variables:
        state_variables['player on wall']
	and state_variables['player dodging']),

	Concept(name="events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible events']),

	Concept(name="good events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible good events']),

	Concept(name="bad events quantity", binary=False,
	concept_function=lambda state_variables:
	state_variables['visible bad events']),
 
	Concept(name="visible wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible walls'] > 0),
 
	Concept(name="visible air wall", binary=True,
	concept_function=lambda state_variables:
	state_variables['visible air walls'] > 0),

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
    
	Concept(name="bullet close", binary=False,
	concept_function=lambda state_variables:
	state_variables['bullet close']),

	Concept(name="bullet aligned with player", binary=True,
	concept_function=lambda state_variables:
	state_variables['bullet aligned with player']),
 
	Concept(name="good coin left of player", binary=True,
	concept_function=lambda state_variables:
	state_variables['good coin left of player']),
]

concept_instances = {concept.name: concept for concept in concept_instances}
