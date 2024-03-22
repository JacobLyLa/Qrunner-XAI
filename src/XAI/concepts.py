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

	Concept(name="player low", binary=False,
	concept_function=lambda state_variables:
	state_variables['player y position']),

	Concept(name="player in air", binary=True,
	concept_function=lambda state_variables:
	state_variables['player falling']),
 
 	Concept(name="player dodging", binary=True,
	concept_function=lambda state_variables:
	state_variables['player dodging']),

	Concept(name="player dodging in air", binary=True,
	concept_function=lambda state_variables:
	state_variables['player falling'] and state_variables['player dodging']),

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
 
  	Concept(name="reachable good coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['reachable good coin']),
  
   	Concept(name="unreachable good coin", binary=True,
	concept_function=lambda state_variables:
	state_variables['unreachable good coin']),
    
	Concept(name="bullet close", binary=False,
	concept_function=lambda state_variables:
	state_variables['bullet close']),

	Concept(name="bullet aligned with player", binary=True,
	concept_function=lambda state_variables:
	state_variables['bullet aligned with player']),

	Concept(name="bullet below player", binary=True,
	concept_function=lambda state_variables:
	state_variables['bullet below player']),
 
	Concept(name="lava below player", binary=True,
	concept_function=lambda state_variables:
	state_variables['lava below player']),
 
	Concept(name="good coin below player", binary=True,
    concept_function=lambda state_variables:
    state_variables['good coin below player']),	
 
	Concept(name="good coin left of player", binary=True,
	concept_function=lambda state_variables:
	state_variables['good coin left of player']),

	Concept(name="bullet left of player", binary=True,
	concept_function=lambda state_variables:
	state_variables['bullet left of player']),
]

concept_instances = {concept.name: concept for concept in concept_instances}
