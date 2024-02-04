import math
import random

from src.XAI.concept import Concept

concept_instances = [
    Concept(name="player to right", binary=False,
            concept_function=lambda state_variables:
            state_variables['player x position'] - state_variables['camera x offset']),
    
    Concept(name="player low", binary=False,
            concept_function=lambda state_variables:
            state_variables['player y position']),
    
    Concept(name="difficulty", binary=False,
            concept_function=lambda state_variables:
            state_variables['camera x offset']),
    
    Concept(name="player in air", binary=True,
            concept_function=lambda state_variables:
            state_variables['player jumping'] or state_variables['player falling']),
    
        Concept(name="player falling", binary=True,
            concept_function=lambda state_variables:
            state_variables['player falling']),
    
    Concept(name="player dodging in air", binary=True,
            concept_function=lambda state_variables:
            (state_variables['player jumping'] or state_variables['player falling'])
            and state_variables['player dodging']),
    
    Concept(name="player fall velocity", binary=False,
            concept_function=lambda state_variables:
            state_variables['player fall velocity']),
    
    Concept(name="player standing on wall", binary=True,
            concept_function=lambda state_variables:
            state_variables['player standing on wall']),
    
     Concept(name="player dodging on wall", binary=True,
            concept_function=lambda state_variables:
            state_variables['player standing on wall']
            and state_variables['player dodging']),
     
     Concept(name="events quantity", binary=False,
            concept_function=lambda state_variables:
            state_variables['visible events']),
     
     Concept(name="visible wall", binary=True,
                concept_function=lambda state_variables:
                state_variables['visible walls'] > 0),
     
     Concept(name="visible bullet", binary=True,
                concept_function=lambda state_variables:
                state_variables['visible bullets'] > 0),
     
     Concept(name="lava quantity", binary=False,
                concept_function=lambda state_variables:
                state_variables['visible lava']),
        
        Concept(name="coin quantity", binary=False,
                concept_function=lambda state_variables:
                state_variables['visible coins']),
        
        Concept(name="good visible events", binary=False,
                concept_function=lambda state_variables:
                state_variables['good visible events']),
        
        Concept(name="bad visible events", binary=False,
                concept_function=lambda state_variables:
                state_variables['bad visible events']),
        
        Concept(name="two close bullets", binary=True,
                concept_function=lambda state_variables:
                state_variables['two close bullets']),
        
        Concept(name="bullet player distance", binary=False,
                concept_function=lambda state_variables:
                state_variables['bullet player distance']),
        
        Concept(name="bullet aligned with player", binary=True,
                concept_function=lambda state_variables:
                state_variables['bullet aligned with player']),
        
        Concept(name="player dodging", binary=True,
                concept_function=lambda state_variables:
                state_variables['player dodging']),
]
concept_instances = {concept.name: concept for concept in concept_instances}
