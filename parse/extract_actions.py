from pysc2.lib.actions import FUNCTIONS
from pysc2.lib import features


def extract_actions(obs):

    agent_intf = features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=(1,1), minimap=(1,1)))
    feat = features.Features(agent_intf)

    actions = []

    for action in obs.actions:
        try:
            func_id = feat.reverse_action(action).function
            func_name = FUNCTIONS[func_id].name
            if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'TrainWarp'}:
                actions.append({func_id: func_name})    
        except:
            pass

    return actions