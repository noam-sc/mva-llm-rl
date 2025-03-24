def Glam_prompt(info):
    prompt = ""
    prompt += "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "Goal: {}\n".format(info["goal"])
    prompt += "Observation: {}\n".format(', '.join(info["obs"]))
    # if info["last_action"]!=None:
    #    prompt += "Last action of the agent: {}\n".format(info["last_action"])
    # else:
    #    prompt += "Last action of the agent: Nothing \n"
    if info["inventory"] == None:
        prompt += "Inventory: Nothing\n"
    else:
        prompt += "Inventory: {}\n".format(info["inventory"])
    prompt += "Next action of the agent: "
    return prompt


def swap_prompt(info):
    prompt = ""
    prompt += "Goal: {}\n".format(info["goal"])
    if info["inventory"] == None:
        prompt += "Inventory: Nothing\n"
    else:
        prompt += "Inventory: {}\n".format(info["inventory"])
    prompt += "Observation: {}\n".format(', '.join(info["obs"]))
    # if info["last_action"]!=None:
    #    prompt += "Last action of the agent: {}\n".format(info["last_action"])
    # else:
    #    prompt += "Last action of the agent: Nothing \n"
    prompt += "Possible actions of the agent: {}\n".format(info["possible_actions"])
    prompt += "Next action of the agent: "
    return prompt


def xml_prompt(info):
    prompt = ""
    prompt += "<Begin Possible actions>\n {} \n<End Possible actions>\n".format(", ".join(info["possible_actions"]))
    prompt += "<Begin Goal>\n {} \n<End Goal>\n".format(info["goal"])
    prompt += "<Begin Observation>\n {} \n<End Observation>\n".format(', '.join(info["obs"]))
    # if info["last_action"]!=None:
    #    prompt += "<Begin Last action>\n {} \n<End Last action>\n".format(info["last_action"])
    # else:
    #    prompt += "<Begin Last action>\n nothing \n<End Last action>\n"
    if info["inventory"] != None:
        prompt += "<Begin Inventory>\n {} \n<End Inventory>\n".format(info["inventory"])
    else:
        prompt += "<Begin Inventory>\n Empty \n<End Inventory>\n"
    prompt += "Next action :"
    return prompt


def paraphrase_prompt(info):
    prompt = "Welcome to TextWorld! You find yourself in a messy house. Many things are not in their usual location. Let's clean up this place. After you'll have done, this little house is going to be spick and span! Look for anything that is out of place and put it away in its proper location. "
    prompt += "What you can do is to  {}. ".format(", ".join(info["possible_actions"]))
    prompt += "Your goal is to {}. ".format(info["goal"])
    prompt += "{}. ".format(', '.join(info["obs"]))
    # if info["last_action"]!=None:
    #    prompt += "Your past action was to {}. ".format(info["last_action"])
    # else:
    #    prompt += "You haven't done any action before. "
    if info["inventory"] == None:
        prompt += "Now your carrying nothing . "
    else:
        prompt += "Now, {}, ".format(info["inventory"])
    prompt += "and your next action  is to "
    return prompt