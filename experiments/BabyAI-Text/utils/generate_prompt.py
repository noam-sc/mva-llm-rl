def Glam_prompt(past_transitions, obs, info):
    prompt = "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    prompt += "{}\n".format(info["goal"])
    for transition in past_transitions:
        prompt += "Past Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Past Action:{}\n".format(transition["act"])

    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt




def paraphrase_prompt(past_transitions, obs, info):
    prompt = "you are on a maze and you have to solve a task, "
    prompt += "what you can do is : {}\n".format(", ".join(info["possible_actions"]))
    prompt += "your task is to {}\n".format(info["goal"].split(":")[-1])
    for transition in past_transitions:
        prompt += "in the past you had seen this: {}\n".format(', '.join(transition["obs"]))
        prompt += "and your action was :{}\n".format(transition["act"])
    prompt += "what you see now: {}\n".format(', '.join(obs))
    prompt += "you next action is to "
    return prompt




def swap_prompt(past_transitions, obs, info):
    prompt = "{}\n".format(info["goal"])
    prompt += "Possible actions of the agent: {}\n".format(", ".join(info["possible_actions"]))
    for transition in past_transitions:
        prompt += "Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Action:{}\n".format(transition["act"])
    prompt += "Observation: {}\n".format(', '.join(obs))
    prompt += "Action:"
    return prompt


def xml_prompt(past_transitions, obs, info):
    prompt = "<Begin Possible actions>{}<End Possible actions>\n".format(", ".join(info["possible_actions"]))
    prompt += "<Begin Goal>{}<End Goal>\n".format(info["goal"].split(":")[-1])
    prompt += "<Begin past Observation>"
    for transition in past_transitions:
        prompt += "Observation: {}\n".format(', '.join(transition["obs"]))
        prompt += "Actions :{}\n".format(transition["act"])
    prompt += "<End past Observation>\n"
    prompt += "<Begin Current Observation>\n"

    prompt += "Observation: {}".format(', '.join(obs))
    prompt += "<End Current Observation>\n"
    prompt += "next action:"
    return prompt