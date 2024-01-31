def sgld_params2dict(client_sgld_params):
    global params
    for param in client_sgld_params:
        params = param.state_dict()
    param_dict = dict((key, value) for key, value in params.items())
    return param_dict
