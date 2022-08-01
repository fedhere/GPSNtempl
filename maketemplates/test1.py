import pickle as pkl


path = "outputs/all_params_scipy_opt.pkl"
tmpl = pkl.load(open(path, "rb"))

print(tmpl)