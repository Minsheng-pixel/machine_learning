import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
import torch
import numpy as np
import seaborn as sns

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def plot_network(model, vals=None, newcmp=None, vmin=-1, vmax=10):
    norm = Normalize(vmin=vmin, vmax=vmax)
    if newcmp is None:
        newcmp = cm.get_cmap("Blues")
    if vals is None:
        npar = sum([p.numel() for p in model.parameters()])
        vals = torch.ones(npar)
    vals = unflatten_like(vals.unsqueeze(0).detach(), model.parameters())
    vspc = 0.1
    hspc = 1.
    n_lyr = len(model)
    
    mrk = 'o'
    mrk_col = 'k'
    mrk_size = 100
    plt.figure(figsize=(10, 6))
    weight_ind = 0
    line_alpha = 0.9
    has_bias = False
    for ind, lyr in enumerate(model):
        if isinstance(lyr, torch.nn.Linear):

            xs = np.ones(lyr.in_features)*ind * hspc
            ys = np.arange(0, lyr.in_features)*vspc
            ys = ys - ys.mean()
            
            ## plot previous layer to this layer ##
#             print(vals[weight_ind].shape)
            if ind != 0:
#                 print("plotting line 46")
#                 print("color shape = " vals[weight_ind].shape)
                for i1, (x1, y1) in enumerate(zip(old_xs, old_ys)):
                    for i2, (x2, y2) in enumerate(zip(xs, ys)):
                        plt.plot([x1, x2], [y1, y2],
                                     color = newcmp(norm(vals[weight_ind][i2, i1])),
                                    alpha=line_alpha)
                weight_ind += 1
                
            ## plot previous layers bias to this layer ##
            if lyr != 0:
                if has_bias:
                    for i2, (x2, y2) in enumerate(zip(xs, ys)):
                        plt.plot([bias_xs, x2], [bias_ys, y2],
                                 color = newcmp(norm(vals[weight_ind][i2])),
                                  alpha=line_alpha)
                    weight_ind += 1

            ## plot all points for this layer ##
            has_bias = lyr.bias is not None
            if has_bias:
                bias_xs = xs[0]
                bias_ys = ys[-1] + vspc
                plt.scatter(bias_xs, bias_ys, marker=mrk,  color=mrk_col,
                           s=mrk_size)
            plt.scatter(xs, ys, marker=mrk,  color=mrk_col,
                           s=mrk_size)    

            old_xs = xs
            old_ys = ys
            
            if ind == (n_lyr-1):
                xs = np.ones(lyr.out_features)*(ind + 2) * hspc
                ys = np.arange(0, lyr.out_features)*vspc
                plt.scatter(xs, ys, marker=mrk,  color=mrk_col,
                           s=mrk_size)
                
                for i1, (x1, y1) in enumerate(zip(old_xs, old_ys)):
                    for i2, (x2, y2) in enumerate(zip(xs, ys)):
                        plt.plot([x1, x2], [y1, y2],
                                     color = newcmp(norm(vals[weight_ind][i2, i1])),
                                    alpha=line_alpha)
                weight_ind += 1
                if has_bias:
                    for i2, (x2, y2) in enumerate(zip(xs, ys)):
                        plt.plot([bias_xs, x2], [bias_ys, y2],
                                 color = newcmp(norm(vals[weight_ind][i2])),
                                  alpha=line_alpha)
                    weight_ind += 1
    return plt



def plotter(model, train_x, train_y):
    '''
    This is just a simple plotting function, you should NOT need to change anything here
    '''
    buffer = 0.3
    h = 0.1
    x_min, x_max = train_x[:, 0].min() - buffer, train_x[:, 0].max() + buffer
    y_min, y_max = train_x[:, 1].min() - buffer, train_x[:, 1].max() + buffer

    xx,yy=np.meshgrid(np.arange(x_min.cpu(), x_max.cpu(), h), 
                      np.arange(y_min.cpu(), y_max.cpu(), h))
    in_grid = torch.FloatTensor([xx.ravel(), yy.ravel()]).t()

    pred = torch.sigmoid(model(in_grid)).detach().reshape(xx.shape)
    plt.figure(figsize=(15, 10))
    cmap = sns.color_palette("crest_r", as_cmap=True)
    plt.contourf(xx, yy, pred, alpha=0.5, cmap=cmap)
    plt.title("Classifier", fontsize=24)
    cbar= plt.colorbar()
    cbar.set_label(label=r"$P(Y = 1)$", size=18)
    cbar.ax.tick_params(labelsize=18)
    plt.scatter(train_x[:, 0].cpu(), train_x[:, 1].cpu(), c=train_y[:, 0].cpu(), cmap=plt.cm.binary, alpha=0.5, label="Data")
    plt.legend(fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def compute_loss_surface(model, train_x, train_y,
                        loss, n_pts=50, range_=torch.tensor(10.)):
    
    n_par = int(sum([p.numel() for p in model.parameters()]))
    v1, v2 = torch.randn(n_par, 1), torch.randn(n_par, 1)
    
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = loss(model(train_x), train_y)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf

def compute_loader_loss(model, loader, loss, n_batch,
                       device=torch.device("cuda:0")):
    total_loss = torch.tensor([0.])
    for i, data in enumerate(loader):
        if i < n_batch:
            x, y = data
            x, y = x.to(device), y.to(device)

            preds = model(x)
            if isinstance(preds, tuple):
                preds = preds[0]
            total_loss += loss(preds, y).item()
        else:
            break

    return total_loss

def compute_loss_surface_loader(model, loader, v1=None, v2=None,
                                loss=torch.nn.CrossEntropyLoss(),
                                n_batch=10, n_pts=50, range_=torch.tensor(10.),
                               device=torch.device("cuda:0")):
    if v1 is None:
        n_par = int(sum([p.numel() for p in model.parameters()]))
        v1, v2 = torch.randn(n_par, 1), torch.randn(n_par, 1)
    
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)
                    
                loss_surf[ii, jj] = compute_loader_loss(model, loader,
                                                        loss, n_batch,
                                                        device=device)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf, v1, v2
