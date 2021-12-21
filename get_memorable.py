import torch
import torch.nn.functional as F
from preds.likelihoods import GaussianLh
from preds.laplace import Laplace

# We only calculate the diagonal elements of the hessian
def logistic_hessian(f):
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi*(1-pi)


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s

# Select memorable points ordered by their lambda values (descending=True picks most important points)
def select_memorable_points(dataloader, model, num_points=10, num_classes=2,
                            use_cuda=False, label_set=None, descending=True):
    memorable_points = {}
    bad_points = {}
    scores = {}
    num_points_per_class = int(num_points/num_classes)
    for i, dt in enumerate(dataloader):
        data, target = dt
        data = torch.unique(data, dim=0)
        if use_cuda:
            data_in = data.cuda()
        else:
            data_in = data
        
        """
        # Here, hessian isn't working! Use variance instead.
        print("Hessian!")
        if label_set == None:
            f = model.forward(data_in)
        else:
            f = model.forward(data_in, label_set)
        if f.shape[-1] > 1:
            lamb = softmax_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.sum(lamb, dim=-1)
            lamb = lamb.detach()
        else:
            lamb = logistic_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.squeeze(lamb, dim=-1)
            lamb = lamb.detach()
        """
        """ 
        # Advantageous 
        f = model.forward(data_in)
        target = f.max(1)[1]
        lamb = torch.var(f,-1)
        if use_cuda:
            lamb = lamb.cpu()
        lamb = lamb.detach()
        
        print("\n\nNAIVE:", lamb)
        """
        
        # Gaussian variance
        f = model.forward(data_in)
        target = f.max(1)[1]
        
        lh = GaussianLh()  # likelihood: GaussianLh for regression, CategoricalLh for classification 
        prior_precision = 1.  # prior
        posterior = Laplace(model, prior_precision, lh)
        # Something here doesn't have a grad_fn?
        posterior.infer([(data_in, f)], cov_type='kron', dampen_kron=False)   
        _, lamb = posterior.predictive_samples_glm(data_in, n_samples=1000)     
        
        print("\n\nBETTER:", lamb)
        print("\n\n")

        
        # Here, let's remove duplicates!
 
        for cid in range(num_classes):
            p_c = data[target == cid]
            if len(p_c) > 0:
                s_c = lamb[target == cid]
                if len(s_c) > 0:
                    if cid not in memorable_points:
                        memorable_points[cid] = p_c
                        scores[cid] = s_c
                    else:
                        memorable_points[cid] = torch.cat([memorable_points[cid], p_c], dim=0)
                        scores[cid] = torch.cat([scores[cid], s_c], dim=0)
                    if len(memorable_points[cid]) > num_points_per_class:
                        _, indices = scores[cid].sort(descending=descending)
                        good = memorable_points[cid][indices[:num_points_per_class]]
                        bad = memorable_points[cid][indices[-num_points_per_class:]]
                        memorable_points[cid] = good
                        scores[cid] = scores[cid][indices[:num_points_per_class]]
                        bad_points[cid] = bad
    r_points = []
    r_labels = []
    b_points = []
    b_labels = []
    for cid in range(num_classes):
        try:
            r_points.append(memorable_points[cid])
            r_labels.append(torch.ones(memorable_points[cid].shape[0], dtype=torch.long,
                                       device=memorable_points[cid].device)*cid)
            b_points.append(bad_points[cid])
            b_labels.append(torch.ones(bad_points[cid].shape[0], dtype=torch.long,
                                       device=memorable_points[cid].device)*cid)
        except:
            pass
    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0), torch.cat(b_points, dim=0), torch.cat(b_labels, dim=0)]