import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from torch.utils.data import DataLoader, TensorDataset



class Evreg(nn.Module):
    
    def __init__(self, input_dim, prototype_dim):
        super(Evreg, self).__init__()
        self.input_dim = input_dim
        self.prototype_dim = prototype_dim

        self.alpha = Parameter(torch.Tensor(1, self.prototype_dim))
        self.beta = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))
        self.sig=Parameter(torch.Tensor(1, self.prototype_dim))
        self.eta = Parameter(torch.Tensor(1, self.prototype_dim))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.w = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))

    def reset_parameters(self, prototype):
        self.alpha = Parameter(prototype['alpha'])
        self.beta = Parameter(prototype['Beta'])
        self.sig = Parameter(prototype['sig'])
        self.eta = Parameter(prototype['eta'])
        self.gamma = Parameter(prototype['gam'])
        self.w = Parameter(prototype['W'])


    def ENNreg_init_kmeans(self,X, y, K, nstart=100, c=1):
        p = X.shape[1]

        clus = KMeans(n_clusters=K, max_iter=5000, n_init=nstart,random_state=0).fit(X)

        Beta = torch.zeros(K, p, dtype = torch.float64)
        alpha = torch.zeros(K, dtype = torch.float64)
        sig = torch.ones(K, dtype = torch.float64)
        W = torch.tensor(clus.cluster_centers_, dtype = torch.float64)
        gam = torch.ones(K, dtype = torch.float64)

        for k in range(K):
            mask = torch.eq(torch.tensor(clus.labels_), k)
            ii = torch.nonzero(mask, as_tuple=True)[0]
            nk = len(ii)
            alpha[k] = torch.mean(y[ii])

            if nk > 1:
                gam[k] = 1 / torch.sqrt(torch.tensor(clus.inertia_) / nk)
                sig[k] = torch.std(y[ii])

        gam *= c
        eta = torch.ones(K) * 2

        init = {'alpha': alpha, 'Beta': Beta, 'sig': sig, 'eta': eta, 'gam': gam, 'W': W}
        return init

  
    
    def forward(self, input):
        assert torch.is_tensor(input)
        nt,p = input.size()
        h = self.eta ** 2

        a = torch.zeros(nt, self.prototype_dim)
        for k in range(self.prototype_dim):
            a[:, k] = torch.exp(-self.gamma[k] ** 2 * torch.sum((input - self.w[k, :].unsqueeze(0).expand(nt, -1)) ** 2, dim=1))


        H = h.expand(nt, -1)
        hx = torch.sum(a * H, dim=1)
        hx = torch.clamp(hx, min=1e-8)
        mu = torch.mm(input, self.beta.T) + self.alpha.expand(nt, -1)
        mux = torch.sum(mu * a * H, dim=1) / hx
        sig2x = torch.sum((self.sig ** 2).expand(nt, -1) * (a ** 2) * (H ** 2), dim=1) / hx ** 2

        return {"mux": mux, "sig2x": sig2x, "hx": hx}
    
    def loss(self, X, y, events, lambd, nu = 1e-16, xi = 0, rho = 0):
        eta = self.eta
        gam = self.gamma
        h = eta ** 2

        pred = self.forward(X)
        mux = pred['mux']
        sig2x = pred['sig2x']
        sig2x = torch.clamp(sig2x, min=1e-8)
        sigx = torch.sqrt(sig2x)

        hx = pred['hx']

        Z2 = hx * sig2x + 1
        Z = torch.sqrt(Z2)
        sig1 = sigx * Z
      
        pl = 1 / Z * torch.exp(-0.5 * hx * (y - mux) ** 2 / Z2)


        # Bel
        eps = 1e-4 * torch.std(y)
        norm_dist = Normal(mux, sigx)
        Fy1 = norm_dist.cdf(y) - pl * Normal(mux, sig1).cdf(y)
        Sy1 = 1 - Fy1

        pl1 = 1 / Z * torch.exp(-0.5 * hx * (y - eps - mux) ** 2 / Z2)
        pl2 = 1 / Z * torch.exp(-0.5 * hx * (y + eps - mux) ** 2 / Z2)

        # Pl
        Fy2 = Fy1 + pl
        Sy2 = 1 - Fy2
        Fy2_1 = norm_dist.cdf(y + eps) + pl1 * Normal(mux, sig1).cdf(y - eps)
        Fy2_2 = norm_dist.cdf(y - eps) - pl2 * (1 - Normal(mux, sig1).cdf(y + eps))

        fy2 = Fy2_1 - Fy2_2
        fy1 = fy2 - pl1 * Normal(mux, sig1).cdf(y) - pl2 * (1 - Normal(mux, sig1).cdf(y))

        Sy1 = torch.clamp(Sy1, min=0.0)
        Sy2 = torch.clamp(Sy2, min=0.0)

        fy1 = torch.clamp(fy1, min=0.0)
        fy2 = torch.clamp(fy2, min=0.0)


        loss = -lambd * torch.mean(torch.log(fy1 + nu) * events + torch.log(Sy1 + nu) * (1 - events)) \
                - (1 - lambd) * torch.mean(torch.log(fy2 + nu) * events + torch.log(Sy2 + nu) * (1 - events)) \
                + xi * torch.mean(h) + rho * torch.mean(gam ** 2)

        return loss


    
    def fit(self, X_train, y_train, event_train,
            X_val, y_val, event_val,
            train_lambd = 0.5, batch_size = 512, epochs=500):

        train_dataset = TensorDataset(X_train, torch.log(y_train), event_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, torch.log(y_val), event_val)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

        prototype = self.ENNreg_init_kmeans(X_train, torch.log(y_train), K=self.prototype_dim)
        self.reset_parameters(prototype)

        optimizer = torch.optim.Adam(self.parameters(), 0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300], gamma=0.1)
        best_val_loss = 1000
        patience = 5

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for batch_idx, (inputs, log_time_survival, events) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss(inputs, log_time_survival, events=events, nu=1e-16, xi=0, rho=0, lambd=train_lambd)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            average_loss = total_loss / len(train_loader)
            print(f"epoch {epoch + 1} average loss: {average_loss:.4f}")
            self.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_idx, (val_inputs, log_durations_val, events_val) in enumerate(val_loader):
                    val_loss += self.loss(val_inputs, log_durations_val, events=events_val, nu=1e-16, xi=0, rho=0, lambd=train_lambd)
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), 'ev.pth')

                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
        self.load_state_dict(torch.load('ev.pth', weights_only=True))
        print(best_val_loss)
    
    def evreg_evaluation(self, X_test, y_test, event_test, weight):
        pred = self.forward(X_test)
        mux = pred['mux']
        sigx = torch.sqrt(pred['sig2x'])
        hx = pred['hx']
        Z2 = hx * pred['sig2x'] + 1
        Z = torch.sqrt(Z2)
        sig1 = sigx * Z

        time_grid = np.linspace(y_test.numpy().min(), y_test.numpy().max(), 100)
        D, M = torch.meshgrid(torch.log(y_test), mux)
        diff = (D - M)

        pl = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
        Fy1 = torch.distributions.Normal(mux, sigx).cdf(D) - pl * torch.distributions.Normal(mux, sig1).cdf(D)
        Fy2 = Fy1 + pl

        surv_df = 1 - (weight * Fy1 + (1-weight) * Fy2)
        surv_df = pd.DataFrame(surv_df.detach().numpy(), index=y_test.numpy())

        ev = EvalSurv(surv_df, y_test.numpy(), event_test, censor_surv='km')
        
        c_index = ev.concordance_td()
        _ = ev.brier_score(time_grid).plot()
        IBS = ev.integrated_brier_score(time_grid)
        NBLL = ev.integrated_nbll(time_grid)

        return c_index,IBS,NBLL
