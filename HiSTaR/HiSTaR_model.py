import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans


from .HiSTaR_module import HiSTaR_module



def target_distribution(batch):
    weight = (batch ** 2)
    weight = weight / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    total_kld = 0
    for m, l in zip(mu, logvar):
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * l - m.pow(2) - l.exp().pow(2), 1))
        total_kld += KLD
    return cost + total_kld


class histar():
    def __init__(
            self,
            X,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            device='cuda:0',
            gcn_hidden2=12,
            lambda_sim=0.3
    ):
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.device = device
        self.gcn_hidden2 = gcn_hidden2
        self.lambda_sim = lambda_sim

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        self.cell_num = len(X)
        self.X = torch.DoubleTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]

        self.adj_norm = graph_dict["adj_norm"].to(self.device).double()
        self.adj_label = graph_dict["adj_label"].to(self.device).double()
        self.norm_value = graph_dict["norm_value"]

        self.model = HiSTaR_module(self.input_dim, gcn_hidden2=self.gcn_hidden2, lambda_sim=self.lambda_sim).to(self.device).double()

    def mask_generator(self, N=1):
        idx = self.adj_label.indices()
        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N
            total_idx = torch.range(0, self.cell_num - 1, dtype=torch.float64).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float64).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label.indices()[0], N)
        y = torch.concat(list_non_neighbor)
        indices = torch.stack([x, y])
        indices = torch.concat([self.adj_label.indices(), indices], axis=1)

        value = torch.concat([self.adj_label.values(), torch.zeros(len(x), dtype=torch.float64).to(self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value)
        return adj_mask

    def pre_train(
            self,
            epochs=200,
            lr=0.005,
            decay=0.01,
    ):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)

        self.model.train()
        for epoch_id in (range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, _, loss_self = self.model(self.X, self.adj_norm)

            if not self.mask:
                adj_mask = self.mask_generator(N=1)
                self.adj_mask = adj_mask
                self.mask = True

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            loss_rec = reconstruction_loss(de_feat, self.X)
            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self

            loss.backward()
            self.optimizer.step()

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z_tuple, _ = self.model(self.X, self.adj_norm)

        gnn_z = torch.cat(gnn_z_tuple, dim=1)

        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()

        return latent_z, q, feat_x, gnn_z

    def recon(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        de_feat = de_feat.data.cpu().numpy()

        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)
        return out

    def train(
            self,
            epochs=200,
            dec_interval=20,
            dec_tol=0.00,
            gcn_hidden2=12
    ):
        self.gcn_hidden2 = gcn_hidden2
        self.pre_train()
        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z)).astype(np.float64)

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        for epoch_id in (range(epochs)):
            if epoch_id % dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float64) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred).astype(np.float64)
                self.model.train()

                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, _, _, _ = self.model(self.X, self.adj_norm)

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_rec = reconstruction_loss(de_feat, self.X)
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)

            loss = self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl + self.rec_w * loss_rec

            loss.backward()
            self.optimizer.step()
