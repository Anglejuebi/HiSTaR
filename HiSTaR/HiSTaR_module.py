import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features, dtype=torch.float64),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001, dtype=torch.float64),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class MultiHopGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_hops=2, dropout=0.1, act=F.relu):
        super(MultiHopGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hops = num_hops
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.empty(in_features, out_features, dtype=torch.float64))

        self.hop_weights = Parameter(torch.empty(num_hops, dtype=torch.float64))

        if in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features)
        else:
            self.res_proj = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        init_logits = torch.tensor([1.2, 0.0], dtype=torch.float64)
        self.hop_weights.data = init_logits
        if isinstance(self.res_proj, nn.Linear):
            torch.nn.init.xavier_uniform_(self.res_proj.weight)


    def forward(self, input, adj):
        residual = self.res_proj(input)

        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)

        adj_powers = [adj]
        for _ in range(1, self.num_hops):
            adj_powers.append(torch.spmm(adj_powers[-1], adj))

        hop_weights = F.softmax(self.hop_weights, dim=0)

        output = 0
        for i in range(self.num_hops):
            output += hop_weights[i] * torch.spmm(adj_powers[i], support)

        output += residual
        return self.act(output)

# GCN Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0.1, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):

        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class HiSTaR_Block(nn.Module):
    def __init__(self, in_features, out_features, gcn_out_features, num_hops=2, dropout=0.1):
        super(HiSTaR_Block, self).__init__()

        self.multi_hop_gcn = MultiHopGraphConvolution(
            in_features=in_features,
            out_features=out_features,
            num_hops=num_hops,
            dropout=dropout,
            act=F.relu
        )

        self.gcn_mu = GraphConvolution(
            in_features=out_features,
            out_features=gcn_out_features,
            dropout=dropout,
            act=lambda x: x
        )

        self.gcn_logvar = GraphConvolution(
            in_features=out_features,
            out_features=gcn_out_features,
            dropout=dropout,
            act=lambda x: x
        )

    def forward(self, x, adj):
        hidden = self.multi_hop_gcn(x, adj)
        mu = self.gcn_mu(hidden, adj)
        logvar = self.gcn_logvar(hidden, adj)
        return mu, logvar, hidden


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))
        return result


class HiSTaR_module(nn.Module):
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
            lambda_sim=0.3
    ):
        super(HiSTaR_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.lambda_sim = lambda_sim

        self.latent_dim = self.feat_hidden2 + self.gcn_hidden2 * 2

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.block1 = HiSTaR_Block(in_features=self.feat_hidden2, out_features=self.gcn_hidden1, gcn_out_features=self.gcn_hidden2, num_hops=2, dropout=self.p_drop)
        self.block2 = HiSTaR_Block(in_features=self.gcn_hidden2, out_features=self.gcn_hidden1, gcn_out_features=self.gcn_hidden2, num_hops=2, dropout=self.p_drop)

        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        self.cluster_layer = Parameter(torch.empty(self.dec_cluster_n, self.latent_dim, dtype=torch.float64))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim, dtype=torch.float64))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn()

    def setup_loss_fn(self):
        criterion = partial(sce_loss, alpha=3)
        return criterion

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        mu1, logvar1, hidden1 = self.block1(feat_x, adj)
        z1 = self.reparameterize(mu1, logvar1)
        mu2, logvar2, hidden2 = self.block2(z1, adj)
        return (mu1, mu2), (logvar1, logvar2), feat_x

    def reparameterize(self, mu, logvar):
        if isinstance(mu, tuple):
            return tuple(self._reparam(m, l) for m, l in zip(mu, logvar))
        return self._reparam(mu, logvar)

    def _reparam(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def forward(self, x, adj):
        x = x.to(torch.float64)
        adj, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self._mask_rate)

        (mu1, mu2), (logvar1, logvar2), feat_x = self.encode(x, adj)

        gnn_z1 = self.reparameterize(mu1, logvar1)
        gnn_z2 = self.reparameterize(mu2, logvar2)

        similarity_loss = -F.cosine_similarity(gnn_z1, gnn_z2, dim=1).mean()

        z = torch.cat((feat_x, gnn_z1, gnn_z2), 1)

        de_feat = self.decoder(z, adj)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        lambda_hop = 0.15
        target_dist = torch.tensor([0.7, 0.3], device=device)

        hop_kl_loss = 0
        for module in self.modules():
            if isinstance(module, MultiHopGraphConvolution):
                curr_probs = F.softmax(module.hop_weights, dim=0)
                kl_loss = F.kl_div(
                    curr_probs.log(),
                    target_dist,
                    reduction='sum'
                )
                hop_kl_loss += kl_loss

        loss_hop = lambda_hop * hop_kl_loss

        recon = de_feat.clone()
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        recon_loss = self.criterion(x_rec, x_init)

        lambda_sim = self.lambda_sim
        loss = recon_loss + lambda_sim * similarity_loss + loss_hop

        return z, (mu1, mu2), (logvar1, logvar2), de_feat, q, feat_x, (gnn_z1, gnn_z2), loss

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()
        return use_adj, out_x, (mask_nodes, keep_nodes)