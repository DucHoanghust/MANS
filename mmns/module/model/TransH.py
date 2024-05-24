import torch
import torch.nn as nn

class TransH(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim_e, dim_r, margin, epsilon):
        super(TransH, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.epsilon = epsilon

        # Embedding layers
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.wr = nn.Embedding(self.rel_tot, self.dim_r)  # Vector pháp tuyến cho mỗi quan hệ

        # Parameter initialization ranges
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data, 
            a=-self.ent_embedding_range.item(), 
            b=self.ent_embedding_range.item()
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data, 
            a=-self.rel_embedding_range.item(), 
            b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    def project_to_hyperplane(self, entity, w):
        return entity - torch.sum(entity * w, dim=1, keepdim=True) * w

    def get_pos_embd(self, sample):
        h = self.ent_embeddings(sample[:, 0])
        r = self.rel_embeddings(sample[:, 1])
        t = self.ent_embeddings(sample[:, 2])
        
        w = self.wr(sample[:, 1]).unsqueeze(dim=1)
        
        h_proj = self.project_to_hyperplane(h, w)
        t_proj = self.project_to_hyperplane(t, w)
        
        return h_proj, r, t_proj, w

    def get_neg_embd(self, sample):
        h = self.ent_embeddings(sample[:, 0])
        t = self.ent_embeddings(sample[:, 2])
        return h if sample[:, 1][0] == 0 else t  # Assuming head-batch or tail-batch

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, w = self.get_pos_embd(pos_sample)
        
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            wr_neg = (w * neg_embd).sum(dim=-1, keepdim=True)
            wr_neg_wr = wr_neg * w
            
            if mode == "head-batch":
                wr_t = (w * t).sum(dim=-1, keepdim=True)
                wr_t_wr = wr_t * w
                score = (neg_embd - wr_neg_wr) + (r - (t - wr_t_wr))
            elif mode == "tail-batch":
                wr_h = (w * h).sum(dim=-1, keepdim=True)
                wr_h_wr = wr_h * w
                score = ((h - wr_h_wr) + r) - (neg_embd - wr_neg_wr)
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            wr_h = (w * h).sum(dim=-1, keepdim=True)
            wr_h_wr = wr_h * w
            wr_t = (w * t).sum(dim=-1, keepdim=True)
            wr_t_wr = wr_t * w
            score = (h - wr_h_wr) + r - (t - wr_t_wr)
        
        score = torch.norm(score, dim=-1)
        return score
