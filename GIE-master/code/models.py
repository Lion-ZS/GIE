from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from hyperbolic import expmap0, project,logmap0,expmap1,logmap1
from tqdm import tqdm
from euclidean import givens_rotations, givens_reflection
class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

        
class GIE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(GIE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size=0.001
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        self.act = nn.Softmax(dim=1)

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings1 = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings1[0].weight.data *= init_size
        self.embeddings1[1].weight.data *= init_size
        self.multi_c=1;self.data_type=torch.float32
        
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)   
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        rel1 = self.embeddings1[0](x[:, 1])
        rel2 = self.embeddings1[1](x[:, 1])
        entities = self.embeddings[0].weight
        entity1 = entities[:, :self.rank]
        entity2= entities[:, self.rank:]
        lhs_t = lhs[:, :self.rank]  , lhs[:, self.rank:]
        rel = rel[:, :self.rank] , rel[:, self.rank:]
        rhs = rhs[:, :self.rank] ,rhs[:, self.rank:]
        rel1 = rel1[:, :self.rank]
        rel2 = rel2[:, :self.rank]
        lhs=lhs_t[0]
        c1 = F.softplus(self.c1[x[:, 1]])
        head1 = expmap0(lhs, c1)
        rel11 = expmap0(rel1, c1)
        lhs = head1
        res_c1 =logmap0(givens_rotations(rel2, lhs),c1)  
        translation1=lhs_t[1] * rel[1]
        c2 = F.softplus(self.c2[x[:, 1]])
        head2 = expmap1(lhs, c2)  
        rel12 = expmap1(rel1, c2) 
        lhss = head2
        res_c2 = logmap1(givens_rotations(rel2, lhss),c2)    
        translation2=lhs_t[1] * rel[0]
        c = F.softplus(self.c[x[:, 1]])
        head = lhs
        rot_q = givens_rotations(rel2, head).view((-1, 1, self.rank))
        cands = torch.cat([res_c1.view(-1, 1, self.rank),res_c2.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(x[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        
        return (
                       (att_q * rel[0] - translation1) @ entity1.t() +(att_q * rel[1] + translation2) @ entity2.t()
               ), [
                   (torch.sqrt(lhs_t[0] ** 2 + lhs_t[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
