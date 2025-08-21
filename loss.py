from torch import nn
import torch
import torch.nn.functional as F
class Regularizer(nn.Module):
    def __init__(self, weight=0.0, T=1000) -> None:
        super().__init__()
        self.weight_T = weight
        self.weight_t = 0
        self.T = T
        self.t = 0

    def step(self):
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.weight_t = self.weight_T * (self.t / self.T) ** 2

    def forward(self, reps):
        raise NotImplementedError("This is an abstract regularizer only.")

class L1(Regularizer):
    def forward(self, reps):
        with torch.cuda.amp.autocast(enabled=False):
            return torch.abs(reps * self.weight_t).sum(dim=1).mean()

class FLOPs(Regularizer):
    def forward(self, reps):
        return (torch.abs(reps).mean(dim=0) ** 2).sum() * self.weight_t

class BaselineLoss(nn.Module):
    def __init__(self, temp=torch.tensor(0.001), q_reg=0.0, d_reg=0.0, T=1000,):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.q_regularizer = L1(q_reg, T)
        self.d_regularizer = L1(d_reg, T)
        self.temp = temp
    def forward(self, dense_vec_texts = None, dense_vec_imgs = None, single_sparse_texts= None, single_sparse_imgs= None, max_sparse_texts= None,  max_sparse_imgs= None, colbert_vec_texts= None, colbert_vec_imgs=None, bow=False, use_all_tokens=False):
        sparse_i2t_scores = single_sparse_imgs @ single_sparse_texts.t()
        sparse_t2i_scores = sparse_i2t_scores.t()
        with torch.no_grad():
            scores_dense_i2t = dense_vec_imgs @ dense_vec_texts.t()
            prob_dense_i2t = torch.softmax(
                scores_dense_i2t/self.temp, dim=1)
            prob_dense_t2i = torch.softmax(
                scores_dense_i2t.t()/self.temp, dim=1)
        loss = (self.ce(sparse_i2t_scores, prob_dense_i2t) +
                self.ce(sparse_t2i_scores, prob_dense_t2i))/2
        reg = (self.q_regularizer(single_sparse_texts) +
               self.d_regularizer(single_sparse_imgs))/2
        self.q_regularizer.step()
        self.d_regularizer.step()
        return loss, reg, 0,0,0,0,0,0,0,0

class BICELoss(nn.Module):
    def __init__(self, temp=torch.tensor(0.001), q_reg=0.0, d_reg=0.0, T=1000, w1=1.0, w2=0.3,
                 lam1=1.0, lam2=0.1, self_supervised = True, hard_negatives = False, pooling = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.q_regularizer = L1(q_reg, T)
        self.d_regularizer = L1(d_reg, T)
        self.w1, self.w2 = w1, w2
        self.lam1, self.lam2 = lam1, lam2
        self.temp = temp
        self.self_supervised = self_supervised
        self.hard_negatives = hard_negatives
        self.pooling = pooling
        # 2) InfoNCE 손실 (양 방향)
    def infoNCE(self, s, idxs):
        i2t = self.ce(s, idxs)
        t2i = self.ce(s.T, idxs)
        return 0.5 * (i2t + t2i)
    # 4) 자기지식증류 손실 (양 방향)
    def distill_loss(self, s, p_teacher):
        log_p = torch.log_softmax(s, dim=1)
        ld = - (p_teacher * log_p).sum(dim=1).mean()
        return ld
    def forward(self, dense_vec_texts = None, dense_vec_imgs = None, single_sparse_texts= None, single_sparse_imgs= None, max_sparse_texts= None,  max_sparse_imgs= None, colbert_vec_texts= None, colbert_vec_imgs=None, bow=False, use_all_tokens=False):
        if dense_vec_imgs is not None:
            B = dense_vec_imgs.size(0)
            device = dense_vec_imgs.device

        elif single_sparse_imgs is not None:
            B = single_sparse_imgs.size(0)
            device = single_sparse_imgs.device

        idxs = torch.arange(B, device=device)
        sdense, slex = torch.tensor(0), torch.tensor(0)
        L_dense, L_lex, L_inter = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        Ld_dense, Ld_lex = torch.tensor(0), torch.tensor(0)
        if not self.self_supervised:
            text_img_pairs = [
                (dense_vec_texts, dense_vec_imgs),
                (single_sparse_texts, single_sparse_imgs)
            ]
            assert all((t is None and i is None) or (t is not None and i is not None) for t, i in text_img_pairs)
        # 1) 점수 계산
        if self.hard_negatives:
            targets = torch.zeros(B, dtype=torch.long, device=device)
            if dense_vec_imgs is not None and dense_vec_texts is not None:
                sdense_t2i = torch.einsum('bd, bkd -> bk', dense_vec_texts[:, 0, :], dense_vec_imgs) / self.temp
                sdense_i2t = torch.einsum('bd, bkd -> bk', dense_vec_imgs[:, 0, :], dense_vec_texts) / self.temp
                L_dense = 0.5*(F.cross_entropy(sdense_t2i, targets)+F.cross_entropy(sdense_i2t, targets))
            if single_sparse_imgs is not None and single_sparse_texts is not None:
                slex_t2i = torch.einsum('bd, bkd -> bk', single_sparse_texts[:, 0, :], single_sparse_imgs) 
                slex_i2t = torch.einsum('bd, bkd -> bk', single_sparse_imgs[:, 0, :], single_sparse_texts)
                L_lex   = 0.5*(F.cross_entropy(slex_t2i, targets)+F.cross_entropy(slex_i2t, targets))
        else:
            if dense_vec_imgs is not None and dense_vec_texts is not None:
                sdense = (dense_vec_imgs @ dense_vec_texts.t()) / self.temp
                L_dense = self.infoNCE(sdense, idxs)
            if single_sparse_imgs is not None and single_sparse_texts is not None:
                slex   = (single_sparse_imgs @ single_sparse_texts.t())
                L_lex   = self.infoNCE(slex, idxs)

        if not self.self_supervised:
            pairs = [
                (dense_vec_texts, dense_vec_imgs),
                (single_sparse_texts, single_sparse_imgs),
            ]
            self.lam1, self.lam2 = [0 if t is None and i is None else 1 for t, i in pairs]
            assert self.lam1 + self.lam2 == 1

        # 3) 통합 점수 (teacher)
        if self.self_supervised:
            if self.hard_negatives:
                sinter_t2i = self.w1 * sdense_t2i + self.w2 * slex_t2i
                sinter_i2t =  self.w1 * sdense_i2t + self.w2 * slex_i2t
                p_sinter_i2t = torch.softmax(sinter_i2t, dim=1)
                p_sinter_t2i = torch.softmax(sinter_t2i, dim=1)

                Ld_dense = 0.5 * (
                    self.distill_loss(sdense_t2i, p_sinter_i2t) +
                    self.distill_loss(sdense_i2t, p_sinter_t2i)
                )
                Ld_lex = 0.5 * (
                    self.distill_loss(slex_t2i,   p_sinter_i2t) +
                    self.distill_loss(slex_i2t, p_sinter_t2i)
                )

                # 5) 통합 손실 L 및 L'
                L_inter = 0.5*(F.cross_entropy(sinter_t2i, targets)+F.cross_entropy(sinter_i2t, targets))
            else:
                sinter_i2t = self.w1 * sdense + self.w2 * slex
                sinter_t2i = self.w1 * sdense + self.w2 * slex

                p_sinter_i2t = torch.softmax(sinter_i2t, dim=1)
                p_sinter_t2i = torch.softmax(sinter_t2i, dim=1)
                if dense_vec_imgs is not None and dense_vec_texts is not None:
                    Ld_dense = 0.5 * (
                    self.distill_loss(sdense, p_sinter_i2t) +
                    self.distill_loss(sdense.t(), p_sinter_t2i)
                    )
                if single_sparse_imgs is not None and single_sparse_texts is not None:
                    Ld_lex = 0.5 * (
                    self.distill_loss(slex,   p_sinter_i2t) +
                    self.distill_loss(slex.t(), p_sinter_t2i)
                    )
                # 5) 통합 손실 L 및 L'
                L_inter = 0.5*(self.infoNCE(sinter_t2i, idxs)+self.infoNCE(sinter_i2t, idxs))
            L  = (self.lam1*L_dense + self.lam2*L_lex + L_inter) / len(self.pooling)

            Lp = (self.lam1*Ld_dense + self.lam2*Ld_lex) / len(self.pooling)



            # 6) 최종 손실
            loss = 0.5 * (L + Lp)
        else:

            L  = (self.lam1*L_dense + self.lam2*L_lex)
            Lp= torch.tensor(0)
            loss = L
        # 7) 정규화 항 (기존 코드 유지)
        if self.self_supervised:
            single_text_reg, single_imgs_reg = torch.tensor(0), torch.tensor(0)
            if single_sparse_imgs is not None and single_sparse_texts is not None:
                single_text_reg = self.q_regularizer(single_sparse_texts) 
                single_imgs_reg = self.d_regularizer(single_sparse_imgs)
            text_reg = single_text_reg
            img_reg = single_imgs_reg
        else:
            if single_sparse_imgs is not None and single_sparse_texts is not None:
                single_text_reg = self.q_regularizer(single_sparse_texts) 
                single_imgs_reg = self.d_regularizer(single_sparse_imgs)
                text_reg = single_text_reg
                img_reg = single_imgs_reg 
        if text_reg is not None and img_reg is not None:
            reg = 0.5 * (text_reg + img_reg)
            self.q_regularizer.step()
            self.d_regularizer.step()
        return loss, reg, text_reg, img_reg, L, Lp, single_text_reg, single_imgs_reg
