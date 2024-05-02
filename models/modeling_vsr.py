# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from models.modeling_finetune import VisionTransformer
import math
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pad_sequence


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, inputBatch):
        return inputBatch.transpose(self.dim1, self.dim2)
    

class outputConv(nn.Module):
    def __init__(self, MaskedNormLayer, dModel, numClasses):
        super(outputConv, self).__init__()
        if MaskedNormLayer == "LN":
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )
        else:
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )

    def forward(self, inputBatch):
        return self.outputconv(inputBatch)


class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch
    

def generate_square_subsequent_mask(sz: int, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def compute_CTC_prob(h, alpha, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, numClasses, blank, eosIx):
    batch = h.shape[0]
    g = h[:, :, :, :-1]
    c = h[:, :, :, -1]
    alphaCTC = torch.zeros_like(alpha)
    eosIxMask = c == eosIx
    eosIxIndex = eosIxMask.nonzero()
    eosIxIndex = torch.cat((eosIxIndex[:, :1], torch.repeat_interleave((T - 1).unsqueeze(-1), numBeam, dim=0), eosIxIndex[:, 1:]), dim=-1).long()
    eosIxIndex[:, -1] = 0
    gamma_eosIxMask = torch.zeros_like(gamma_n).bool()
    gamma_eosIxMask.index_put_(tuple(map(torch.stack, zip(*eosIxIndex))), torch.tensor(True))
    alphaCTC[eosIxMask] = np.logaddexp(gamma_n[gamma_eosIxMask], gamma_b[gamma_eosIxMask])

    if g.shape[-1] == 1:
        gamma_n[:, 1, 0, 1:-1] = CTCOutLogProbs[:, 1, 1:-1]
    else:
        gamma_n[:, 1, :numBeam, 1:-1] = -np.inf
    gamma_b[:, 1, :numBeam, 1:-1] = -np.inf

    psi = gamma_n[:, 1, :numBeam, 1:-1]
    for t in range(2, T.max()):
        activeBatch = t < T
        gEndWithc = (g[:, :, :, -1] == c)[:, :, :-1].nonzero()
        added_gamma_n = torch.repeat_interleave(gamma_n[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1)
        if len(gEndWithc):
            added_gamma_n.index_put_(tuple(map(torch.stack, zip(*gEndWithc))), torch.tensor(-np.inf).float())
        phi = np.logaddexp(torch.repeat_interleave(gamma_b[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1), added_gamma_n)
        expandShape = [batch, numBeam, numClasses - 1]
        gamma_n[:, t, :numBeam, 1:-1][activeBatch] = np.logaddexp(gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch], phi[activeBatch]) \
                                                     + CTCOutLogProbs[:, t, None, 1:-1].expand(expandShape)[activeBatch]
        gamma_b[:, t, :numBeam, 1:-1][activeBatch] = \
            np.logaddexp(gamma_b[:, t - 1, :numBeam, 1:-1][activeBatch], gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch]) \
            + CTCOutLogProbs[:, t, None, None, blank].expand(expandShape)[activeBatch]
        psi[activeBatch] = np.logaddexp(psi[activeBatch], phi[activeBatch] + CTCOutLogProbs[:, t, None, 1:-1].expand(phi.shape)[activeBatch])
    return torch.cat((psi, alphaCTC[:, :, -1:]), dim=-1)


class MaskedLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(MaskedLayerNorm, self).__init__()
        # self.register_buffer('mask', None, persistent=False)
        # self.register_buffer('inputLenBatch', None, persistent=False)
        self.eps = eps

    def SetMaskandLength(self, mask, inputLenBatch):
        self.mask = mask
        self.inputLenBatch = inputLenBatch

    def expand2shape(self, inputBatch, expandedShape):
        return inputBatch.unsqueeze(-1).unsqueeze(-1).expand(expandedShape)

    def forward(self, inputBatch):
        dModel = inputBatch.shape[-1]
        maskBatch = ~self.mask.unsqueeze(-1).expand(inputBatch.shape)

        meanBatch = (inputBatch * maskBatch).sum((1, 2)) / (self.inputLenBatch * dModel)
        stdBatch = ((inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) ** 2 * maskBatch).sum((1, 2))
        stdBatch = stdBatch / (self.inputLenBatch * dModel)

        # Norm the input
        normed = (inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) / \
                 (torch.sqrt(self.expand2shape(stdBatch + self.eps, inputBatch.shape)))
        return normed


class SmoothCTCLoss(_Loss):
    def __init__(self, num_classes, blank=0, weight=0.01):
        super().__init__(reduction='mean')
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)
        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon: float = 0.01, reduction='mean', pad = 0):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.pad = pad

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds.reshape(-1, n), dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target.long().reshape(-1), ignore_index=self.pad, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class ViT_VSR(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 with_cp=False,
                 cos_attn=False):
        super().__init__()
        self.pre_model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            head_drop_rate=head_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            init_scale=init_scale,
            all_frames=all_frames,
            tubelet_size=tubelet_size,
            use_mean_pooling=False,
            with_cp=with_cp,
            cos_attn=cos_attn
        )
        self.numClasses = num_classes
        dModel = 384 # TODO
        peMaxLen = 3000
        self.maskedLayerNorm = "LN" #MaskedLayerNorm()
        tx_norm = nn.LayerNorm(dModel)

        self.jointOutputConv = outputConv(self.maskedLayerNorm, dModel, self.numClasses)
        self.decoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.embed = torch.nn.Sequential(
            nn.Embedding(self.numClasses, dModel),
            self.decoderPositionalEncoding
        )
        jointDecoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=num_heads, dim_feedforward=1024, dropout=0.1)
        self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=depth, norm=tx_norm)
        self.jointAttentionOutputConv = outputConv("LN", dModel, self.numClasses)

        self.CEloss = SmoothCrossEntropyLoss(pad=0)
        self.CTCloss = SmoothCTCLoss(self.numClasses, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    def forward(self, src, tgt, src_len, tgt_len):
        B = src.size(0)
        T = src.size(2)

        with torch.no_grad():
            x = self.pre_model.patch_embed(src)
            # B x 3 x T x H x W
            # x = self.pre_model.patch_embed.proj(src)
            # B x 384 x T' x H' x W'
            # x = x.transpose(1,2).reshape(B, -1, 75264) # 384*14*14
            # B x T' x D'

            if self.pre_model.pos_embed is not None:
                # TODO replace by dynamic length pe
                x = x + self.get_sinusoid_encoding_table(x.shape[1], x.shape[2]).type_as(x).to(x.device).clone().detach()
            x = self.pre_model.pos_drop(x)
            for blk in self.pre_model.blocks:
                if self.pre_model.with_cp:
                    x = cp.checkpoint(blk, x)
                else:
                    x = blk(x)
        # B x (T'x14x14) x 384
        x = x.reshape(B, -1, 196, 384)
        x = x.mean(dim=2)
        # B x T' x 384
        jointCTCOutputBatch = x.transpose(0, 1).transpose(1, 2)
        jointCTCOutputBatch = self.jointOutputConv(jointCTCOutputBatch)
        jointCTCOutputBatch = jointCTCOutputBatch.permute(0,2,1)
        jointCTCOutputBatch = F.log_softmax(jointCTCOutputBatch, dim=2)

        tgt_list = [tgt[i][:tgt_len[i]-1] for i in range(B)] # we dont need eos
        targetinBatch = pad_sequence(tgt_list, batch_first=True)
        targetinBatch = self.embed(targetinBatch.transpose(0, 1))
        targetinMask = self.makeMaskfromLength(targetinBatch.shape[:-1][::-1], tgt_len-1, targetinBatch.device)
        squareMask = generate_square_subsequent_mask(targetinBatch.shape[0], targetinBatch.device)
        mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for i in range(len(src_len)):
            mask[i, (src_len[i]-2)//2+1:] = 1
        x = x.permute(1,0,2)
        jointAttentionOutputBatch = self.jointAttentionDecoder(targetinBatch, x, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask, memory_key_padding_mask=mask)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(0, 1).transpose(1, 2)
        jointAttentionOutputBatch = self.jointAttentionOutputConv(jointAttentionOutputBatch)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(1, 2)

        outputBatch = (jointCTCOutputBatch, jointAttentionOutputBatch) # B x T x C
        inputLenBatch=(src_len-2)//2+1
        return inputLenBatch, outputBatch
    
    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def inference(self, src, tgt, src_len, tgt_len):
        B = src.size(0)
        eosIx = 39
        beamWidth = 5
        blank = 0
        device = src.device
        Lambda = 0.1
        # T = src.size(2)
        with torch.no_grad():
            x = self.pre_model.patch_embed(src)
            if self.pre_model.pos_embed is not None:
                # TODO replace by dynamic length pe
                x = x + self.get_sinusoid_encoding_table(x.shape[1], x.shape[2]).type_as(x).to(x.device).clone().detach()
            x = self.pre_model.pos_drop(x)
            for blk in self.pre_model.blocks:
                if self.pre_model.with_cp:
                    x = cp.checkpoint(blk, x)
                else:
                    x = blk(x)
        # B x (T'x14x14) x 384
        x = x.reshape(B, -1, 196, 384)
        x = x.mean(dim=2)
        # B x T' x 384
        CTCOutputConv = self.jointOutputConv
        attentionDecoder = self.jointAttentionDecoder
        attentionOutputConv = self.jointAttentionOutputConv

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2)
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)
        # claim batch and time step
        batch = CTCOutputBatch.shape[0]
        T = src_len.cpu()
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1)
        predictionLenBatch = torch.ones(batch, device=device).long()
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]
        Omegahat = [[] for i in range(batch)]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]

        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool()
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1)
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu()

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)
            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1)

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)
            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]
            hPaddingShape = list(h.shape)
            hPaddingShape[-2] = 1
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)

            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()


@register_model
def vit_giant_patch14_224_vsr(pretrained=False, **kwargs):
    model = ViT_VSR(
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_mean_pooling=False,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_patch16_224_vsr(pretrained=False, **kwargs):
    model = ViT_VSR(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model