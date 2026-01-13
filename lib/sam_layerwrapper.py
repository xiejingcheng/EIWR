import torch
import torch.nn as nn
import math
import time

import transformers
from tqdm import tqdm

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", add_output=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        if add_output:
            self.scaler_row = torch.zeros((self.rows), device=self.dev)
        else:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.is_H = False
        self.wanda_v2 = False

    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def add_batch_out(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            out = out.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        out = out.type(torch.float32)

        self.scaler_row += torch.norm(out, p=2, dim=1) ** 2  / self.nsamples

    def add_batch_out_withH(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            out = out.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        out = out.type(torch.float32)
        self.scaler_row += torch.norm(out, p=2, dim=1) ** 2  / self.nsamples

        self.is_H = True
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())



    def add_batch_withH(self, inp, out):
        self.is_H = True
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.H *= self.nsamples / (self.nsamples + tmp)

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

class WrappedGPTV3:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, have_scaler=True, valid=True):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.H_B = torch.zeros((self.rows, self.columns), device=self.dev)
        self.nsamples = 0
        self.have_scaler = have_scaler
        if self.have_scaler:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.valid = valid
        self.valid_num = [0,3,122,125]
        if self.valid:
            self.valid_inps = []
            self.valid_outs = []

    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1])
        out = out.reshape(-1, out.shape[-1])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            inp = inp.t() 
            out = out.t()

        if self.valid:
            if self.nsamples in self.valid_num:
                self.valid_inps.append(inp)
                self.valid_outs.append(out)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.H_B *= self.nsamples / (self.nsamples + tmp)

        if self.have_scaler:
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp

        if self.have_scaler:
            self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2  / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        out = math.sqrt(2 / self.nsamples) * out.float()
        self.H += inp.matmul(inp.t())
        self.H_B += out.matmul(inp.t())

    def dual_ascent(self, beta=0.005, alpha=0.995, gama=0.0000, rho=1, epsilon=2e-2, max_iter=10000, lambda_zero=False, percdamp=.01):
        W_old = self.layer.weight.data.clone()
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        H_A = self.H
        H_B = self.H_B
        del self.H, self.H_B

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, H_A) - H_B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)

        for k in range(max_iter):
            # 保存上一次的 W
            W_prev = W.clone()

            # 更新 W
            A = (beta + gama) * H_A + alpha * torch.eye(H_A.shape[0], device=H_A.device)
            try:
                damp = percdamp * torch.mean(torch.diag(A))
                diag = torch.arange(A.shape[-1], device=A.device)
                A[diag, diag] += damp
                A = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(A)
                # A_inv = torch.linalg.cholesky(A, upper=True)
            except RuntimeError as e:
                print(f"Cholesky decomposition failed: {e}. Falling back to direct inverse.")
                raise e
            
            B = beta * H_B + alpha * W_old
            W = torch.mm(B - (M * Lambda), A_inv)

            # 更新 Lambda
            Lambda = Lambda + rho * (M * W)

            # 收敛判断
            if k % 10 == 0 :
                W = W - M * W
                
                if torch.norm(W - W_prev) < epsilon:
                    print(f"Converged at iteration {k}")
                    print(torch.norm(W - W_prev))
                    break
            # if k % 10 == 0 :
            #     print(torch.norm(W - W_prev))

        torch.cuda.synchronize()
        W = W - M * W
        print("Dual ascent finished!")
        print(torch.norm(W - W_prev))
        self.layer.weight.data = W.to(torch.float32)

    def dual_ascent2(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()
        old_score = (W_old @ self.valid_inps[0] - self.valid_outs[0]).abs().mean()
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        H_A = self.H
        H_B = self.H_B
        del self.H, self.H_B

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, H_A) - H_B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)

        for k in range(max_iter):
            # 保存上一次的 W
            W_prev = W.clone()

            # 更新 W
            A = (beta + gama) * H_A + alpha * torch.eye(H_A.shape[0], device=H_A.device)
            try:
                damp = percdamp * torch.mean(torch.diag(A))
                diag = torch.arange(A.shape[-1], device=A.device)
                A[diag, diag] += damp
                A = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(A)
                # A_inv = torch.linalg.cholesky(A, upper=True)
            except RuntimeError as e:
                print(f"Cholesky decomposition failed: {e}. Falling back to direct inverse.")
                raise e
            
            B = beta * H_B + alpha * W_old
            W = torch.mm(B - (M * Lambda), A_inv)

            # 更新 Lambda
            Lambda = Lambda + rho * (M * W)

            # 收敛判断
            if k % 10 == 0 :
                # W = W - M * W
                
                if k > min_iter:
                    if torch.norm(W - W_prev) < epsilon:
                        print(f"Converged at iteration {k}")
                        print(torch.norm(W - W_prev))
                        break
            # if k % 10 == 0 :
            #     print(torch.norm(W - W_prev))

        torch.cuda.synchronize()
        W = W - M * W
        print(torch.norm(W - W_prev))
        new_score = (W.to(torch.float32) @ self.valid_inps[0] - self.valid_outs[0]).abs().mean()
        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float32)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float32)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def del_valid(self):
        self.valid = False
        self.valid_inps = []
        self.valid_outs = []
        
    def validate(self):
        W = self.layer.weight.data
        print("out_inf:", self.valid_outs[0].max(), self.valid_outs[0].abs().mean())
        if self.valid:
            for i in range(len(self.valid_inps)):
                inp = self.valid_inps[i]
                out = self.valid_outs[i]
                print((W @ inp - out).abs().max(), (W @ inp - out).abs().mean())
   
    def get_args(self):
        if (self.layer.weight.data @ self.valid_inps[0] - self.valid_outs[0]).abs().mean() < 0.1 * self.valid_outs[0].abs().mean():
            return False, 0.99, 0.01
        if self.valid_outs[0].abs().mean().item() < 0.1:
            return True, 0.99, 0.01
        if self.valid_outs[0].abs().mean().item() < 0.5:
            return True, 0.99, 0.01
        else:
            return True, 0.99, 0.01

    def free(self):
        self.H = None
        self.H_A = None
        self.H_B = None
        torch.cuda.empty_cache()


class SparseGPT:

    def __init__(self, layer, have_scaler=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.have_scaler = have_scaler
        if self.have_scaler:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.valid_out = False

    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 

        # if self.valid_out:
        #     valid_out = self.layer.weight.data @ inp
        #     print((valid_out - out).abs().max(), (valid_out - out).abs().mean())
        #     print((valid_out.t() - out).abs().max(), (valid_out.t() - out).abs().mean())
        #     print("valid_out is valid")

        self.H *= self.nsamples / (self.nsamples + tmp)
        if self.have_scaler:
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        if self.have_scaler:
            self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2  / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        # if hasattr(self.layer, 'is_RHT'):
        #     dead = torch.diag(H) == 0
        #     H[dead, dead] = 1
        #     W[:, dead] = 0
        #     damp = percdamp * torch.mean(torch.diag(H))
        #     diag = torch.arange(self.columns, device=self.dev)
        #     H[diag, diag] += damp

        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class SparseGPTV3:

    def __init__(self, layer, have_scaler=False, valid=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.H_B = torch.zeros((self.rows, self.columns), device=self.dev)
        self.nsamples = 0
        self.have_scaler = have_scaler
        if self.have_scaler:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.valid = valid
        self.valid_num = [0, 32, 64, 125, 126]
        if self.valid:
            self.valid_inps = []
            self.valid_outs = []

    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1])
        out = out.reshape(-1, out.shape[-1])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            inp = inp.t() 
            out = out.t()

        if self.valid:
            if self.nsamples in self.valid_num:
                self.valid_inps.append(inp)
                self.valid_outs.append(out)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.H_B *= self.nsamples / (self.nsamples + tmp)

        if self.have_scaler:
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp

        if self.have_scaler:
            self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2  / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        out = math.sqrt(2 / self.nsamples) * out.float()
        self.H += inp.matmul(inp.t())
        self.H_B += out.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        try:
            H = torch.linalg.cholesky(H)
        except RuntimeError as e:
            print(f"Cholesky decomposition failed: {e}. Falling back to direct inverse.")
            import numpy as np
            np.save("/h3cstore_ns/jcxie/LISA/wanda-main/npys/error/H.npy", H.cpu().numpy())
            np.save("/h3cstore_ns/jcxie/LISA/wanda-main/npys/error/W.npy", W.cpu().numpy())
            np.save("/h3cstore_ns/jcxie/LISA/wanda-main/npys/error/H_B.npy", self.H_B.cpu().numpy())
            print(torch.isnan(H).sum())
            raise e
        H = torch.cholesky_inverse(H)


        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def dual_ascent(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300):
        W_old = self.layer.weight.data.clone()
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        H_A = self.H
        H_B = self.H_B
        del self.H, self.H_B

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, H_A) - H_B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)

        for k in range(max_iter):
            # 保存上一次的 W
            W_prev = W.clone()

            # 更新 W
            A = (beta + gama) * H_A + alpha * torch.eye(H_A.shape[0], device=H_A.device)
            try:
                damp = percdamp * torch.mean(torch.diag(A))
                diag = torch.arange(A.shape[-1], device=A.device)
                A[diag, diag] += damp
                A = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(A)
                # A_inv = torch.linalg.cholesky(A, upper=True)
            except RuntimeError as e:
                print(f"Cholesky decomposition failed: {e}. Falling back to direct inverse.")
                raise e
            
            B = beta * H_B + alpha * W_old
            W = torch.mm(B - (M * Lambda), A_inv)

            # 更新 Lambda
            Lambda = Lambda + rho * (M * W)

            # 收敛判断
            if k % 10 == 0 :
                W = W - M * W
                
                if k > min_iter:
                    if torch.norm(W - W_prev) < epsilon:
                        print(f"Converged at iteration {k}")
                        print(torch.norm(W - W_prev))
                        break
            # if k % 10 == 0 :
            #     print(torch.norm(W - W_prev))

        torch.cuda.synchronize()
        W = W - M * W
        print("Dual ascent finished!")
        print(torch.norm(W - W_prev))
        self.layer.weight.data = W.to(torch.float32)

        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def dual_ascent2(self, beta=0.01, alpha=0.99, gama=0.0000, rho=1, epsilon=1e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()

        old_score = 0
        new_score = 0
        for v in range(len(self.valid_inps)):
            old_score += (W_old @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        old_score = old_score / len(self.valid_inps)
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        H_A = self.H
        H_B = self.H_B
        del self.H, self.H_B

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, H_A) - H_B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)

        for k in range(max_iter):
            # 保存上一次的 W
            W_prev = W.clone()

            # 更新 W
            A = (beta + gama) * H_A + alpha * torch.eye(H_A.shape[0], device=H_A.device)
            try:
                damp = percdamp * torch.mean(torch.diag(A))
                diag = torch.arange(A.shape[-1], device=A.device)
                A[diag, diag] += damp
                A = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(A)
                # A_inv = torch.linalg.cholesky(A, upper=True)
            except RuntimeError as e:
                print(f"Cholesky decomposition failed: {e}. Falling back to direct inverse.")
                raise e
            
            B = beta * H_B + alpha * W_old
            W = torch.mm(B - (M * Lambda), A_inv)

            # 更新 Lambda
            Lambda = Lambda + rho * (M * W)

            # 收敛判断
            if k % 10 == 0 :
                # W = W - M * W
                
                if k > min_iter:
                    if torch.norm(W - W_prev) < epsilon:
                        print(f"Converged at iteration {k}")
                        print(torch.norm(W - W_prev))
                        break
            # if k % 10 == 0 :
            #     print(torch.norm(W - W_prev))

        torch.cuda.synchronize()
        W = W - M * W
        print(torch.norm(W - W_prev))

        for v in range(len(self.valid_inps)):
            new_score = (W.to(torch.float32) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float32)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float32)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def del_valid(self):
        self.valid = False
        self.valid_inps = []
        self.valid_outs = []
        
    def validate(self):
        W = self.layer.weight.data
        print("out_inf:", self.valid_outs[0].max(), self.valid_outs[0].abs().mean())
        if self.valid:
            for i in range(len(self.valid_inps)):
                inp = self.valid_inps[i]
                out = self.valid_outs[i]
                print((W @ inp - out).abs().max(), (W @ inp - out).abs().mean())
   
    def get_args(self):
        if (self.layer.weight.data @ self.valid_inps[0] - self.valid_outs[0]).abs().mean() < 0.1 * self.valid_outs[0].abs().mean():
            return False, 0.99, 0.01
        if self.valid_outs[0].abs().mean().item() < 0.1:
            return True, 0.99, 0.01
        if self.valid_outs[0].abs().mean().item() < 0.5:
            return True, 0.99, 0.01
        else:
            return True, 0.99, 0.01

    def free(self):
        self.H = None
        self.H_A = None
        self.H_B = None
        torch.cuda.empty_cache()


class SparseGPTV2:

    def __init__(self, layer, have_scaler=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.have_scaler = have_scaler
        if self.have_scaler:
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.valid_out = False

        self.inps = []
        self.outs = []

    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1])
        out = out.reshape(-1, out.shape[-1])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 

        self.inps.append(inp.cpu())
        self.outs.append(out.transpose(-1,-2).cpu())

        self.H *= self.nsamples / (self.nsamples + tmp)
        if self.have_scaler:
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        if self.have_scaler:
            self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2  / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        # if hasattr(self.layer, 'is_RHT'):
        #     dead = torch.diag(H) == 0
        #     H[dead, dead] = 1
        #     W[:, dead] = 0
        #     damp = percdamp * torch.mean(torch.diag(H))
        #     diag = torch.arange(self.columns, device=self.dev)
        #     H[diag, diag] += damp

        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        self.inps = []
        self.outs = []
        torch.cuda.empty_cache()

