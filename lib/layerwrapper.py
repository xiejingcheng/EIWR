import torch
import torch.nn as nn
import math
import time
import gc

import numpy as np

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

class WrappedGPTV2:
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

        self.inps = []
        self.outs = []

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.inps.append(inp.cpu())
        self.outs.append(out.transpose(-1,-2).cpu())

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

class WrappedGPTV6:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, have_scaler=True, valid=True, ora_W=False):
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
        if ora_W:
            self.ora_W = layer.weight.data.clone()

    def add_batch(self, inp, out):
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
        self.layer.weight.data = W.to(torch.float16)

    def dual_ascent2(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()
        old_score = 0
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)
        
        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def dual_ascent3(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()
        old_score = 0
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
            
            B = beta * H_B + alpha * W_prev
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)
        
        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def alpha_schedule_exp(self, k, max_iter=100, alpha_start=0.9, alpha_end=0.99, gamma=5.0):
        # if k >= max_iter:
        #     return alpha_end
        ratio = k / max_iter
        return alpha_end - (alpha_end - alpha_start) * np.exp(-gamma * ratio) 

    def dual_ascent4(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=True, percdamp=.01, min_iter=100, theld=0.07):
        start = time.time()
        W_old = self.layer.weight.data.clone()
        old_score = 0
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

        eigvals, Q = torch.linalg.eigh(H_A)

        def apply_A_inv_right(X, eigvals, Q, alpha):
            # X: (n, n)
            d = 1.0 / ((1 - alpha) * eigvals + alpha)  # shape: (n,)
            tmp = X @ Q                                # right multiply Q
            tmp = tmp * d                              # element-wise scale columns
            return tmp @ Q.T                           # right multiply Q^T
        

        for k in range(max_iter):
            alpha = self.alpha_schedule_exp(k)
            W_prev = W.clone()

            B = (1-alpha) * H_B + alpha * W_prev
            MLambda = Lambda
            X = B - MLambda

            W = apply_A_inv_right(X, eigvals, Q, alpha)

            Lambda = Lambda + rho * (M * W)

            if k % 50 == 0 :
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        print("Time:", time.time()-start)
        del W, W_old, H_A, H_B, B, Lambda
        torch.cuda.empty_cache()

    def dual_ascent5(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=True, percdamp=.01, min_iter=100, theld=0.07, alpha_2 = False):
        start = time.time()
        W_old = self.layer.weight.data.clone()
        old_score = 0
        for v in range(len(self.valid_inps)):
            old_score += (W_old @ self.valid_inps[v] - self.ora_W @ self.valid_inps[v]).abs().mean()
        old_score = old_score / len(self.valid_inps)
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        H_A = self.H
        H_B = self.ora_W.to(torch.float32) @ self.H
        del self.H, self.H_B

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, H_A) - H_B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)

        eigvals, Q = torch.linalg.eigh(H_A)

        def apply_A_inv_right(X, eigvals, Q, alpha):
            # X: (n, n)
            d = 1.0 / ((1 - alpha) * eigvals + alpha)  # shape: (n,)
            tmp = X @ Q                                # right multiply Q
            tmp = tmp * d                              # element-wise scale columns
            return tmp @ Q.T                           # right multiply Q^T
        

        for k in range(max_iter):
            if alpha_2:
                alpha = self.alpha_schedule_exp2(k)
            else:
                alpha = self.alpha_schedule_exp(k)
            W_prev = W.clone()

            B = (1-alpha) * H_B + alpha * W_prev
            MLambda = Lambda
            X = B - MLambda

            W = apply_A_inv_right(X, eigvals, Q, alpha)

            Lambda = Lambda + rho * (M * W)

            if k % 50 == 0 :
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.ora_W @ self.valid_inps[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)
        del self.ora_W

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        print("Time:", time.time()-start)
        del W, W_old, H_A, H_B, B, Lambda
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
        self.layer.weight.data = W.to(torch.float16)

    def dual_ascent2(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()
        old_score = 0
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)
        
        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def dual_ascent3(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=False, percdamp=.01, min_iter=300, theld=0.07):
        W_old = self.layer.weight.data.clone()
        old_score = 0
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
            
            B = beta * H_B + alpha * W_prev
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)
        
        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        del W, W_old, H_A, H_B, A, B, Lambda
        torch.cuda.empty_cache()

    def alpha_schedule_exp(self, k, max_iter=100, alpha_start=0.9, alpha_end=0.99, gamma=5.0):
        # if k >= max_iter:
        #     return alpha_end
        ratio = k / max_iter
        return alpha_end - (alpha_end - alpha_start) * np.exp(-gamma * ratio) 

    def dual_ascent4(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=True, percdamp=.01, min_iter=100, theld=0.07):
        start = time.time()
        W_old = self.layer.weight.data.clone()
        old_score = 0
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

        eigvals, Q = torch.linalg.eigh(H_A)

        def apply_A_inv_right(X, eigvals, Q, alpha):
            # X: (n, n)
            d = 1.0 / ((1 - alpha) * eigvals + alpha)  # shape: (n,)
            tmp = X @ Q                                # right multiply Q
            tmp = tmp * d                              # element-wise scale columns
            return tmp @ Q.T                           # right multiply Q^T
        

        for k in range(max_iter):
            alpha = self.alpha_schedule_exp(k)
            W_prev = W.clone()

            B = (1-alpha) * H_B + alpha * W_prev
            MLambda = Lambda
            X = B - MLambda

            W = apply_A_inv_right(X, eigvals, Q, alpha)

            Lambda = Lambda + rho * (M * W)

            if k % 50 == 0 :
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        print("Time:", time.time()-start)
        del W, W_old, H_A, H_B, B, Lambda
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



class WrappedGPTV4:
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
        self.ora_W = W.clone()
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
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))

            inp = inp.t() 

        if self.valid:
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
                out = out.t()
            if self.nsamples in self.valid_num:
                self.valid_inps.append(inp)
                self.valid_outs.append(out)

        self.H *= self.nsamples / (self.nsamples + tmp)


        if self.have_scaler:
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)

        self.nsamples += tmp

        if self.have_scaler:
            self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2  / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        self.H += inp.matmul(inp.t())

    def dual_ascent2(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-3, max_iter=1000, lambda_zero=True, percdamp=.01, min_iter=300, theld=0.07):
        start = time.time()
        W_old = self.layer.weight.data.clone()
        old_score = 0
        for v in range(len(self.valid_inps)):
            old_score += (W_old @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        old_score = old_score / len(self.valid_inps)
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        A = self.H
        B = self.ora_W.to(torch.float32) @ A
        del self.H, self.ora_W
        torch.cuda.empty_cache()

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)
        else:
            term1 = beta * (torch.mm(W, A) - B)
            term2 = alpha * (W - W_old)
            Lambda = -M * (term1 + term2)
        
        A = (beta + gama) * A 
        A = A + alpha * torch.eye(A.shape[0], device=A.device)
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
        B = beta * B + alpha * W_old


        torch.cuda.empty_cache()

        for k in range(max_iter):
            # 保存上一次的 W
            W_prev = W.clone()
            
            
            W = torch.mm(B - (M * Lambda), A_inv)

            # 更新 Lambda
            Lambda = Lambda + rho * (M * W)

            # 收敛判断
            if k % 100 == 0 :
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        print("Time:", time.time()-start)
        del W, W_old, A, B, Lambda
        torch.cuda.empty_cache() 


    def alpha_schedule_exp(self, k, max_iter=100, alpha_start=0.9, alpha_end=0.99, gamma=5.0):

        ratio = k / max_iter
        return alpha_end - (alpha_end - alpha_start) * np.exp(-gamma * ratio) 

    def dual_ascent4(self, beta=0.05, alpha=0.95, gama=0.0000, rho=1, epsilon=2e-2, max_iter=1000, lambda_zero=True, percdamp=.01, min_iter=100, theld=0.07):
        start = time.time()
        W_old = self.layer.weight.data.clone()
        old_score = 0
        for v in range(len(self.valid_inps)):
            old_score += (W_old @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        old_score = old_score / len(self.valid_inps)
        W_old = W_old.to(torch.float32)
        M = (W_old == 0).to(torch.float32)
        B_H = self.ora_W.to(torch.float32) @ self.H
        eigvals, Q = torch.linalg.eigh(self.H)
        del self.H, self.ora_W
        gc.collect()
        torch.cuda.empty_cache()

        W = W_old.clone()
        if lambda_zero:
            Lambda = torch.zeros_like(W)

        gc.collect()
        torch.cuda.empty_cache()

        def apply_A_inv_right(X, eigvals, Q, alpha):
            # X: (n, n)
            d = 1.0 / ((1 - alpha) * eigvals + alpha)  # shape: (n,)
            tmp = X @ Q                                # right multiply Q
            tmp = tmp * d                              # element-wise scale columns
            return tmp @ Q.T                           # right multiply Q^T
        

        for k in tqdm(range(max_iter)):
            alpha = self.alpha_schedule_exp(k)
            W_prev = W.clone()

            B = (1-alpha) * B_H + alpha * W_prev
            MLambda = Lambda
            B = B - MLambda

            W = apply_A_inv_right(B, eigvals, Q, alpha)

            Lambda = Lambda + rho * (M * W)

            if k % 50 == 0 :
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
        new_score = 0
        for v in range(len(self.valid_inps)):
            new_score += (W.to(torch.float16) @ self.valid_inps[v] - self.valid_outs[v]).abs().mean()
        new_score = new_score / len(self.valid_inps)

        print("old_score:", old_score, "new_score:", new_score)
        if new_score < (old_score * (1 - theld)):
            print("Converged!")
            self.layer.weight.data = W.to(torch.float16)
        else:
            print("Not converged!")
            self.layer.weight.data = W_old.to(torch.float16)

        print("Dual ascent finished!")
        print("Time:", time.time()-start)
        del W, W_old, B, Lambda, Q, eigvals, B_H
        gc.collect()
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

class WrappedGPTV10:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, initial_method = None, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0

        self.initial_method = initial_method
        if self.initial_method == "sparsegpt":
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.sum_metric_row = torch.zeros((self.columns), device=self.dev)
        
        self.mean = torch.zeros((self.columns), device=self.dev)
        self.var = torch.zeros((self.columns), device=self.dev)
        self.ntokens = 0
        
        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.type(torch.float32)

        mean_inp = torch.mean(inp, dim=1, keepdim=True)

        var_inp = torch.var(inp, dim=1, unbiased=False, keepdim=True)
        num_inp = inp.shape[1]
        self.var = var_inp if self.ntokens == 0 else (self.var * self.ntokens + var_inp * num_inp) / (self.ntokens + num_inp)
        self.mean = mean_inp if self.ntokens == 0 else (self.mean * self.ntokens + mean_inp * num_inp) / (self.ntokens + num_inp)
        self.ntokens += num_inp
        
        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.sum_metric_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.sum_metric_row += torch.sum(inp, dim=1) / self.nsamples

        if self.initial_method == "sparsegpt":
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    def free(self):
        self.H = None
        torch.cuda.empty_cache()