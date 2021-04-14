import numpy as np

class pos_emb_calculator:
    def __init__(self, Dim=20, L=50):
        assert Dim%2==0
        d = int(Dim/2)
        self.Dim = Dim
        self.L = L
        tmp_u = np.tile(np.arange(L).reshape(-1,1),d) #[L,d]
        tmp_v = np.power(10000, np.arange(d)*2/(2*d)).reshape(1,-1) #[1,d]
        self.pos_arr = np.zeros((L,Dim),dtype=np.float32)
        self.pos_arr[:,::2] = np.sin(tmp_u/tmp_v)
        self.pos_arr[:,1::2] = np.cos(tmp_u/tmp_v)
    
    def calc(self, arr):
        # input: arr = [N,3]
        # output: ret = [N,3*D]
        arr[arr>=self.L] = self.L-1
        arr[arr<0] = self.L-1
        ret = self.pos_arr[arr]
        return ret

pos_e = pos_emb_calculator(Dim=4,L=6)
a = np.arange(30).reshape(-1,3)
pos_e.calc(a)
    
    
