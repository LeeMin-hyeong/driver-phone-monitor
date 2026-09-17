"""Vectorized batch-one ExtraTrees traversal with unchanged learned trees."""
import numpy as np


class BatchOneForest:
    def __init__(self,model):
        trees=[e.tree_ for e in model.estimators_]
        self.count=len(trees)
        size=max(t.node_count for t in trees)
        self.features=np.full((self.count,size),-2,dtype=np.int32)
        self.threshold=np.zeros((self.count,size),dtype=np.float64)
        self.left=np.zeros((self.count,size),dtype=np.int32)
        self.right=np.zeros((self.count,size),dtype=np.int32)
        self.values=np.zeros((self.count,size,len(model.classes_)),dtype=np.float64)
        self.depth=max(t.max_depth for t in trees)
        self.indices=np.arange(self.count)
        self.classes_=model.classes_
        self.n_features=model.n_features_in_
        for i,t in enumerate(trees):
            n=t.node_count
            self.features[i,:n]=t.feature
            self.threshold[i,:n]=t.threshold
            self.left[i,:n]=t.children_left
            self.right[i,:n]=t.children_right
            v=t.value[:,0,:]
            self.values[i,:n]=v/np.maximum(v.sum(axis=1,keepdims=True),1e-300)

    def predict_proba(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(1,self.n_features) or not np.isfinite(x).all():
            raise ValueError('BatchOneForest requires one finite feature row')
        nodes=np.zeros(self.count,dtype=np.int32)
        for _ in range(self.depth+1):
            f=self.features[self.indices,nodes]
            active=f>=0
            if not active.any():break
            indices=self.indices[active];n=nodes[active]
            left=x[0,f[active]]<=self.threshold[indices,n]
            nodes[active]=np.where(left,self.left[indices,n],self.right[indices,n])
        return self.values[self.indices,nodes].mean(axis=0,keepdims=True)
