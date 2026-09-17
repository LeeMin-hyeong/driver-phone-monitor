import unittest
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from fast_forest import BatchOneForest


class FastForestTests(unittest.TestCase):
    def test_matches_sklearn_at_float32_split_boundaries(self):
        rng=np.random.default_rng(10)
        x=rng.normal(size=(100,5)).astype(np.float32)
        y=(x[:,0]+x[:,1]>.2).astype(int)
        model=ExtraTreesClassifier(n_estimators=7,max_depth=5,random_state=2,n_jobs=1).fit(x,y)
        fast=BatchOneForest(model)
        probes=list(x)
        for e in model.estimators_:
            for feature,threshold in zip(e.tree_.feature,e.tree_.threshold):
                if feature<0:continue
                for value in [np.float32(threshold),np.nextafter(np.float32(threshold),np.float32(-np.inf)),np.nextafter(np.float32(threshold),np.float32(np.inf))]:
                    row=np.zeros(5,dtype=np.float32);row[feature]=value;probes.append(row)
        expected=model.predict_proba(np.asarray(probes))
        actual=np.concatenate([fast.predict_proba([x]) for x in probes])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_rejects_unsupported_batch_and_nan(self):
        model=ExtraTreesClassifier(n_estimators=2,random_state=0).fit([[0],[1]],[0,1])
        fast=BatchOneForest(model)
        with self.assertRaises(ValueError):fast.predict_proba([[0],[1]])
        with self.assertRaises(ValueError):fast.predict_proba([[np.nan]])


if __name__=='__main__':unittest.main()
