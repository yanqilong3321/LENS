import copy
import json
from pathlib import Path
import unittest

import torch

from lens import core
from lens.data import load_graph, load_pool
from run_cora import dispatch
from summarize import select

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        core.EPOCHS = 2
        core.BACKBONE = 'GCN'
        cls.graph, cls.tensors, cls.splits = load_graph(ROOT / 'data')

    def test_pool_disjointness_and_budget(self):
        for seed in range(360,370):
            p = load_pool(self.graph,self.tensors,self.splits,seed)
            self.assertEqual(len(p['nodes']),350)
            self.assertEqual(int(p['test'].sum()),1858)
            self.assertFalse(bool((p['test'] & p['val']).any()))
            self.assertTrue(bool((p['y'][p['nodes']] >= 0).all()))
            self.assertTrue(bool((p['y'][p['test'] | p['val']] == -1).all()))

    def test_configuration_completeness(self):
        cfg = json.loads((ROOT / 'configs/cora.json').read_text())
        self.assertEqual(cfg['seeds'], list(range(360,370)))
        self.assertEqual(set(cfg['groups']), {'GCN','SAGE','GAT','robust'})
        for suite,groups in cfg['groups'].items():
            self.assertEqual(len(groups), 6 if suite == 'robust' else 4)
            for candidates in groups.values():
                self.assertTrue(candidates)
                self.assertEqual(len(candidates),len(set(candidates)))
                for method in candidates:
                    name,*items = method.split(':')
                    self.assertIn(name, {'ce','recipe','plug'})
                    for item in items:
                        key,value = item.split('=')
                        self.assertTrue(key)
                        float(value)

    def test_selection_does_not_use_test_accuracy(self):
        records = {'a/360':{'val':70,'test':99}, 'b/360':{'val':71,'test':1}}
        self.assertEqual(select(records,['a','b'],360)[0],'b')
        records['a/360']['val'] = 71
        self.assertEqual(select(records,['a','b'],360)[0],'a')
        with self.assertRaises(ValueError):
            select(records,['a','missing'],360)

    def test_gold_isolation_and_determinism(self):
        p = load_pool(self.graph,self.tensors,self.splits,360)
        changed = dict(self.graph,gold=torch.zeros_like(self.graph['gold']))
        methods = ['ce','recipe:sf=0.25:dg=2:dmode=5:aw=1:qw=1:cbp=0:lp=0',
                   'recipe:st=1:sf=0.25:dg=2:dmode=5:aw=1:qw=1',
                   'plug:base=0:sf=0.25:dg=2','plug:base=1:sf=0.25:dg=2',
                   'plug:base=2:sf=0.25:dg=2']
        for method in methods:
            a = dispatch(method,self.graph,copy.deepcopy(p))
            b = dispatch(method,changed,copy.deepcopy(p))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(bool(torch.isfinite(a).all()))
            torch.testing.assert_close(a.sum(1),torch.ones(self.graph['n']))
        w = core.module_weights(self.graph,p,sf=0.25,dg=2)
        self.assertTrue(bool(((w > 0) & (w <= 1)).all()))


if __name__ == '__main__':
    unittest.main()
