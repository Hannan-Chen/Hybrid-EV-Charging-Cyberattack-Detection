import unittest
import numpy as np
from data_protocol import Dataset
from protocol import exact_threshold, evaluate, registry, paired_bank, members


class ProtocolTests(unittest.TestCase):
    def data(self, normal_count=382, updates=18, attacks=20):
        count = normal_count+attacks
        y = np.r_[np.zeros(normal_count, int), np.ones(attacks, int)]
        update = np.zeros(count, bool)
        update[normal_count-updates:normal_count] = True
        x = np.zeros((count, 11)); x[:, 10] = update | (y == 1)
        return Dataset(x,y,y,np.arange(count),np.arange(count),np.where(y,"Type-1","Normal"),update,np.arange(count))

    def test_threshold_discrete(self):
        d = self.data().take(np.arange(382))
        s = np.arange(382, dtype=float)
        t = exact_threshold(d,s)
        self.assertEqual(int((s[d.update]>=t).sum()),1)

    def test_ties_saturation(self):
        d = self.data().take(np.arange(382))
        t = exact_threshold(d,np.ones(382))
        self.assertGreater(t,1.)

    def test_every_change_balanced_accuracy(self):
        d = self.data();m=evaluate(d,d.x[:,10],.5,"Type-1")
        self.assertEqual(m["change_balanced_accuracy"],.5)
        self.assertEqual(m["update_fp"],18)

    def test_perfect_binary_macro(self):
        d=self.data();m=evaluate(d,d.y,.5,"Type-1")
        self.assertEqual(m["change_class_macro_f1"],1.)

    def test_registry_budgets(self):
        counts={f:sum(c["family"]==f for c in registry()) for f in ["RF","SDTN","SS","DeepSAD"]}
        self.assertEqual(counts,{"RF":4,"SDTN":4,"SS":8,"DeepSAD":8})

    def test_mechanism_equal_variant_weight(self):
        d=self.data(attacks=20)
        d.setting=d.setting.astype(object)
        d.setting[-1]="Type-1-ta"
        score=np.zeros(len(d.y));score[-1]=1
        first=evaluate(d,score,.5,"Type-1")
        second=evaluate(d,score,.5,"Type-1-ta")
        together=evaluate(d,score,.5,"Mechanism-1")
        self.assertEqual(together["recall"],.5)
        self.assertAlmostEqual(together["change_class_macro_f1"],.5*(first["change_class_macro_f1"]+second["change_class_macro_f1"]))
        self.assertEqual(together["tp"],1)
        self.assertEqual(together["fn"],19)

    def test_paired_bank_full_union_and_nested_balance(self):
        class Fake:
            source=np.arange(100)
            eligible={"Type-1":np.arange(90),"Type-1-ta":np.arange(40,100)}
        bank=Fake();allkeys=paired_bank(bank,list(range(100)),"Mechanism-1",66)
        self.assertEqual(len(allkeys),100)
        self.assertEqual(len({int(k.split(":")[1]) for k in allkeys}),100)
        for n in [2,5,10,20,50]:
            count=[sum(k.startswith(v+":") for k in allkeys[:n]) for v in members("Mechanism-1")]
            self.assertLessEqual(abs(count[0]-count[1]),1)

    def test_imbalanced_all_keeps_extra_unique_sources(self):
        class Fake:
            source=np.arange(100)
            eligible={"Type-1":np.arange(100),"Type-1-ta":np.arange(30)}
        bank=Fake();keys=paired_bank(bank,list(range(100)),"Mechanism-1",166)
        self.assertEqual(len(keys),100)
        self.assertEqual(sum(k.startswith("Type-1-ta:") for k in keys),30)


if __name__ == "__main__":
    unittest.main()
