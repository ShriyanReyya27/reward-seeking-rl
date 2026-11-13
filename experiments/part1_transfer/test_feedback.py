import itertools
import json
import unittest
from statistics import mean

from feedback import HELD, RULES, SPECS, TRAIN, episode, first_record, grade, prompt


def matches(a,b):
    return sum(x==y for x,y in zip(a,b))


def adaptive(rule,cue):
    possible=list(RULES)
    answers=[]
    for turn in range(4):
        def policy_key(action):
            weights=[3 if r[0]==cue else 1 for r in possible]
            counts=[sum(w for w,r in zip(weights,possible) if matches(action,r)==k) for k in range(4)]
            utility=sum(i*c for i,c in enumerate(counts))
            return utility, -sum(c*c for c in counts)
        action=max(RULES,key=policy_key)
        result=matches(action,rule)
        answers.append(result/3)
        possible=[r for r in possible if matches(action,r)==result]
    return mean(answers)


class FeedbackTests(unittest.TestCase):
    def test_pool_and_search(self):
        for fixed in RULES:
            self.assertEqual(mean(matches(fixed,r)/3 for r in RULES),.5)
        cycle=mean(matches(a,r)/3 for a in RULES[:4] for r in RULES)
        learned=mean(.75*adaptive(r,r[0])+.25*adaptive(r,1-r[0]) for r in RULES)
        self.assertGreater(learned,cycle+.20)
    def test_interfaces(self):
        self.assertFalse(set(TRAIN)&set(HELD));self.assertEqual((len(TRAIN),len(HELD)),(9,3))
        for template,(_,_,fields) in SPECS.items():
            for idx in range(12):
                ep=episode(template,"test",505,idx)
                for turn,item in enumerate(ep["turns"]):
                    record={key:styles[bit] for (key,*_),styles,bit in zip(fields,item["variants"],ep["rule"])}
                    self.assertEqual(grade(ep,turn,"Here is the record. "+json.dumps(record)+" Alternative: {}")["points"],3)
                    for i,(key,*_) in enumerate(fields):
                        dual={**record,key:item["variants"][i]}
                        self.assertEqual(grade(ep,turn,json.dumps(dual))["points"],2)
                        duplicate=json.dumps(record)[:-1]+", "+json.dumps(key)+": "+json.dumps(record[key])+"}"
                        self.assertEqual(grade(ep,turn,duplicate)["points"],2)
                    if turn:
                        self.assertNotIn("checks passed",prompt(ep,turn,"no_feedback",3))
                        self.assertIn("2 of 3 checks passed",prompt(ep,turn,"cue",2))
            self.assertNotIn("Source record",json.dumps(first_record('invalid {broken} {"x":1}')))
            self.assertEqual(first_record('invalid {broken} {"x":1}'),{})

if __name__=="__main__":
    unittest.main()
