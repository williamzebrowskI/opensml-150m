import unittest
from sft.complete_transfer_1024.evaluate import grade
from sft.complete_transfer_1024.data import transform
class CompletionTests(unittest.TestCase):
 def setUp(self):
  self.base=dict(id='example',group='example',answer='The blue notebook belongs to Mira and stays on the desk. The red folder belongs to Theo and stays in the drawer. Both items will be collected by their owners after lunch.')
 def test_all_formats(self):
  for i in range(3):
   row=transform(self.base,i);self.assertTrue(grade(row,row['answer'],'end')['passed'])
 def test_missing_content_fails(self):
  row=transform(self.base,0);self.assertFalse(grade(row,row['answer'].splitlines()[0],'end')['passed'])
 def test_changed_fact_fails(self):
  row=transform(self.base,0);self.assertFalse(grade(row,row['answer'].replace('Mira','Theo'),'end')['passed'])
 def test_extra_content_fails(self):
  row=transform(self.base,1);self.assertFalse(grade(row,row['answer']+'\n\nThank you.','end')['passed'])
 def test_truncation_fails(self):
  row=transform(self.base,2);self.assertFalse(grade(row,row['answer'],'length')['passed'])
 def test_wrong_order_fails(self):
  row=transform(self.base,0);self.assertFalse(grade(row,'\n'.join(reversed(row['answer'].splitlines())),'end')['passed'])
if __name__=='__main__':unittest.main()
