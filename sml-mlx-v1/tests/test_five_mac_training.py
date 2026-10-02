import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from sml_v1.common import file_sha256,fingerprint,read_json
from sml_v1.continuation import PRE_FIVE_RANK_CODE,check_resume,extend_recipe
from sml_v1.lr_scheduler import DEFAULTS,adaptive_recipe,restore_scheduler
from sml_v1.recipe import learning_rate,tokens_per_update,validate
from sml_v1.topology_transition import migrate_recipe


def saved():
    recipe=read_json(ROOT/'configs/pilot.json')
    metadata=dict(recipe=recipe,step=4696,tokens=500105216,best_val_loss=3.0,replica_hash='test-replica',
        v2_contract=dict(code=PRE_FIVE_RANK_CODE,recipe=fingerprint(recipe),data='data',tokenizer='tokenizer'),
        validation_fingerprint='fixed-heldout')
    recipe=extend_recipe(metadata,15000000000)
    metadata.update(recipe=recipe,step=50096,tokens=5335023616)
    metadata['v2_contract']['recipe']=fingerprint(recipe)
    recipe=adaptive_recipe(metadata,DEFAULTS)
    scheduler=restore_scheduler(recipe,metadata)
    metadata.update(recipe=recipe,lr_scheduler=scheduler.state_dict())
    metadata['v2_contract']['recipe']=fingerprint(recipe)
    return metadata


def contract(metadata,recipe):
    return dict(metadata['v2_contract'],recipe=fingerprint(recipe),
        code=fingerprint({p.name:file_sha256(p) for p in (ROOT/'sml_v1').glob('*.py')}))


class FiveMacTrainingTests(unittest.TestCase):
    def test_only_topology_and_batch_distribution_change(self):
        meta=saved();original=copy.deepcopy(meta)
        new=migrate_recipe(meta,[4,3,2,2,2])
        self.assertEqual(meta,original)
        self.assertEqual(tokens_per_update(new),106496)
        for key,value in meta['recipe'].items():
            if key!='batches':self.assertEqual(new[key],value,key)
        self.assertEqual(check_resume(meta,contract(meta,new),new),'explicit-five-rank-transition')
        self.assertEqual(restore_scheduler(new,meta).state_dict(),meta['lr_scheduler'])
        for t in [meta['tokens'],12000000000,15000000000]:
            self.assertEqual(learning_rate(t,new),learning_rate(t,meta['recipe']))

    def test_migrated_resume_keeps_geometry_and_scheduler(self):
        meta=saved();new=migrate_recipe(meta,[4,3,2,2,2])
        current=dict(meta,recipe=new,v2_contract=contract(meta,new))
        self.assertEqual(migrate_recipe(current,[4,3,2,2,2]),new)
        self.assertEqual(check_resume(current,current['v2_contract'],new),'unchanged')
        with self.assertRaises(ValueError):migrate_recipe(current,[3,4,2,2,2])

    def test_changed_model_schedule_provenance_and_data_rejected(self):
        meta=saved();new=migrate_recipe(meta,[4,3,2,2,2])
        for kind in ['model','lr','parent','data']:
            changed=copy.deepcopy(new)
            if kind=='model':changed['model']['n_layers']+=1
            if kind=='lr':changed['peak_lr']/=2
            if kind=='parent':changed['topology_transition']['parent_replica_hash']='wrong'
            proposed=contract(meta,changed)
            if kind=='data':proposed['data']='changed'
            with self.subTest(kind=kind),self.assertRaises(ValueError):check_resume(meta,proposed,changed)

    def test_invalid_sizes_order_and_token_accounting_rejected(self):
        meta=saved()
        for batches in [[4,3,3,3,3],[4,3,3,3],[0,7,2,2,2],[True,6,2,2,2]]:
            with self.subTest(batches=batches),self.assertRaises(ValueError):migrate_recipe(meta,batches)
        with self.assertRaises(ValueError):migrate_recipe(meta,[4,3,2,2,2],[0,1,2,3,4])
        meta['tokens']+=1
        with self.assertRaises(ValueError):migrate_recipe(meta,[4,3,2,2,2])

    def test_old_launcher_does_not_fall_back_to_local_for_five(self):
        from sml_v1.launch import main
        recipe=read_json(ROOT/'configs/pilot.json');recipe['batches']=[4,3,2,2,2]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(recipe))
            with patch('sys.argv',['launch','--config',str(p)]),patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):main()

    def test_new_launcher_plan_is_read_only(self):
        import launch_five_mac as launch
        with patch('sys.argv',['launch']),patch('sys.stdout',new_callable=io.StringIO) as out, \
                patch.object(launch,'select_checkpoint',return_value=Path('/tmp/source/latest.json')), \
                patch.object(launch,'verified_metadata',return_value=saved()):
            launch.main()
            self.assertIn('No model loaded',out.getvalue())

    def test_four_and_five_rank_slicing_consumes_identical_token_pool(self):
        import numpy as np
        from sml_v1.data import TokenPools
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            values=np.arange(20000,dtype=np.uint16)
            values.tofile(p/'a.bin');(p/'manifest.json').write_text('{}')
            manifest=dict(weights={'a':100},files=[dict(source='a',file='a.bin',split='train')])
            a=TokenPools(p,manifest,'train',16);a.batch(3)
            b=TokenPools(p,manifest,'train',16,a.state_dict())
            four=a.global_batch(4,[4,3,3,3]);five=b.global_batch(4,[4,3,2,2,2])
            np.testing.assert_array_equal(four,five)
            self.assertEqual(a.state_dict(),b.state_dict())
            slices=[];start=0
            for size in [4,3,2,2,2]:
                slices.append(five[:,:,start:start+size]);start+=size
            np.testing.assert_array_equal(np.concatenate(slices,axis=2),four)


if __name__=='__main__':unittest.main()
