"""Verify the outer input adapter without a Kaggle session or training."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.package_kaggle import render_wrapper
from scripts import package_kaggle

PROJECT='NLU-Authorship-Verification'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name).resolve()
        self.files={'README.md':b'Frozen source fixture\n','nlu_authorship/models.py':b'VALUE = 42\n',
                    'configs/experiment-plan.json':b'{"model": "roberta-large"}\n'}
        self.record={'files':{name:{'bytes':len(value),'sha256':digest(value)} for name,value in self.files.items()}}

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, extra=None, omitted=None):
        inputs=self.root/'zip-input';inputs.mkdir()
        path=inputs/'nlu-source.zip'
        with zipfile.ZipFile(path,'w') as bundle:
            for name,value in self.files.items():
                if name!=omitted:bundle.writestr(PROJECT+'/'+name,value)
            if extra is not None:bundle.writestr(*extra)
        self.record.update(archive_bytes=path.stat().st_size,archive_sha256=digest(path.read_bytes()))
        return inputs

    def expanded(self, name='expanded-input'):
        inputs=self.root/name;project=inputs/PROJECT
        for relative,value in self.files.items():
            path=project/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        self.record.setdefault('archive_bytes',1);self.record.setdefault('archive_sha256','0'*64)
        return inputs,project

    def module(self, profile_only=False, throughput_profile=False):
        text=render_wrapper(self.record,profile_only=profile_only,throughput_profile=throughput_profile)
        compile(text,'generated-kaggle-wrapper','exec')
        path=self.root/('profile.py' if profile_only else 'full.py');path.write_text(text,encoding='utf8')
        spec=importlib.util.spec_from_file_location('test_generated_wrapper',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def prepare(self,module,inputs,name):
        with redirect_stdout(io.StringIO()):return module.prepare_source(inputs,self.root/name)

    def fixture_model(self, name='saved-stylometric'):
        from nlu_authorship.models import write_artifact_inventory
        model=self.root/name;model.mkdir()
        for relative in package_kaggle._STYLOMETRIC_REQUIRED:
            (model/relative).write_bytes((relative+' local CPU artifact fixture\n').encode())
        write_artifact_inventory(model)
        return model

    def model_archive(self, model):
        _,files=package_kaggle._verified_stylometric_files(model)
        inputs=self.root/'model-zip-input';inputs.mkdir()
        archive=inputs/'stylometric-model.zip'
        with zipfile.ZipFile(archive,'w') as bundle:
            for name in files:bundle.write(model/name,'NLU-Stylometric-Model/'+name)
        self.record['stylometric_model']={'files':files,'archive_bytes':archive.stat().st_size,
                                         'archive_sha256':digest(archive.read_bytes())}
        return inputs

    def fixture_project(self):
        from nlu_authorship.data import build_split_manifest,save_manifest
        project=self.root/'temporary-project';(project/'data').mkdir(parents=True)
        for source,count,offset in [('train',40,0),('dev',8,40)]:
            with (project/'data'/f'{source}.csv').open('w',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['text_1','text_2','label']);writer.writeheader()
                for i in range(offset,offset+count):
                    writer.writerow({'text_1':f'First passage number {i} walks along a river in the morning.',
                                     'text_2':f'Second passage number {i} reads a book beside the window.',
                                     'label':i%2})
        files={'README.md':'Local package fixture\n','nlu_authorship/__init__.py':'VERSION=1\n',
               'scripts/setup_kaggle_runtime.py':'print("fixture only")\n',
               'configs/experiment-plan.json':'{"model":"roberta-large"}\n',
               'pyproject.toml':'[project]\nname="fixture"\n',
               'models/old.safetensors':'exclude historical weights',
               '.kaggle/credentials.json':'exclude fake credential fixture',
               'data/test.csv':'exclude unlabeled test', 'personal.log':'exclude personal log'}
        for name,value in files.items():
            p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf8')
        save_manifest(build_split_manifest(project),project/'reports/split-manifest.json')
        return project

    def test_zip_and_expanded_paths_verify_and_copy_exact_same_inventory(self):
        zipped=self.archive();module=self.module()
        first,receipt=self.prepare(module,zipped,'zip-working')
        self.assertEqual(receipt['input_kind'],'zip')
        expanded,source=self.expanded()
        (source/'unlisted-personal-log.txt').write_text('must never be copied')
        second,receipt=self.prepare(module,expanded,'expanded-working')
        self.assertEqual(receipt['input_kind'],'expanded')
        for name,value in self.files.items():
            self.assertEqual((first/name).read_bytes(),value)
            self.assertEqual((second/name).read_bytes(),value)
        self.assertFalse((second/'unlisted-personal-log.txt').exists())

    def test_corrupted_source_is_rejected_before_working_copy(self):
        inputs,source=self.expanded();module=self.module()
        (source/'nlu_authorship/models.py').write_bytes(b'VALUE = 99\n')
        with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
            self.prepare(module,inputs,'corrupt-working')
        self.assertFalse((self.root/'corrupt-working').exists())

    def test_two_valid_representations_are_ambiguous(self):
        inputs=self.archive()
        for name,value in self.files.items():
            path=inputs/PROJECT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        with self.assertRaisesRegex(RuntimeError,'found 2'):
            self.prepare(self.module(),inputs,'ambiguous-working')

    def test_extra_zip_file_is_rejected(self):
        inputs=self.archive(extra=(PROJECT+'/private.txt',b'not declared'))
        with self.assertRaisesRegex(RuntimeError,'unlisted'):
            self.prepare(self.module(),inputs,'extra-working')

    def test_missing_declared_zip_file_is_rejected(self):
        inputs=self.archive(omitted='nlu_authorship/models.py')
        with self.assertRaisesRegex(RuntimeError,'missing declared'):
            self.prepare(self.module(),inputs,'missing-working')

    def test_symlink_archive_member_is_rejected(self):
        entry=zipfile.ZipInfo(PROJECT+'/shortcut')
        entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16
        inputs=self.archive(extra=(entry,b'nlu_authorship/models.py'))
        with self.assertRaisesRegex(RuntimeError,'symbolic links'):
            self.prepare(self.module(),inputs,'symlink-working')

    def test_unsafe_inventory_paths_cannot_be_rendered(self):
        self.expanded()
        for path in ('../outside','/absolute','C:/drive','folder\\outside','folder//name','folder/./name'):
            bad={**self.record,'files':{path:{'bytes':1,'sha256':'0'*64}}}
            with self.assertRaisesRegex(ValueError,'Unsafe'):render_wrapper(bad)

    def test_profile_worker_flag_excludes_full_release_flag(self):
        self.expanded()
        for profile,throughput in ((False,False),(True,False),(False,True)):
            module=self.module(profile,throughput)
            with patch.object(module,'prepare_source',return_value=(self.root,{})),patch.object(module.subprocess,'run') as run,redirect_stdout(io.StringIO()):
                run.return_value.returncode=0
                module.main()
            command=run.call_args.args[0]
            self.assertEqual('--profile-only' in command,profile)
            self.assertEqual('--throughput-profile' in command,throughput)
            self.assertEqual('--complete-release' in command,not (profile or throughput))
            self.assertEqual(module.PROFILE_ONLY,profile or throughput)
            self.assertEqual(module.THROUGHPUT_PROFILE,throughput)

    def test_public_packager_uses_new_fixture_inventory_and_profile_metadata(self):
        project=self.fixture_project()
        output=self.root/'new-profile-package'
        with patch.object(package_kaggle,'ROOT',project):
            record=package_kaggle.package(output,'fixture-owner/data','fixture-owner/profile',profile_only=True)
            with self.assertRaises(FileExistsError):
                package_kaggle.package(output,'fixture-owner/data','fixture-owner/profile',profile_only=True)
        self.assertTrue(record['profile_only'])
        self.assertFalse(set(record['files']) & {'models/old.safetensors','.kaggle/credentials.json','data/test.csv','personal.log'})
        metadata=json.loads((output/'kernel/kernel-metadata.json').read_text())
        self.assertIn('profile',metadata['title'].lower())
        self.assertEqual(metadata['title'].replace(' ','-'),metadata['id'].rsplit('/',1)[-1])
        generated=(output/'kernel/run-experiment.py').read_text()
        compile(generated,'generated-full-packager-fixture','exec')
        spec=importlib.util.spec_from_file_location('actual_packaged_profile',output/'kernel/run-experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.EXPECTED_FILES,record['files'])
        with redirect_stdout(io.StringIO()):
            copied,receipt=module.prepare_source(output/'dataset',self.root/'profile-fixture-working')
        self.assertEqual(receipt['files'],len(record['files']))
        self.assertTrue(receipt['profile_only'])
        self.assertEqual((copied/'data/train.csv').read_bytes(),(project/'data/train.csv').read_bytes())

    def test_saved_cpu_model_zip_and_expanded_mount_copy_identical_allowlist(self):
        self.expanded();model=self.fixture_model();inputs=self.model_archive(model)
        module=self.module()
        with redirect_stdout(io.StringIO()):
            zipped,receipt=module.prepare_stylometric_model(inputs,self.root/'model-zip-working')
        self.assertEqual(receipt['bundle_kind'],'stylometric-model')
        self.assertEqual(receipt['input_kind'],'zip')
        expanded=self.root/'model-expanded-input';saved=expanded/'NLU-Stylometric-Model'
        saved.mkdir(parents=True)
        for path in model.iterdir():(saved/path.name).write_bytes(path.read_bytes())
        (saved/'private-log.txt').write_text('unlisted file must not be copied')
        (saved/'historical.safetensors').write_bytes(b'unlisted historical transformer')
        with redirect_stdout(io.StringIO()):
            copied,receipt=module.prepare_stylometric_model(expanded,self.root/'model-expanded-working')
        self.assertEqual(receipt['input_kind'],'expanded')
        declared=set(self.record['stylometric_model']['files'])
        self.assertEqual({p.name for p in copied.iterdir()},declared)
        for name in declared:self.assertEqual((copied/name).read_bytes(),(zipped/name).read_bytes())

    def test_expanded_model_hash_tampering_is_rejected_before_copy(self):
        self.expanded();model=self.fixture_model();self.model_archive(model)
        inputs=self.root/'tampered-model';saved=inputs/'NLU-Stylometric-Model';saved.mkdir(parents=True)
        for path in model.iterdir():(saved/path.name).write_bytes(path.read_bytes())
        original=(saved/'lgb_model.pkl').read_bytes()
        (saved/'lgb_model.pkl').write_bytes(bytes([original[0]^1])+original[1:])
        module=self.module()
        with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
            module.prepare_stylometric_model(inputs,self.root/'tampered-model-working')
        self.assertFalse((self.root/'tampered-model-working').exists())

    def test_actual_package_transport_verifies_model_and_passes_worker_directory(self):
        project=self.fixture_project();model=self.fixture_model();output=self.root/'full-package'
        with patch.object(package_kaggle,'ROOT',project):
            record=package_kaggle.package(output,'fixture-owner/data','fixture-owner/full',stylometric_model=model)
        bundled=record['stylometric_model'];self.record=record;module=self.module()
        self.assertTrue(bundled['artifact_inventory_verified'])
        self.assertEqual(set(bundled['files']),set(package_kaggle._STYLOMETRIC_REQUIRED)|{'artifact-inventory.json'})
        self.assertNotIn('models/old.safetensors',record['files'])
        with redirect_stdout(io.StringIO()):
            copied,receipt=module.prepare_stylometric_model(output/'dataset',self.root/'actual-model-working')
        self.assertEqual(receipt['files'],7)
        for name in bundled['files']:self.assertEqual((copied/name).read_bytes(),(model/name).read_bytes())
        worker_model=Path('/kaggle/working/nlu-stylometric-model/NLU-Stylometric-Model')
        with patch.object(module,'prepare_source',return_value=(self.root,{})),patch.object(module,'prepare_stylometric_model',return_value=(worker_model,{})),patch.object(module.subprocess,'run') as run,redirect_stdout(io.StringIO()):
            run.return_value.returncode=0;module.main()
        command=run.call_args.args[0]
        self.assertEqual(command[command.index('--stylometric-model-dir')+1],str(worker_model))
        self.assertIn('--complete-release',command)
        self.assertNotIn('--profile-only',command)

    def test_profile_rejects_model_before_any_output_is_created(self):
        output=self.root/'invalid-profile'
        with self.assertRaisesRegex(ValueError,'profile-only'):
            package_kaggle.package(output,'fixture-owner/data','fixture-owner/profile',profile_only=True,stylometric_model=self.root/'absent')
        self.assertFalse(output.exists())
        self.expanded();model=self.fixture_model();self.model_archive(model)
        with self.assertRaisesRegex(ValueError,'profile-only'):render_wrapper(self.record,profile_only=True)

    def test_saved_model_corruption_or_transformer_weights_cannot_be_packaged(self):
        from nlu_authorship.models import write_artifact_inventory
        model=self.fixture_model();target=model/'xgb_model.json';data=target.read_bytes()
        target.write_bytes(bytes([data[0]^1])+data[1:]);output=self.root/'corrupt-package'
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            package_kaggle.package(output,'fixture-owner/data','fixture-owner/full',stylometric_model=model)
        self.assertFalse(output.exists())
        target.write_bytes(data);(model/'model.safetensors').write_bytes(b'historical transformer weights')
        write_artifact_inventory(model)
        with self.assertRaisesRegex(ValueError,'canonical stylometric'):
            package_kaggle.package(output,'fixture-owner/data','fixture-owner/full',stylometric_model=model)
        self.assertFalse(output.exists())

    def test_kernel_title_keeps_requested_large_profile_slug(self):
        project=self.fixture_project();output=self.root/'requested-profile-package'
        kernel_id='fixture-owner/nlu-large-t4-capability-20261012'
        with patch.object(package_kaggle,'ROOT',project):
            package_kaggle.package(output,'fixture-owner/data',kernel_id,profile_only=True)
        metadata=json.loads((output/'kernel/kernel-metadata.json').read_text())
        self.assertEqual(metadata['title'],'nlu large t4 capability 20261012')
        self.assertEqual(metadata['title'].replace(' ','-'),kernel_id.rsplit('/',1)[-1])

    def test_throughput_package_is_source_only_and_keeps_requested_mode(self):
        project=self.fixture_project();output=self.root/'throughput-package'
        with patch.object(package_kaggle,'ROOT',project):
            record=package_kaggle.package(output,'fixture-owner/data','fixture-owner/nlu-large-throughput',throughput_profile=True)
        self.assertTrue(record['profile_only']);self.assertTrue(record['throughput_profile'])
        self.assertEqual(record['profile_mode'],'throughput')
        self.assertNotIn('stylometric_model',record)
        self.assertFalse((output/'dataset/stylometric-model.zip').exists())
        script=output/'kernel/run-experiment.py'
        compile(script.read_text(encoding='utf8'),'throughput-packaged-wrapper','exec')
        spec=importlib.util.spec_from_file_location('actual_packaged_throughput',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(module.PROFILE_ONLY);self.assertTrue(module.THROUGHPUT_PROFILE)
        self.assertIsNone(module.MODEL_BUNDLE)
        with patch.object(module,'prepare_source',return_value=(self.root,{})),patch.object(module.subprocess,'run') as run,redirect_stdout(io.StringIO()):
            run.return_value.returncode=0;module.main()
        command=run.call_args.args[0]
        self.assertIn('--throughput-profile',command)
        for forbidden in ('--profile-only','--complete-release','--stylometric-model-dir'):
            self.assertNotIn(forbidden,command)

    def test_throughput_rejects_cpu_model_and_both_profile_modes_before_output(self):
        output=self.root/'invalid-throughput'
        for kwargs in ({'throughput_profile':True,'stylometric_model':self.root/'absent'},
                       {'throughput_profile':True,'profile_only':True}):
            with self.assertRaises(ValueError):
                package_kaggle.package(output,'fixture-owner/data','fixture-owner/throughput',**kwargs)
            self.assertFalse(output.exists())
        self.expanded();model=self.fixture_model();self.model_archive(model)
        with self.assertRaisesRegex(ValueError,'profile-only'):render_wrapper(self.record,throughput_profile=True)
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            render_wrapper(self.record,profile_only=True,throughput_profile=True)
        arguments=['package_kaggle.py','--output',str(output),'--dataset-id','fixture-owner/data',
                   '--kernel-id','fixture-owner/throughput','--profile-only','--throughput-profile']
        with patch.object(package_kaggle,'package') as packaging,patch('sys.argv',arguments),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
            package_kaggle.main()
        self.assertEqual(error.exception.code,2);packaging.assert_not_called()


if __name__=='__main__':unittest.main()
