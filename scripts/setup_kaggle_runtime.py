"""Isolated managed Python 3.12.15 and PyTorch cu124 on Kaggle only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.request
import uuid

UV_VERSION='0.13.0'
PYTHON_VERSION='3.12.15'
UV_URL='https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz'
UV_ARCHIVE_BYTES=17380567
UV_ARCHIVE_SHA256='1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154'
UV_BINARY_BYTES=39714040
UV_BINARY_SHA256='3e801df892439f5cde65d8b51ea69b69abff199ef37cc3cae4481be15d592df0'
OWNER={'schema_version':1,'purpose':'nlu-authorship-kaggle-runtime'}


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def scoped_path(path):
    path=Path(path).absolute()
    resolved=path.resolve()
    if not any(resolved!=root and resolved.is_relative_to(root) for root in (Path('/tmp'),Path('/kaggle/working'))):
        raise ValueError('Runtime paths must be children of /tmp or /kaggle/working')
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Runtime paths cannot pass through symbolic links')
    return resolved


def require_kaggle():
    if platform.system()!='Linux' or platform.machine()!='x86_64' or not Path('/kaggle/working').is_dir() or not (
        os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or Path('/kaggle/input').is_dir()):
        raise RuntimeError('GPU setup is restricted to an actual Linux x86-64 Kaggle kernel')


def child_environment(runtime):
    env=os.environ.copy()
    for key in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','VIRTUAL_ENV','CUDA_HOME','CUDA_PATH','TF_USE_LEGACY_KERAS','XLA_FLAGS'):
        env.pop(key,None)
    for key in list(env):
        if key.startswith('UV_') or (key.startswith('PIP_') and key not in {'PIP_CERT','PIP_CLIENT_CERT','PIP_PROXY','PIP_TIMEOUT','PIP_DEFAULT_TIMEOUT','PIP_RETRIES'}):
            env.pop(key,None)
    temporary=runtime/'temp';temporary.mkdir(exist_ok=True)
    env.update(TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),PYTHONNOUSERSITE='1',PYTHONUTF8='1',
               PIP_CONFIG_FILE=os.devnull,UV_PYTHON_INSTALL_DIR=str(runtime/'python'),UV_CACHE_DIR=str(runtime/'uv-cache'),
               MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='2'
    # Assigned CUDA visibility is preserved, never replaced by a local GPU choice.
    return env


def install_uv(runtime):
    directory=scoped_path(runtime/'uv-tools'/UV_VERSION);directory.mkdir(parents=True,exist_ok=True)
    binary=directory/'uv'
    if binary.is_file() and not binary.is_symlink() and binary.stat().st_size==UV_BINARY_BYTES and sha256_file(binary)==UV_BINARY_SHA256:
        binary.chmod(0o700);return binary
    archive=directory/'uv.tar.gz';temporary=directory/(uuid.uuid4().hex+'.part')
    digest=hashlib.sha256();size=0
    try:
        request=urllib.request.Request(UV_URL,headers={'User-Agent':'NLU-Authorship-Kaggle/2.0'})
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('xb') as stream:
            if not response.geturl().startswith('https://'):raise ValueError('uv redirected away from HTTPS')
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>UV_ARCHIVE_BYTES:raise ValueError('uv archive exceeded its verified release size')
                digest.update(block);stream.write(block)
        if size!=UV_ARCHIVE_BYTES or digest.hexdigest()!=UV_ARCHIVE_SHA256:
            raise ValueError('uv archive checksum failed; it was not executed')
        temporary.replace(archive)
        with tarfile.open(archive,'r:gz') as bundle:
            member=bundle.getmember('uv-x86_64-unknown-linux-gnu/uv')
            if not member.isfile() or member.size!=UV_BINARY_BYTES:raise ValueError('uv executable is not the expected regular archive member')
            source=bundle.extractfile(member)
            if source is None:raise ValueError('Missing uv executable')
            data=source.read(UV_BINARY_BYTES+1)
        if len(data)!=UV_BINARY_BYTES or hashlib.sha256(data).hexdigest()!=UV_BINARY_SHA256:
            raise ValueError('uv executable checksum failed; it was not executed')
        pending=directory/(uuid.uuid4().hex+'.part');pending.write_bytes(data);pending.chmod(0o700);pending.replace(binary)
        return binary
    finally:temporary.unlink(missing_ok=True)


PROBE=r'''
import importlib.metadata as m,json,pathlib,sys
problems=[];versions={}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 line=line.strip()
 if not line or line.startswith('#'):continue
 name,wanted=line.split('==',1)
 try:
  actual=m.version(name);versions[name]=actual
  if actual!=wanted:problems.append(name+' version mismatch')
 except m.PackageNotFoundError:problems.append(name+' missing')
for name,wanted in {'torch':'2.5.1+cu124','nlu-authorship-verification':'2.0.0'}.items():
 try:
  actual=m.version(name);versions[name]=actual
  if actual!=wanted:problems.append(name+' version mismatch')
 except m.PackageNotFoundError:problems.append(name+' missing')
if sys.version_info[:3]!=(3,12,15) or sys.prefix==sys.base_prefix:problems.append('Managed isolated Python 3.12.15 is required')
cfg=pathlib.Path(sys.prefix)/'pyvenv.cfg'
if not cfg.is_file() or 'include-system-site-packages = true' in cfg.read_text().lower():problems.append('System site packages must be disabled')
print(json.dumps({'healthy':not problems,'python':sys.version.split()[0],'versions':versions,'problems':problems}))
sys.exit(bool(problems))
'''

GPU_PROBE=r'''
import json,torch
if not torch.cuda.is_available() or torch.version.cuda!='12.4':raise RuntimeError('Isolated PyTorch cu124 found no assigned Kaggle GPU')
with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
 a=torch.tensor([[1.,2.]],device='cuda:0');b=torch.tensor([[3.],[4.]],device='cuda:0');value=a@b
if value.device.type!='cuda' or float(value.item())!=11.:raise RuntimeError('Kaggle FP16 GPU matmul failed')
print(json.dumps({'gpu_count':torch.cuda.device_count(),'gpus':[{'name':torch.cuda.get_device_name(i),'compute_capability':list(torch.cuda.get_device_capability(i)),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],'gpu_smoke_result':11.,'training_gpu_index':0,'cuda_build':torch.version.cuda}))
'''


def probe(python,project,env):
    if not python.is_file():return {'healthy':False,'problems':['Interpreter missing']}
    try:
        result=subprocess.run([str(python),'-I','-c',PROBE,str(project/'requirements-training.txt')],env=env,cwd=project,capture_output=True,text=True,timeout=90)
        return json.loads(result.stdout)
    except (OSError,ValueError,subprocess.TimeoutExpired):return {'healthy':False,'problems':['Dependency probe failed']}


def setup(project,runtime):
    require_kaggle();project=Path(project).resolve(strict=True)
    if not (project/'requirements-training.txt').is_file() or not (project/'pyproject.toml').is_file():
        raise ValueError('Project pinned training requirements and package metadata are required')
    runtime=scoped_path(runtime);marker=runtime/'runtime-owner.json'
    if runtime.exists():
        if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text())!=OWNER:
            raise ValueError('Existing runtime belongs to another task; choose a different path')
    else:
        runtime.mkdir(parents=True);marker.write_text(json.dumps(OWNER),encoding='utf8')
    env=child_environment(runtime);uv=install_uv(runtime)
    subprocess.run([str(uv),'--no-config','python','install','--no-bin',PYTHON_VERSION],cwd=project,env=env,check=True)
    result=subprocess.run([str(uv),'--no-config','python','find','--managed-python',PYTHON_VERSION],cwd=project,env=env,capture_output=True,text=True,check=True)
    managed=Path(result.stdout.strip()).resolve(strict=True)
    if not managed.is_relative_to((runtime/'python').resolve()):raise ValueError('Managed Python resolved outside this task runtime')
    venv=scoped_path(runtime/'.venv');python=venv/'bin/python'
    status=probe(python,project,env)
    if not status['healthy']:
        if venv.exists():venv.rename(runtime/('venv-backup-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]))
        subprocess.run([str(managed),'-I','-m','venv',str(venv)],cwd=project,env=env,check=True)
        pip=[str(python),'-I','-m','pip','--disable-pip-version-check','--no-cache-dir']
        subprocess.run(pip+['install','--index-url','https://download.pytorch.org/whl/cu124','--extra-index-url','https://pypi.org/simple','torch==2.5.1+cu124'],cwd=project,env=env,check=True)
        subprocess.run(pip+['install','--index-url','https://pypi.org/simple','-r',str(project/'requirements-training.txt')],cwd=project,env=env,check=True)
        subprocess.run(pip+['install','--no-deps','--no-build-isolation','-e',str(project)],cwd=project,env=env,check=True)
        subprocess.run([str(python),'-I','-m','pip','check'],cwd=project,env=env,check=True)
        status=probe(python,project,env)
        if not status['healthy']:raise RuntimeError('Isolated Kaggle dependency validation failed: '+ '; '.join(status['problems']))
    result=subprocess.run([str(python),'-I','-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],env=env,capture_output=True,text=True,check=True)
    libraries=sorted((Path(result.stdout.strip())/'nvidia').glob('*/lib'))
    env['LD_LIBRARY_PATH']=':'.join([str(p) for p in libraries]+([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    gpu=subprocess.run([str(python),'-I','-c',GPU_PROBE],env=env,cwd=project,capture_output=True,text=True,timeout=120)
    if gpu.returncode:raise RuntimeError('Isolated GPU smoke test failed: '+gpu.stderr[-3000:])
    metadata={'schema_version':1,'purpose':OWNER['purpose'],'python':status['python'],'host_python':sys.version.split()[0],
              'versions':status['versions'],'isolated':True,'system_site_packages':False,'uv_version':UV_VERSION,
              'uv_archive_sha256':UV_ARCHIVE_SHA256,'uv_binary_sha256':UV_BINARY_SHA256,
              'requirements_sha256':sha256_file(project/'requirements-training.txt'),'local_gpu_used':False,**json.loads(gpu.stdout)}
    return python,env,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runtime',type=Path,default=Path('/tmp/nlu-authorship-runtime'))
    parser.add_argument('--metadata-file',type=Path)
    parser.add_argument('--setup-only',action='store_true')
    parser.add_argument('python_arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    try:
        python,env,metadata=setup(args.project_root,args.runtime)
        if args.metadata_file:
            path=scoped_path(args.metadata_file);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
        print(json.dumps(metadata),flush=True)
        command=args.python_arguments
        if command[:1]==['--']:command=command[1:]
        if args.setup_only or not command:return 0
        return subprocess.run([str(python),'-I','-X','utf8',*command],cwd=args.project_root,env=env).returncode
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.CalledProcessError,subprocess.TimeoutExpired,tarfile.TarError) as error:
        print(json.dumps({'ready':False,'error':str(error)}),file=sys.stderr,flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
