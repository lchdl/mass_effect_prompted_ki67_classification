# defines some utility functions for reading/writing 
# nifti(*.nii)/pickle(*.pkl)/csv(*.csv) files
import os 
from shutil import copyfile, rmtree
import ntpath
import random
import shutil
import warnings
from glob import glob
from distutils.dir_util import copy_tree
import os, csv, gzip, tarfile, pickle, json, xlsxwriter, shutil, imageio, openpyxl
import nibabel as nib
import numpy as np
from typing import Union
from xlsxwriter.format import Format
from copy import deepcopy
from scipy.io import loadmat
from nibabel import processing as nibproc
import os
import subprocess
from time import sleep
from typing import Union
import shlex
from contextlib import contextmanager
import os, sys, time, datetime
import signal
import psutil
import warnings
from time import sleep
from typing import Callable, List
import multiprocessing
import traceback
from multiprocessing import Pool
import os, sys

def kill_process_tree(pid, kill_self=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess: # already killed or due to some other reasons
        return
    childs = parent.children(recursive=True)
    for child in childs:
        child.kill()
    if kill_self:
        parent.kill()

def print_ts(*args,**kwargs):
    ts = '['+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'] '
    if 'start' in kwargs:
        print(kwargs['start'],end='')
        kwargs.pop('start')
    print(ts,end='')
    print(*args,**kwargs)

def printi(*args):
    '''
    immediate print without '\\n'
    '''
    print(*args,end='')
    sys.stdout.flush()

def printx(msg):
    '''
    single line erasable output.
    '''
    assert isinstance(msg,str),'msg must be a string object.'
    
    try:
        columns = list(os.get_terminal_size())[0]
    except Exception:
        # get_terminal_size() failed,
        # probably due to invalid output device, then we ignore and return
        return
    except BaseException:
        raise

    outsize = columns-1

    print('\r' +' '*outsize + '\r',end='')
    print(msg[0:outsize],end='')
    sys.stdout.flush()

def printv(level, msg):
    '''
    print with verbose setting.
    '''
    need_print = False
    if 'verbose_level' in os.environ:
        required_level = int(os.environ['verbose_level'])
        need_print = True if level <= required_level else False
    else:
        need_print = False
    if need_print:
        print(msg)

def format_sec(t:int):
    s = t%60; t = t//60
    m = t%60; t = t//60
    h = t%24; t = t//24
    d = t
    if d>30: 
        # this task is too long... if you run a task that will execute for more than 30 days,
        # it seems more wisely to terminate the program and think about some other workarounds
        # to reduce the execution time :-)
        return '>30d' 
    else:
        if d>0: return '%dd%dh' % (d,h)
        elif h>0: return '%dh%dm' % (h,m)
        elif m>0: return '%dm%ds' % (m,s)
        else: return '%ds' % s

def minibar(msg=None,a=None,b=None,time=None,fill='=',length=15,last=None):
    if length<5: length=5
    perc = 0.0
    if b != 0:
        perc = a/b

    na=int((length-2)*perc)
    if na<0: na=0
    if na>length-2: na=length-2
    head = ('%s '%msg) if len(msg)>0 else ''
    perc = '%d%%' % int(100.0*perc)
    bar = '['+fill*na+'.'*(length-2-na)+']'
    pstart = 1+na-len(perc)
    if pstart<1: pstart=1
    pend = pstart+len(perc)
    bar = bar[:pstart] + perc + bar[pend:]

    time_est = ''
    if time is not None:
        if a == 0:
            elapsed = int(time)
            time_est = ' '+format_sec(elapsed)
        else:
            elapsed, remaining = int(time), int(time*(b-a)/a)
            time_est = ' '+format_sec(elapsed)
            time_est += '|'+format_sec(remaining)

    if last is None:
        last = ''

    printx(head+bar+time_est+' '+last)

# simple text logging
class SimpleTxtLog(object):
    def __init__(self, location):
        self.location = location
        with open(self.location, 'w') as f: pass
    
    def now(self):
        ts = '['+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+']'
        return ts

    def write(self, msg, timestamp = False, auto_newline=True):
        msg0 = msg if timestamp == False else self.now() + ' ' + msg
        if auto_newline:
            msg0 += '\n'
        with open(self.location, 'a') as f:
            f.write(msg0)
    
class Timer(object):
    def __init__(self, tick_now=True):
        self.time_start = time.time()
        self.time_end = 0
        if tick_now:
            self.tick()
        self.elapse_start = self.time_start
    def tick(self):
        self.time_end = time.time()
        dt = self.time_end - self.time_start
        self.time_start = self.time_end
        return dt
    def elapsed(self):
        self.elapse_end = time.time()
        return self.elapse_end - self.elapse_start
    def now(self, format = '%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.now().strftime(format)

class TimeStamps(object):
    def __init__(self):
        self.tstamps = {}

    def _format_now(self): # return a formatted time string
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return ts

    def record(self, name):
        self.tstamps[name] = self._format_now() # save

    def get(self, name):
        if name not in self.tstamps:
            return "unknown"
        else:
            return self.tstamps[name]

@contextmanager
def ignore_SIGINT():
    '''
    Description
    ------------
    Temporarily ignore keyboard interrupt signal (SIGINT).

    Usage
    ------------
    >>> with ignore_SIGINT():
    >>>     # do something here, SIGINT ignored
    >>>     # ...
    >>> # SIGINT no longer ignored here
    '''
    last_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    yield
    signal.signal(signal.SIGINT, last_handler) # restore default handler

@contextmanager
def ignore_print():
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    yield
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr

class Checkpoints(object):
    '''
    Example
    --------------
    >>> ckpts = Checkpoints('./ckpts/')
    >>> if not ckpts.is_finished('ckpt1'):
    >>>     # do some work
    >>>     ckpts.set_finish('ckpt1')
    '''
    def __init__(self, save_folder):
        self._save_folder = mkdir(save_folder)
        self._is_disabled = False

    def disable_all_checkpoints(self):
        self._is_disabled = True
    def enable_all_checkpoints(self):
        self._is_disabled = False

    def is_finished(self, ckpt_name):
        if self._is_disabled: # if all checkpoints are disabled then we just ignore all finished checkpoints
                             # this is useful for debugging
            return False
        if file_exist(join_path(self._save_folder, ckpt_name)):
            return True
        else:
            return False
    def set_finish(self, ckpt_name):
        with open(join_path(self._save_folder, ckpt_name), 'w') as f:
            pass # only create an empty file to indicate that the checkpoint is finished. 
    

def contain_duplicates(l: list):
    '''
    check if a list contains duplicated items.
    '''
    assert isinstance(l, list), 'object should be a list.'
    if len(l) != len(set(l)): return True
    else: return False

def remove_duplicates(l:list):
    return list(dict.fromkeys(l))


def remove_items(l,s):
    newl = []
    for item in l:
        print(item)
        if item not in s:
            newl.append(item)
        else:
            print('delete', item)
    return newl

# emulating shell command
def run_shell(command:str, print_command:bool=True, print_output:bool=True, force_continue:bool=False,
    env_vars: Union[dict, None] = None):
    '''
    Description
    ----------------
    run_shell: send command to shell by simply using a Python function call. 

    Paramaters
    ----------------
    command: str
        command that will be sent to shell for execution
    print_command: bool
        if you want to see what command is being executed 
        you can set print_command=True.
    print_output: bool 
        if you want to see outputs from subprocess you need
        to set this to True.
    force_continue: bool 
        if force_continue=True, then the main process will
        still continue even if error occurs in sub-process.
    env_vars: dict or None
        if set, this param will overload environment variables
        of the child process.

    Returns
    ----------------
    retcode: int
        return value from sub-process.
    '''
    if print_command:
        print(command)
    retcode = None
    stdout, stderr = None, None # default
    if print_output == False:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    
    # now start the process
    retcode = None
    try:
        p = None
        args = shlex.split(command)
        p = subprocess.Popen(args, shell=False, stdout=stdout, stderr=stderr, env=env_vars)
        retcode = p.wait()
    except BaseException as e:
        if p is not None:
            with ignore_SIGINT():
                kill_process_tree(p.pid, kill_self=True)
                sleep(3.0) # wait till all messes are cleaned up

        if not isinstance(e, Exception):
            # re-raise system error
            raise e

    # handling return values
    if retcode != 0:
        s=''
        if retcode == None: s = 'None'
        else: s = str(retcode)
        print('\n>>> ** Unexpected return value "%s" from command:\n' % s)
        print('>>>',command)
        if force_continue==False:
            print('\n>>> ** Process will exit now.\n')
            exit('Error occurred (code %s) when executing command:\n"%s"' % (s, command))

    return retcode

def try_shell(command: str, stdio = False):
    '''
    Try running a command and get its return value or stdio strings (stdout, stderr).
    '''
    retval = None
    strout, strerr = None, None
    try:
        p = None
        args = shlex.split(command)
        if stdio == False: # return code instead
            p = subprocess.Popen(args, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            retval = p.wait()
        else:
            p = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            strout, strerr = p.communicate()
    except BaseException as e:
        if p is not None:
            kill_process_tree(p.pid)
        if not isinstance(e, Exception):
            raise e
    if stdio:
        return strout.decode("utf-8"), strerr.decode("utf-8")
    else:
        return retval

def ls_tree(folder,depth=2,stat_size=False,closed=False):
    '''
    Description
    -------------
    ls_tree: similar to "ls" command in unix-based systems but can print file hierarchies and 
    display file sizes.

    Parameters
    -------------
    folder: str
        path to folder that wants to be displayed.
    depth: int
        display depth.
    stat_size: bool
        display file sizes.
    closed: bool
        display directory in open/closed style.
    '''

    def __get_size_in_bytes(start_path):
        '''
        from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
        '''
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def __size_format(size_in_bytes):
        if size_in_bytes < 1024:
            return '%dB' % size_in_bytes
        elif size_in_bytes >=1024 and size_in_bytes <=1024*1024-1:
            return '%dKB' % (size_in_bytes//1024)
        elif size_in_bytes >=1024*1024 and size_in_bytes <=1024*1024*1024-1:
            return '%.2fMB' % (size_in_bytes/(1024*1024))
        else:
            return '%.2fGB' % (size_in_bytes/(1024*1024*1024))

    def __abbr_filename(filename, length):
        if len(filename)<=length: return filename
        else:
            return '...'+filename[len(filename)-length+3:]

    def __max_item_len(l):
        maxl = 0
        for item in l:
            if len(item)> maxl:
                maxl = len(item)
        return maxl

    def __rreplace_once(s,subs,replace):
        return replace.join(s.rsplit(subs, 1))

    def __ls_ext_from(folder,indent,cur_depth,max_depth,max_len=24):
        # indent: spaces added in each indent level
        # folder: folder path in current search
        # max_len: maximum length of file name

        prespaces =  ('|' + ' '*(indent-1)) * cur_depth
        terminal_cols = os.get_terminal_size()[0]
        
        items_all = os.listdir(folder)
        items_f = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == True])
        items_d = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == False])

        abbr_items_f_repr = [ __abbr_filename(item,max_len-1)+' ' for item in items_f ]
        abbr_items_d_repr = [ __abbr_filename(item,max_len-1)+'/' for item in items_d ]

        if cur_depth == max_depth:
            max_repr_len = __max_item_len(abbr_items_f_repr + abbr_items_d_repr)

            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr + abbr_items_d_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0 and closed == False:
                print(repr_string)
        else:
            max_repr_len = __max_item_len(abbr_items_f_repr) if len(abbr_items_f_repr)>0 else max_len
            # print files
            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0:
                print(repr_string)

            # recursive print dirs
            for it, it_repr in zip(items_d, abbr_items_d_repr):
                if it_repr[-1] == '/': # is directory
                    fc = len(os.listdir(os.path.join(folder, it))) # number of files in directory
                    subs = '|' + ' '*(indent-1)
                    replace = '+' + '-'*(indent-1)
                    # replace "|  " to "+--"
                    prespaces2 = __rreplace_once(prespaces,subs,replace)
                    if stat_size:
                        size_str = __size_format(__get_size_in_bytes(os.path.join(folder, it)))
                        print( prespaces2 + it + ' +%d item(s)' % fc + ', ' + size_str)
                    else:
                        print( prespaces2 + it + ' +%d item(s)' % fc)
                    __ls_ext_from(os.path.join(folder, it),indent, cur_depth+1,max_depth,max_len=max_len)
                else:
                    print(prespaces + it_repr)

    # check if folder exists
    folder = os.path.abspath(folder)
    if os.path.exists(folder) and os.path.isdir(folder) == False:
        raise FileNotFoundError('folder not exist: "%s".' % os.path.abspath(folder))
    folder = folder + os.path.sep

    foldc = os.listdir(folder)
    if stat_size:
        size_str = __size_format(__get_size_in_bytes(folder))
        print('file(s) in "%s"  +%d item(s), %s' % (folder, len(foldc), size_str))
    else:
        print('file(s) in "%s"  +%d item(s)' % (folder, len(foldc)))
    if len(foldc) == 0: # empty root folder
        print('(folder is empty)')
        return
    else:
        __ls_ext_from(folder, 3, 1, max_depth=depth)


def find(pathname):
    glob(pathname, recursive=True) # accept "**" expression

def chmod(file:str, access:str):
    '''
    Description
    ------------
    Change access/permisson of a single file/directory.

    Usage
    ------------
    >>> chmod('/path/to/file', '755')
    '''
    access_oct = int(access,8)
    os.chmod(file, access_oct)

# list all files inside a directory and return as a list.
def laf(root_dir):
    l=list()
    for path, _, files in os.walk(root_dir):
        for name in files:
            l.append(os.path.abspath(os.path.join(path, name)))
    return l

def mv(src,dst):
    if os.path.exists(src):
        # cp(src,dst)
        # rm(src)
        shutil.move(src,dst)
    else:
        warnings.warn('file or folder "%s" does not exist.' % src)

def rm(file_or_dir):
    if os.path.exists(file_or_dir) == False: return
    if os.path.isfile(file_or_dir) == False:
        rmtree(file_or_dir)
    else:
        os.remove(file_or_dir)

def cd(path):
    os.chdir(path)

def cwd():
    return os.getcwd()

def cp(src,dst):
    '''
    copy a single file or an entire dir
    '''
    if file_exist(src): # copy a single file
        copyfile(src,dst)
    elif dir_exist(src): # copy an entire dir
        if dir_exist(dst) == False:
            mkdir(dst)
        copy_tree(src, dst)
    else:
        raise RuntimeError('file or dir not exist or have no access: "%s".' % src)

# make directory if not exist and returns newly created dir path
def mkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    return os.path.abspath(path)

def abs_path(path):
    return os.path.abspath(path)

def join_path(*args):
    path = os.path.join(*args)
    return os.path.abspath(path)

def file_exist(path:str):
    if os.path.exists(path) and os.path.isfile(path): return True
    else: return False

def file_empty(path:str):
    if file_exist(path) == False:
        raise RuntimeError('"%s" is not a file or not exist.' % path)
    fsize = os.stat(path).st_size
    if fsize == 0:
        return True
    else:
        return False

def files_exist(path_list:list):
    for path in path_list:
        if os.path.exists(path) and os.path.isfile(path):
            continue
        else: return False
    return True

def file_size(path:str): # return file size in bytes
    st = os.stat(path)
    bytes = st.st_size
    return bytes

def dir_exist(path):
    if os.path.exists(path) and os.path.isdir(path): return True
    else: return False

def ls(root_dir, full_path = False):
    if full_path == False:
        return os.listdir(root_dir)
    else:
        l = []
        for item in os.listdir(root_dir):
            l.append(join_path(root_dir, item))
        return l

def lsdir(root_dir, full_path=False):
    '''
    list all directories in a path
    '''
    dirs = [ item for item in os.listdir(root_dir) if dir_exist(join_path(root_dir, item))]
    if full_path == False:
        return dirs
    else:
        l=[]
        for folder in dirs:
            l.append( join_path(root_dir, folder) )
        return l

def lsfile(root_dir, full_path=False):
    '''
    list all files in a path
    '''
    dirs = [ item for item in os.listdir(root_dir) if file_exist(join_path(root_dir, item))]
    if full_path == False:
        return dirs
    else:
        l=[]
        for folder in dirs:
            l.append( join_path(root_dir, folder) )
        return l

# get filename from path
def gn(path, no_extension = False) -> str:
    name = ntpath.basename(os.path.abspath(path))
    if no_extension:
        index = name.find('.')
        name = name[:index]
    return name

# get file/dir directory from path
def gd(path):
    return os.path.abspath(os.path.dirname(os.path.abspath(path)))

def make_unique_dir(basedir=None):
    while True:
        randstr = ''.join(random.choice('0123456789abcdef') for _ in range(8))
        randstr = '__' + randstr + '__'
        if basedir is not None:
            dirpath = join_path(basedir,randstr)
        else:
            dirpath = abs_path(randstr) # current working directory
        #print('checking if dir %s exist.' % dirpath)
        if dir_exist(dirpath):
            #print('exist. change')
            continue
        else:
            mkdir(dirpath)
            return dirpath

def fsize(path):
    '''
    get file size in bytes
    '''
    return os.stat(path).st_size


# utility function used to compress a file into "*.gz"
# (not suitable for compressing folders)
def gz_compress(file_path, out_path=None, compress_level:int=9, verbose=False, overwrite = True):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_path), 'file "%s" not exist or it is a directory. '\
        'gzip can be only used to compress files, if you want to compress folder structure, '\
        'please use targz_compress(...) instead.' % file_path
    f = open(file_path,"rb")
    data = f.read()
    bindata = bytearray(data)
    gz_path = join_path( gd(file_path) , gn(file_path) + '.gz' ) if out_path is None else out_path
    if file_exist(gz_path) and not overwrite:
        if verbose:
            print('skip %s' % (file_path))
    else:
        if verbose:
            print('%s >>> %s' % (file_path, gz_path))
        with gzip.GzipFile(filename=gz_path, mode='wb', compresslevel=compress_level) as f:
            f.write(bindata)
    return gz_path

def gz_uncompress(gz_path, out_path=None, verbose=False):
    out_path0 = ''
    if out_path is not None:
        out_path0 = out_path
    else:
        if gz_path[-3:] == '.gz':
            out_path0 = gz_path[:-3]
        else:
            raise RuntimeError(
                'Incorrect gz file name. Input file name must '
                'end with "*.gz" if out_path is not set.')
    out_path0 = abs_path(out_path0)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path0, 'wb') as f_out:
            if verbose: print('%s >>> %s' % (gz_path, out_path))
            shutil.copyfileobj(f_in, f_out)

# compress a file or a folder structure into *.tar.gz format.
def targz_compress(file_or_dir_path, out_file=None, compress_level:int=9, verbose=False):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_or_dir_path) or dir_exist(file_or_dir_path), \
        'file or directory not exist: "%s".' % file_or_dir_path
    targz_path = join_path(gd(file_or_dir_path) , gn(file_or_dir_path) + '.tar.gz') if out_file is None else out_file
    if file_exist(file_or_dir_path):
        # target path is a file
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            tar.add(file_or_dir_path, arcname=gn(file_or_dir_path) )
            if verbose:
                print('>>> %s' % file_or_dir_path)
    elif dir_exist(file_or_dir_path):
        # target path is a folder
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            for name in os.listdir( file_or_dir_path ):
                tar.add( join_path(file_or_dir_path, name) , recursive=True, arcname=name)
                if verbose:
                    print('>>> %s' % name)
    else:
        raise RuntimeError('only file or folder can be compressed.')

def targz_uncompress(targz_file, out_path):
    '''
    out_path: str
        path to output folder.
    '''
    targz = tarfile.open(targz_file)
    targz.extractall(out_path)
    targz.close()


def load_csv_simple(file_path, key_names = None):
    '''
    load a csv file as python dict.
    '''
    parsed_dataset = {}

    if key_names is None:
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',',quotechar='"')
            for row in csv_reader:
                key_names = row
                break
        key_names = [key for key in key_names if len(key) > 0] # remove ''
    
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',',quotechar='"')
        table_head = None
        for row in csv_reader:
            if table_head is None: 
                table_head = row
            else:
                for key in key_names:
                    if key not in table_head:
                        raise Exception('Cannot find key name "%s" in CSV file "%s". Expected key names can be %s.' % \
                            (key,file_path, table_head))
                    else:
                        column_index = table_head.index(key)
                        if key not in parsed_dataset:
                            parsed_dataset[key] = [] # create list
                        parsed_dataset[key].append(row[column_index])

    return parsed_dataset

def write_csv_simple(file_path, csv_dict):
    keys = list(csv_dict.keys())
    lines=0
    for key in keys: # calculate max lines
        if len(csv_dict[key]) > lines:
            lines = len(csv_dict[key])
    mkdir(gd(file_path))
    with open(file_path,'w') as f:
        table_head = ''
        for key in keys: table_head += key+','
        f.write(table_head+'\n')
        for i in range(lines):
            for key in keys:
                if i >= len(csv_dict[key]):
                    f.write(',')
                else:
                    f.write('%s,' % csv_dict[key][i])
            f.write('\n')

def write_xlsx(dictobj, xlsx_path):
    worksheet_name = 'Sheet1'
    xlsx = SimpleExcelWriter( xlsx_path, worksheet_names=[worksheet_name])
    topfmt = xlsx.new_format(font_color='#FFFFFF', bg_color='#606060', bold=True)
    colfmt = xlsx.new_format(font_color='#000000', bg_color='#D9D9D9')
    errfmt = xlsx.new_format(font_color='#000000', bg_color='#FF9090')
    # write table head
    keys = list(dictobj.keys())
    for key, i in zip(keys, [q for q in range(len(keys))] ):
        xlsx.write((0, i), key, format=topfmt, worksheet_name=worksheet_name)
    # write content
    num_records = len(dictobj[keys[0]])
    for i in range(num_records):
        for key, j in zip(keys, [q for q in range(len(keys))] ):
            content = dictobj[key][i]
            if len(content) == 0:
                xlsx.write((i+1, j), content, format=errfmt, worksheet_name=worksheet_name)
            else:
                xlsx.write((i+1, j), content, worksheet_name=worksheet_name)
    xlsx.set_zoom(85,worksheet_name=worksheet_name)
    xlsx.set_filter((0,0), (num_records,len(keys)-1), worksheet_name=worksheet_name)
    xlsx.save_and_close()

def save_pkl(obj, pkl_path):
    with open(pkl_path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(pkl_path):
    content = None
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
    return content

def save_json(obj, json_path, indent=4):
    with open(json_path,'w') as f:
        json.dump(obj, f,indent=indent)

def load_json(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
    return obj

def load_pyval(py_path) -> dict:
    '''
    read a python file and parse all its variables as a dictionary
    '''
    d = {}
    with open(py_path,'r') as f:
        exec(f.read(), d)
        d.pop('__builtins__')
    return d

def try_load_gif(file_path):
    '''
    test if a GIF file can be successfully loaded.
    '''
    if file_exist(file_path) == False:
        return False
    if file_exist(file_path) and file_empty(file_path):
        return False
    try:
        success = True
        imageio.get_reader(file_path)
    except Exception:
        success = False
    except BaseException:
        raise
    return success

def load_mat(file_path):
    '''
    load a MATLAB *.mat file.
    '''
    mat = loadmat(file_path)
    return mat

def try_load_mat(file_path):
    '''
    test if a MATLAB matrix save file (*.mat) can be successfully loaded.
    '''
    success = True
    try:
        loadmat(file_path)
    except Exception:
        success = False
    except BaseException:
        raise
    return success


#
# NIFTI file operations
#

def _nifti_RAS_fix(image_data, image_affine):
    """
    Check the NIfTI orientation, and flip to 'RAS+' if needed.
    return: array after flipping
    """
    image_data0 = deepcopy(image_data)
    x, y, z = nib.aff2axcodes(image_affine)
    if x != 'R':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=0)
    if y != 'A':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=1)
    if z != 'S':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=2)
    return image_data0

def load_nifti(path, return_type='float32', force_RAS = False, 
    nan = None, posinf = None, neginf = None):
    '''
    Description
    -----------
    returns the loaded nifti data and header.

    Notes
    -----------
    force_RAS: if you want the loaded data is in RAS+ orientation
        (left to Right, posterior to Anterior, inferior to Superior),
        set it to True (default is False).

    nan, posinf, neginf: convert NaNs, +inf, -inf to floating point numbers. 
        If you do not want to convert them, leave them to None (which is the 
        default setting)

    usage
    -----------
    >>> data, header = load_nifti("example.nii.gz")
    >>> data, header = load_nifti("example.nii.gz", return_type='int32')
    '''
    try:
        nifti = nib.load(path)
    except KeyboardInterrupt:
        raise
    except:
        raise RuntimeError('cannot load nifti file "%s"\n'
            'Here are some possible reasons:\n'
            '1) file not exist or have no access;\n'
            '2) the file was corrupted.' % path)

    header = nifti.header.copy()
    data = nifti.get_fdata()
    
    if nan is not None:
        assert isinstance(nan, float), 'param "nan" should be a floating point number.'
        data = np.nan_to_num(data, nan=nan)
    if posinf is not None:
        assert isinstance(posinf, float), 'param "posinf" should be a floating point number.'
        data[data == np.inf] = posinf
    if neginf is not None:
        assert isinstance(neginf, float), 'param "neginf" should be a floating point number.'
        data[data == np.inf] = neginf

    if force_RAS:
        data = _nifti_RAS_fix( data, nifti.affine )
    if return_type is not None:
        data = data.astype(return_type)
    return data, header

def try_load_nifti(path):
    '''
    Sometimes we need to check if the NIFTI file is already exists, but only checking 
    the existense of the file is not enough, we need to guarantee the file is not 
    corrupted and can be successfully read.
    '''
    if file_exist(path) == False:
        return False
    if file_exist(path) and file_empty(path):
        return False
    try:
        success = True
        load_nifti(path)
    except Exception:
        success = False
    except BaseException: 
        # other types of system errors are triggered and we dont know how to handle it
        raise
    return success

def save_nifti(data, header, path, save_type='float32'):
    # Sometimes you might to avoid any loss of precision by making the data type the same as the input
    new_header = deepcopy(header)
    new_data = data.astype(save_type)
    new_header.set_data_dtype(new_data.dtype)
    nib.save(nib.nifti1.Nifti1Image(new_data, None, header=new_header),path)

def load_nifti_simple(path, return_type='float32'):
    data, _ = load_nifti(path, return_type=return_type)
    return data # only retreive data from file, ignore its header info

def save_nifti_simple(data,path, save_type='float32'): 
    # save NIFTI using the default header (clear position offset, using identity matrix 
    # and 1x1x1 unknown isotropic resolution)
	nib.save(nib.Nifti1Image(data.astype(save_type),affine=np.eye(4)),path)

def get_nifti_header(path):
    _, header = load_nifti(path, return_type=None)
    return header

def get_nifti_data(path, return_type='float32'):
    return load_nifti_simple(path, return_type=return_type)

# synchronize NIFTI file header
def sync_nifti_header(source_path, target_path, output_path):
    target_header = nib.load(target_path).header.copy()
    source_data = nib.load(source_path).get_fdata()
    save_nifti(source_data, target_header, output_path)

# get physical resolution
def get_nifti_pixdim(nii_path):
    nii = nib.load(nii_path)
    nii_dim = list(nii.header['dim'])
    nii_pixdim = list(nii.header['pixdim'])
    actual_dim = list(nii.get_fdata().shape)
    physical_resolution = []
    for v in actual_dim:
        physical_resolution.append(nii_pixdim[nii_dim.index(v)])
    return physical_resolution

def get_nifti_dtype(nii_path):
    # return nifti data type
    nii = nib.load(nii_path)
    return str(nii.dataobj.dtype)


def resample_nifti(source_path, new_resolution, output_path):
    '''
    Description
    --------------
    resample NIFTI file to another physical resolution.

    Parameters
    --------------
    source_path: str
        source NIFTI image path. Can be "*.nii" or "*.nii.gz" format
    new_resolution: list
        new physical resolution. Units are "mm". For example, if you
        want to resample image to 1mm isotropic resolution, use [1,1,1].
    output_path: str
        output NIFTI image path with resampled resolution. Can be "*.nii"
        or "*.nii.gz" format.
    '''
    input_img = nib.load(source_path)
    resampled_img = nibproc.resample_to_output(input_img, new_resolution, order=0)
    nib.save(resampled_img, output_path)

def nifti_main_axis(pixdim:list) -> str:
    assert len(pixdim) == 3, 'error, cannot determine main axis for non three dimension data.'
    axis = np.argmax(pixdim)
    if axis == 0: return 'sagittal'
    elif axis == 1: return 'coronal'
    else: return 'axial'

#
# Excel file operations
#

class SimpleExcelWriter(object):
    def __init__(self, file_path, worksheet_names='default'):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx".')
        self.file_path = file_path

        if isinstance(worksheet_names, str):
            self.worksheet_names = [worksheet_names]
        elif isinstance(worksheet_names, list):
            self.worksheet_names = worksheet_names
        else:
            raise RuntimeError('Only str or list type are accepted. Got type "%s".' % type(worksheet_names).__name__)

        # create workbook and worksheet(s)
        self.workbook = xlsxwriter.Workbook(self.file_path)
        self.worksheets = {}
        for worksheet_name in self.worksheet_names:
            self.worksheets[worksheet_name] = self.workbook.add_worksheet(worksheet_name)

    def _is_closed(self):
        return self.workbook is None
    
    def _check_closed(self):
        if self._is_closed():
            raise RuntimeError('Excel file is already closed and saved, which cannot be written anymore!')

    def _close(self):
        self.workbook.close()
        self.workbook = None

    def new_format(self,bold=False,italic=False,underline=False,font_color='#000000',bg_color='#FFFFFF'):

        self._check_closed()
        cell_format = self.workbook.add_format()
        cell_format.set_bold(bold)
        cell_format.set_italic(italic)
        cell_format.set_underline(underline)
        cell_format.set_font_color(font_color)
        cell_format.set_bg_color(bg_color)

        return cell_format

    def set_column_width(self, pos, width, worksheet_name='default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(pos, int):
            self.worksheets[worksheet_name].set_column(pos,pos, width)
        elif isinstance(pos,str):
            self.worksheets[worksheet_name].set_column(pos, width)
        elif isinstance(pos,tuple) or isinstance(pos,list):
            assert len(pos)==2, 'Invalid position setting.'
            start,end=pos
            self.worksheets[worksheet_name].set_column(start,end,width)

    def write(self, cell_name_or_pos, content, worksheet_name='default', format=None):
        self._check_closed()        
        if format is not None:
            assert isinstance(format, Format), 'Invalid cell format.'
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(cell_name_or_pos, tuple) or isinstance(cell_name_or_pos, list):
            assert len(cell_name_or_pos) == 2, 'Invalid cell position.'
            row,col = cell_name_or_pos
            if format:
                self.worksheets[worksheet_name].write(row,col,content,format)
            else:
                self.worksheets[worksheet_name].write(row,col,content)
        elif isinstance(cell_name_or_pos, str):
            if format:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content,format)
            else:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content)
        else:
            raise RuntimeError('Invalid cell name or position. Accepted types are: tuple, list or str but got "%s".' % \
                type(cell_name_or_pos).__name__)

    def set_freeze(self, first_n_rows = 0, first_n_cols = 0,  worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].freeze_panes(first_n_rows, first_n_cols)

    def set_zoom(self, zoom_factor = 100, worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].set_zoom(zoom_factor)
    
    def set_filter(self, start_pos, end_pos, worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].autofilter(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

    def save_and_close(self):
        self._close()


class SimpleExcelReader(object):
    def __init__(self, file_path):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx". ".xls" format is not supported.')
        self.file_path = file_path

        self.xlsx = openpyxl.load_workbook(file_path)
    
    def max_row(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_row
    
    def max_column(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_column
    
    def read(self, pos: Union[list, tuple], worksheet_name='default'):
        if self.xlsx is None:
            raise RuntimeError('file is already closed.')
        assert len(pos) == 2, 'invalid cell position'
        pos0 = pos[0]+1, pos[1]+1 # cell index starts with 1
        return self.xlsx[worksheet_name].cell(pos0[0], pos0[1]).value
    
    def close(self):
        self.xlsx.close()
        self.xlsx = None

def save_nparray_2D_as_nifti(x:np.ndarray, folder:str, prefix:str):
    '''
    Save a 2D numpy array [x, y] to disk as NIFTI format.
    '''
    assert len(x.shape) == 2, 'Array must be 2D. Got array with shape %s.' % str(x.shape)
    mkdir(folder)
    path = join_path(folder, '%s.nii.gz' % prefix)
    save_nifti_simple(x[None,:,:], path) # add one dimension

def save_nparray_3D_as_nifti(x:np.ndarray, folder:str, prefix:str):
    '''
    Save a 3D numpy array [x, y, z] to disk as NIFTI format.
    '''
    assert len(x.shape) == 3, 'Array must be 3D. Got array with shape %s.' % str(x.shape)
    mkdir(folder)
    path = join_path(folder, '%s.nii.gz' % prefix)
    save_nifti_simple(x, path)

def save_nparray_4D_as_nifti(x:np.ndarray, folder:str, prefix:str):
    '''
    Save a 4D numpy array [channel, x, y, z] to disk as NIFTI format.
    '''
    assert len(x.shape) == 4, 'Array must be 4D. Got array with shape %s.' % str(x.shape)
    mkdir(folder)
    num_channels = x.shape[0]
    for channel in range(num_channels):
        path = join_path(folder, '%s%d.nii.gz' % (prefix, channel))
        save_nifti_simple(x[channel], path)

def save_nparray_5D_as_nifti(x:np.ndarray, folder:str, prefix:str):
    '''
    Save a 5D numpy array [batch, channel, x, y, z] to disk as NIFTI format.
    '''
    assert len(x.shape) == 5, 'Array must be 5D. Got array with shape %s.' % str(x.shape)
    mkdir(folder)
    num_batches = x.shape[0]
    num_channels = x.shape[1]
    for batch in range(num_batches):
        for channel in range(num_channels):
            path = join_path(folder, '%s%d_%d.nii.gz' % (prefix, batch, channel))
            save_nifti_simple(x[batch,channel], path)

def save_nparray_nD_as_nifti(x:np.ndarray, folder:str, prefix:str):
    '''
    Save a nD numpy array to disk as NIFTI format.
    n = [3, 4, 5]

    n=3: [x, y, z]
    n=4: [channels, x, y, z]
    n=5: [batch_size, channels, x, y, z]
    '''
    D = len(x.shape)
    assert D>=2 and D<=5, 'Array must be 2D, 3D, 4D or 5D. Got array with shape %s.' % str(x.shape)
    if   D == 2: return save_nparray_2D_as_nifti(x,folder,prefix)
    elif D == 3: return save_nparray_3D_as_nifti(x,folder,prefix)
    elif D == 4: return save_nparray_4D_as_nifti(x,folder,prefix)
    elif D == 5: return save_nparray_5D_as_nifti(x,folder,prefix)
    else:
        raise RuntimeError("Invalid input shape %s." % str(x.shape))

##########################################################
# Implement simple and robust process-based parallelism. # 
##########################################################

# mute current process
def _mute_this():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

class ParallelRuntimeError(Exception):
    pass

class ParallelFunctionExceptionWrapper(object):
    '''
    A simple wrapper class that wraps a Python function to have basic exception handling mechanics
    '''
    def __init__(self, callable_object):
        self.__callable_object = callable_object
    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable_object(*args, **kwargs)
        except BaseException as e:
            multiprocessing.get_logger().error( traceback.format_exc() )
            try:
                kill_process_tree(os.getpid(), kill_self=False)
            except: pass
            trace = traceback.format_exc()
            raise ParallelRuntimeError(trace)
        return result

# process-based parallelism
def run_parallel(worker_function:Callable, list_of_tasks_args: List[tuple], num_workers:int, progress_bar_msg:str, 
    print_output:bool=False, show_progress_bar:bool=True):
    '''
    Description
    ------------
    Process-based parallelism. Using multiple CPU cores to execute the same function using different
    parameters. Each worker function is independent and it should not communicate with other workers
    when running.

    Parameters
    ------------
    "worker_function": <function object>
        Function that will be executed concurrently.
    "list_of_tasks_args": list
        A list of arguments for all tasks. Its format should be something like this:  
        [ (arg1, arg2, ...), (arg1, arg2, ...), (arg1, arg2, ...) , ... ]
          ^ worker1 params   ^ worker2 params   ^ worker3 params    ...
    "num_workers": int
        Number of workers.
    "progress_bar_msg": str
        A short string that will be displayed in progress bar.
    "print_output": bool
        If you don't want to see any output or error message during parallel execution, 
        set it to False (default). Otherwise turn it on.
    "show_progress_bar": bool
        If you don't want to show progress bar set this to False (default: True).

    Note
    ------------
    * Currently nested parallelism is not supported.
    * Please DO NOT call this function inside a worker function! The behavior might be unexpected.
    * The worker function can safely invoke a bash call using run_shell(...). You can encapsulate
    the run_parallel(...) in a ignore_SIGINT() context to temporarily disable keyboard interrupt 
    (Ctrl+C) such as:

    >>> from digicare.utilities.misc import ignore_SIGINT
    >>> with ignore_SIGINT():
    >>>     run_parallel(...)

    * For more detailed usage please see the examples provided below.
    
    Example
    ------------
    >>> tasks = [ (arg1, arg2, ...), (arg1, arg2, ...), ... ]
    >>> # define your worker function here
    >>> def _worker_function(params):
    >>>     arg1, arg2, ... = params
    >>>     # do more work here,
    >>>     # but please DO NOT call "run_parallel" again in this worker function!
    >>>     # ...
    >>> # start running 8 tasks in parallel
    >>> run_parallel(_worker_function, tasks, 8, "working...")
    '''

    pool = Pool(processes=num_workers, initializer=_mute_this if not print_output else None)
    total_tasks = len(list_of_tasks_args)
    if total_tasks == 0: 
        # no task to execute in parallel, just return
        return
    tasks_returned = []
    tasks_error = []

    for i in range(total_tasks):
        pool.apply_async(
            ParallelFunctionExceptionWrapper(worker_function), 
            (list_of_tasks_args[i],) , 
            callback=tasks_returned.append,
            error_callback=tasks_error.append
        )
        
    pool.close() # no more work to append
    # now waiting all tasks to finish
    # and we draw a simple progress bar to track real time progress 
    timer = Timer()
    finished, error_ = 0, 0
    anim_counter, anim_chars = 0, '-\|/' # simple animation effect
    try:
        while True:
            finished = len(tasks_returned)
            error_ = len(tasks_error)

            if error_ > 0:
                trace = tasks_error[0] # only retrieve first error trace
                
                error_msg = '\n\n==========\n'  \
                            'One of the worker process crashed due to unhandled exception.\n' \
                            'Worker function name is: "%s".\n\n' % worker_function.__name__
                error_msg += '** Here is the traceback message:\n\n'
                error_msg += str(trace)
                error_msg += '\n** Main process will exit now.\n'
                error_msg += '==========\n\n'
                print(error_msg)
                raise RuntimeError('One of the worker process crashed due to unhandled exception.')
                
            if show_progress_bar: 
                minibar(msg=progress_bar_msg, a=finished, b=total_tasks, time=timer.elapsed(), last=anim_chars[anim_counter])
                anim_counter += 1
                anim_counter %= len(anim_chars)

            # end condition
            if finished == total_tasks:
                if show_progress_bar: 
                    minibar(msg=progress_bar_msg, a=finished, b=total_tasks, time=timer.elapsed()) # draw bar again
                break
            else:
                sleep(0.2)
        print('')
        pool.join()
    except: 
        # exit the whole program if anything bad happens (for safety)
        try: # try killing all its child, in case worker process
             # spawns child processes 
            for wproc in pool._pool: # for each worker process in pool
                kill_process_tree(wproc.pid, kill_self=False)
                sleep(0.5)
        except: 
            pass # omit any exception when killing child process
        pool.terminate()
        pool.join()
        exit(1)

    return tasks_returned

###################################################################################
# This script wraps the registration command into a simple Python function call   #
# (they are: "image_registration_with_label(...)" and "image_registration(...)"). #
#                                                                                 #
# For more tutorials about medical image registration using ANTs, please          #
# visit: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call.  # 
# "antsRegistration" can be used to register two images even if their imaging     #
# modalities are different from each other.                                       #
###################################################################################


def get_ANTs_version_string():
    stdout, _ = try_shell('antsRegistration --version', stdio=True)
    stdout = stdout.replace('\n',' ')
    return stdout

def antsRegistration(source, target, warped, 
    interpolation_method='Linear', use_histogram_matching=False,
    deform_type='Elastic',advanced_config=None, dtype_check=False) -> str:
    '''
    Description
    ------------
    Generates bash call from Python function to register a pair of medical
    images using ANTs. Combining it with run_shell(...) to launch the task.

    - NOTE: Deformation fields will be saved under the directory where the 
    warped image is located.

    Usage
    ------------
    >>> source = '/path/to/source.nii.gz'
    >>> target = '/path/to/target.nii.gz'
    >>> warped = '/path/to/warped.nii.gz'
    >>> run_shell( antsRegistration(source, target, warped, interpolation_method='Linear') )

    Parameters
    ------------
    source: str
        source image path
    target: str
        target image path
    warped: str
        warped output image path
    deform_type: str
        deform type can be either "Linear" or "Elastic" (default).
        Rigid = only "Rigid" transform,
        Linear = "Rigid + Affine" transforms,
        Elastic = "Rigid + Affine + SyN" transforms.
    advanced_config: dict
        advanced configurations for deformable registration method 
        "SyN". Usually you don't need to manually adjust this.
    '''

    assert deform_type in ['Elastic', 'Linear', 'Rigid'], 'unknown deformation type.'
    assert interpolation_method in ['Linear', 'NearestNeighbor'], 'unknown interpolation method.'
    assert use_histogram_matching in [True, False], 'invalid parameter setting for "use_histogram_matching".'

    assert file_exist(source), 'Cannot open source image "%s". File not exist or insufficient privilege.' % source
    assert file_exist(target), 'Cannot open target image "%s". File not exist or insufficient privilege.' % target
    if dtype_check:
        if get_nifti_dtype(source) != get_nifti_dtype(target):
            raise RuntimeError('Source data type (%s) and target data type (%s) are different! '
                            'antsRegistration may fail if the two images have different storage data type.\n'
                            'Source image: "%s".\nTarget image: "%s".\n' % \
                                (get_nifti_dtype(source), get_nifti_dtype(target), source, target))
    
    output_directory = gd(abs_path(warped))
    mkdir(output_directory)
    output_transform_prefix = join_path(output_directory,'warp_')

    # fill in default configurations
    config = {
        'SyN_gradientStep' : 0.1,
        'SyN_updateFieldVarianceInVoxelSpace' : 3.0,
        'SyN_totalFieldVarianceInVoxelSpace' : 0.0,
        'SyN_CC_neighborVoxels': 4,
        'SyN_convergence' : '100x70x50x20',
        'SyN_shrinkFactors' : '8x4x2x1',
        'SyN_smoothingSigmas' : '3x2x1x0'
    }
    if advanced_config is not None:
        for key in advanced_config:
            if key in config:
                config[key] = advanced_config[key] # override setting
            else:
                warnings.warn('Unknown config setting "%s".' % key, UserWarning)

    # generate registration command
    command = 'antsRegistration '
    command += '--dimensionality 3 '                                     # 3D image
    command += '--float 1 '                                              # 0: use float64, 1: use float32 (save mem)
    command += '--collapse-output-transforms 1 '
    command += '--output [%s,%s] ' % (output_transform_prefix,warped)
    command += '--interpolation %s ' % interpolation_method
    command += '--use-histogram-matching %s ' % ( '0' if use_histogram_matching == False else '1')
    command += '--winsorize-image-intensities [0.005,0.995] '
    command += '--initial-moving-transform [%s,%s,1] ' % (target,source) # initial moving transform
    command += '--transform Rigid[0.1] '                                 # rigid transform
    command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
    command += '--convergence [1000x500x250x0,1e-6,10] '
    command += '--shrink-factors 8x4x2x1 '
    command += '--smoothing-sigmas 3x2x1x0vox '
    if deform_type in ['Linear', 'Elastic']:
        command += '--transform Affine[0.1] ' # affine transform
        command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
        command += '--convergence [1000x500x250x0,1e-6,10] '
        command += '--shrink-factors 8x4x2x1 '
        command += '--smoothing-sigmas 3x2x1x0vox '
    if deform_type in ['Elastic']:
        # For medical image registration with large image deformations, maybe adding a
        # "TimeVaryingVelocityField" transform is better (see https://github.com/stnava/C).
        # But for robustness here I will only use SyN (which is the most commonly used
        # method) for deformable registration.
        command += '--transform SyN[%f,%f,%f] ' % \
            (config['SyN_gradientStep'], config['SyN_updateFieldVarianceInVoxelSpace'], \
            config['SyN_totalFieldVarianceInVoxelSpace'])
        command += '--metric CC[%s,%s,1,%d] ' % (target,source, config['SyN_CC_neighborVoxels'])
        command += '--convergence [%s,1e-6,10] ' % (config['SyN_convergence'])
        command += '--shrink-factors %s ' % config['SyN_shrinkFactors']
        command += '--smoothing-sigmas %svox ' % config['SyN_smoothingSigmas']
    return command

def antsApplyTransforms(source,reference,transform,output,
    interpolation_method='Linear',inverse_transform=False):

    assert interpolation_method in ['Linear', 'NearestNeighbor'], 'unknown interpolation method.'
    assert inverse_transform in [True,False], 'invalid parameter setting for "inverse_transform".'

    command = 'antsApplyTransforms '
    command += '-d 3 --float --default-value 0 '
    command += '-i %s ' % source
    command += '-r %s ' % reference
    command += '-o %s ' % output
    command += '-n %s ' % interpolation_method
    command += '-t [%s,%s] ' % (transform, ( '0' if inverse_transform == False else '1')  )

    return command
    
def image_registration(moving, fixed, moved, save_deformations_to = None, advanced_config = None,  dtype_check = False):
    try:
        output_directory = mkdir(gd(moved))
        run_shell(antsRegistration(moving, fixed, moved, 
            interpolation_method='Linear', use_histogram_matching=False, advanced_config=advanced_config, dtype_check=dtype_check),
            print_command=False)
    except:
        raise # if anything bad happens, the temporary file will not be deleted,
              # user need to delete them manually.
    finally:
        # remove temporary files
        if save_deformations_to is not None:
            mkdir(save_deformations_to)
            mv( join_path(output_directory, 'warp_0GenericAffine.mat'), join_path(save_deformations_to, 'warp_0GenericAffine.mat') )
            mv( join_path(output_directory, 'warp_1Warp.nii.gz'), join_path(save_deformations_to, 'warp_1Warp.nii.gz') )
            mv( join_path(output_directory, 'warp_1InverseWarp.nii.gz'), join_path(save_deformations_to, 'warp_1InverseWarp.nii.gz') )
        else:
            rm(join_path(output_directory, 'warp_0GenericAffine.mat'))
            rm(join_path(output_directory, 'warp_1Warp.nii.gz'))
            rm(join_path(output_directory, 'warp_1InverseWarp.nii.gz'))


def _parallel_registration(args):
    # unpack args
    moving, fixed, save_dir, work_dir, \
        allow_large_deformations, allow_quick_registration, keep_deformation, dtype_check = args

    moving_case, moving_img = moving
    fixed_case, fixed_img = fixed
    output_case_name = '%s_to_%s' % (moving_case, fixed_case)
    output_file = join_path(save_dir, '%s.nii.gz' % output_case_name)
    output_deformations = [
        join_path(save_dir, output_case_name, 'warp_0GenericAffine.mat'), # rigid + affine
        join_path(save_dir, output_case_name, 'warp_1Warp.nii.gz'),       # forward elastic deformation (S->T)
        join_path(save_dir, output_case_name, 'warp_1InverseWarp.nii.gz') # backward elastic deformation (T->S)
    ]

    # check if this registration can be skipped (finished from previous run)

    files_need_to_exist = [output_file] if keep_deformation == False else [output_file] + output_deformations
    skip = True
    for file in files_need_to_exist:
        if file.endswith('.nii.gz'):
            if try_load_nifti(file) == False:
                skip = False
                break
        if file.endswith('.mat'):
            if try_load_mat(file) == False:
                skip = False
                break
                    
    if skip: 
        return

    try:
        advanced_config = {}
        if allow_large_deformations:
            advanced_config['SyN_gradientStep'] = 0.3
            advanced_config['SyN_updateFieldVarianceInVoxelSpace'] = 3
            advanced_config['SyN_convergence'] = '200x100x50x25'
            advanced_config['SyN_shrinkFactors'] = '8x4x2x1'
            advanced_config['SyN_smoothingSigmas'] = '3x2x1x0'
        if allow_quick_registration:
            advanced_config['SyN_convergence'] = '200x100x50'
            advanced_config['SyN_shrinkFactors'] = '8x4x2'
            advanced_config['SyN_smoothingSigmas'] = '3x2x1'
        temp_dir = make_unique_dir(basedir=work_dir)
        #print(temp_dir)
        temp_output = join_path(temp_dir,'%s.nii.gz' % output_case_name)
        if not keep_deformation:
            image_registration( moving_img, fixed_img, temp_output , save_deformations_to=None, advanced_config=advanced_config, dtype_check=dtype_check)
        else:
            deformation_dir = mkdir(join_path(save_dir, output_case_name))
            image_registration( moving_img, fixed_img, temp_output , save_deformations_to=deformation_dir, advanced_config=advanced_config, dtype_check=dtype_check)
        mv(temp_output, output_file)
        
    except:
        raise

    else:
        print('cleaning up...')
        rm(temp_dir)

# launch from class instance
class ANTsGroupRegistration(object):
    def __init__(self, sources, targets, save_dir, work_dir, num_workers, 
        allow_large_deformations:bool=True, allow_quick_registration:bool=False, keep_deformation:bool=True,
        data_type_checking: bool=False):
        
        '''
        sources: [(case1, img1), (case2, img2), ... , (caseN, imgN)]
        targets: [(case1, img1), (case2, img2), ... , (caseN, imgN)]
        '''

        self.sources = sources
        self.targets = targets
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.allow_large_deformations = allow_large_deformations
        self.allow_quick_registration = allow_quick_registration
        self.keep_deformation = keep_deformation
        self.work_dir = work_dir
        self.data_type_checking = data_type_checking

    def _organize_tasks(self, distributed = 'none'):

        task_list = []
        for i in range(len(self.sources)):
            for j in range(len(self.targets)):
                task_args = (self.sources[i], self.targets[j], self.save_dir, self.work_dir, \
                    self.allow_large_deformations, self.allow_quick_registration, 
                    self.keep_deformation, self.data_type_checking)
                task_list.append(task_args)
            
        if distributed != 'none':
            sub_task, total_task = distributed.split('/')
            sub_task, total_task = int(sub_task), int(total_task)
            if sub_task < 1 or sub_task > total_task:
                raise RuntimeError('parameter error : "distributed"')
            sub_task_list = []
            for i in range(len(task_list)):
                if i % total_task == sub_task - 1:
                    sub_task_list.append(task_list[i])
            task_list = sub_task_list # override task list
            print('Distributed task %d/%d, %d image registration operations.' % (sub_task, total_task, len(task_list)))
        else:
            print('Running full task (%d image registration operations).' % len(task_list))
        return task_list

    def launch(self, task_list = None):

        mkdir(self.save_dir)
        mkdir(self.work_dir)

        if task_list == None:
            task_list = self._organize_tasks()

        print('start image registration.')
        run_parallel(_parallel_registration, task_list, self.num_workers, "registration", 
                     print_output=False)
        print('registration finished.')
