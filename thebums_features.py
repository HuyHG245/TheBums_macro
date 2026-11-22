"""Small offline/testable support features for The Bums Macro.

User data stays outside the install folder; no network or injected game access.
Full developer tuning tools are not included in the main GUI.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path


def _atomic_json(path, content):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    os.replace(temp, dest)


def _read_json(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            value=json.load(f)
        return value
    except (OSError, ValueError, TypeError):
        return copy.deepcopy(default)


class ConfigPresetStore:
    """Named presets, deliberately excluding account secrets and Discord credentials."""
    ALLOWED=('family','timings','keybinds','move_shifted','fix_loot',
             'adaptive_resolution','performance_mode')
    def __init__(self, data_folder):
        self.path=Path(data_folder)/'config_presets.json'
        self.lock=threading.RLock()

    def list(self):
        with self.lock:
            content=_read_json(self.path, {'schema':1,'presets':{}})
            return copy.deepcopy(content.get('presets',{}))

    def save(self, name, settings):
        if not isinstance(name,str) or not (1<=len(name.strip())<=48) or any(ord(c)<32 for c in name):
            raise ValueError('Preset name must have 1 to 48 printable characters')
        key=name.strip()
        with self.lock:
            content=self.list()
            values={k:copy.deepcopy(settings[k]) for k in self.ALLOWED if k in settings}
            content[key]={'created':dt.datetime.now(dt.timezone.utc).isoformat(),
                          'family':str(settings.get('family','')),
                          'values':values}
            if len(content)>100:raise ValueError('Maximum of 100 presets')
            _atomic_json(self.path,{'schema':1,'presets':content})
            return key

    def load(self,name,current_family):
        preset=self.list().get(name)
        if not preset:raise KeyError('Preset not found')
        if preset['family']!=current_family:
            raise ValueError('Preset belongs to '+preset['family'])
        vals=preset['values']
        return {k:copy.deepcopy(vals[k]) for k in self.ALLOWED if k in vals}

    def delete(self,name):
        with self.lock:
            presets=self.list()
            if name not in presets:raise KeyError('Preset not found')
            del presets[name]
            _atomic_json(self.path,{'schema':1,'presets':presets})

    @staticmethod
    def _valid_name(name):
        if not isinstance(name,str) or not (1<=len(name.strip())<=48) or any(ord(c)<32 for c in name):
            raise ValueError('Preset name must have 1 to 48 printable characters')
        return name.strip()

    def duplicate(self,old,new):
        target=self._valid_name(new)
        with self.lock:
            presets=self.list()
            if old not in presets:raise KeyError('Preset not found')
            if target in presets:raise ValueError('Preset name is already used')
            if len(presets)>=100:raise ValueError('Maximum of 100 presets')
            presets[target]=copy.deepcopy(presets[old])
            presets[target]['created']=dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_json(self.path,{'schema':1,'presets':presets})
            return target

    def rename(self,old,new):
        target=self._valid_name(new)
        with self.lock:
            presets=self.list()
            if old not in presets:raise KeyError('Preset not found')
            if target in presets and target!=old:raise ValueError('Preset name is already used')
            if target!=old:
                presets[target]=presets.pop(old)
                _atomic_json(self.path,{'schema':1,'presets':presets})
            return target


class SessionHistory:
    """Bounded event journal: lightweight even during an overnight session."""
    def __init__(self, data_folder):
        self.path=Path(data_folder)/'session_history.jsonl'
        self.lock=threading.RLock()
        self.session_id=None
        self.family=None

    def record(self, kind, **attrs):
        payload={'time':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'kind':str(kind),'session':self.session_id,'family':self.family}
        for k,v in attrs.items():
            if k in ('ps_code','webhook','token','password'):
                continue
            if isinstance(v,(int,float,bool,str)) and len(str(v))<150:
                payload[k]=v
        with self.lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            if self.path.exists() and self.path.stat().st_size>5_000_000:
                # Keep latest log separately; no large unbounded logs.
                rotated=self.path.with_suffix('.old.jsonl')
                os.replace(self.path,rotated)
            with self.path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(payload,ensure_ascii=False)+'\n')

    def begin(self,family):
        self.session_id=uuid.uuid4().hex[:16]
        self.family=family
        self.record('session_started')
        return self.session_id

    def end(self,status):
        self.record('session_finished',status=status)

    def events(self,limit=20000):
        records=[]
        with self.lock:
            try:
                with self.path.open(encoding='utf-8') as f:
                    for line in f:
                        try:
                            row=json.loads(line)
                            if isinstance(row,dict):records.append(row)
                        except (ValueError,TypeError):pass
            except OSError:pass
        return records[-limit:]

    def summary(self):
        events=self.events()
        if not events:return {'sessions':0,'completed':0,'failed':0,'reconnects':0,'errors':0,'last_session':None}
        last=next((e.get('session') for e in reversed(events) if e.get('session')),None)
        last_events=[e for e in events if e.get('session')==last]
        def count(kind,rows):return sum(1 for e in rows if e.get('kind')==kind)
        started=next((e.get('time') for e in last_events if e.get('kind')=='session_started'),None)
        ended=next((e.get('time') for e in reversed(last_events) if e.get('kind')=='session_finished'),None)
        return {'sessions':count('session_started',events),
                'completed':count('mission_completed',events),
                'failed':count('mission_failed',events),
                'reconnects':count('reconnect_completed',events),
                'errors':count('error',events),
                'last_session':{'id':last,'start':started,'end':ended,
                                'completed':count('mission_completed',last_events),
                                'failed':count('mission_failed',last_events),
                                'reconnects':count('reconnect_completed',last_events),
                                'errors':count('error',last_events)}}


class ErrorSnapshotStore:
    """Bounded on-error diagnostic capture. Private PS screens are never saved."""
    def __init__(self,base):
        self.path=Path(base)/'error_snapshots'
        self.lock=threading.Lock()
        self.last={}

    def save(self,category,state,frame=None,secrets=(),sensitive_screen=False,log_lines=()):
        now=time.monotonic()
        with self.lock:
            if now-self.last.get(category,0)<60:return None
            self.last[category]=now
        self.path.mkdir(parents=True,exist_ok=True)
        folder=self.path/(dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        folder.mkdir()
        clean={}
        for key,value in state.items():
            if any(s in key.lower() for s in ('token','webhook','ps_code','password','secret')):
                continue
            if isinstance(value,(str,int,float,bool)):
                txt=str(value)
                for secret in secrets:
                    if secret:txt=txt.replace(secret,'[REDACTED]')
                clean[key]=txt[:300]
        _atomic_json(folder/'state.json',{'category':category,'time':dt.datetime.now().isoformat(),
                                          'state':clean})
        # Text-only recent activity is useful for remote debugging. Redact
        # credentials, and exclude URL-like tokens as a fallback safeguard.
        if log_lines:
            sanitized=[]
            for line in list(log_lines)[-60:]:
                item=str(line)[:600]
                for secret in secrets:
                    if secret:item=item.replace(secret,'[REDACTED]')
                item=re.sub(r'https?://\S+', '[REDACTED_URL]', item)
                sanitized.append(item)
            (folder/'activity.log').write_text('\n'.join(sanitized),encoding='utf-8')
        if frame is not None and not sensitive_screen:
            try:
                import cv2
                cv2.imwrite(str(folder/'screen.png'),frame)
            except Exception:pass
        # bounded file count. Remove oldest snapshots over 30.
        snapshots=sorted((p for p in self.path.iterdir() if p.is_dir()),key=lambda p:p.name)
        if len(snapshots)>30:
            import shutil
            for p in snapshots[:-30]:shutil.rmtree(p,ignore_errors=True)
        return str(folder)


class UpdateHistory:
    """Rollback/failed version metadata. Never retries the exact failed artifact."""
    def __init__(self,data_folder):
        self.path=Path(data_folder)/'update_history.json'
        self.lock=threading.RLock()

    def read(self):
        with self.lock:
            value=_read_json(self.path,{'entries':[],'blocked':{}})
            return value if isinstance(value,dict) else {'entries':[],'blocked':{}}

    def record(self,tag,status,digest='',details=''):
        with self.lock:
            h=self.read()
            entries=h.get('entries',[])
            entries.insert(0,{'version':str(tag),'status':str(status),
                              'digest':str(digest),'details':str(details)[:250],
                              'time':dt.datetime.now(dt.timezone.utc).isoformat()})
            h['entries']=entries[:20]
            if status=='rolled_back' and digest:
                h.setdefault('blocked',{})[str(tag)]=str(digest)
            _atomic_json(self.path,h)

    def blocked(self,tag,digest):
        h=self.read()
        return bool(digest and h.get('blocked',{}).get(tag)==digest)
