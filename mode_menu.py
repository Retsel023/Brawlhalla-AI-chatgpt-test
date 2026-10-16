"""Native Windows session launcher; importing this module needs no desktop toolkit."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

from session_profiles import PROFILE_PRESETS, SessionOptions, default_options


def build_options(form):
    """Convert editable text fields into the runner's validated session settings."""
    enemies = form.get('opponents', '')
    if isinstance(enemies, str):
        enemies = [value.strip() for value in enemies.split(',') if value.strip()]
    match_format = form.get('match_format') or PROFILE_PRESETS[form['profile']]['match_format']
    return default_options(
        form['profile'], match_format=match_format,
        self_player=form.get('self_player', 'Player1').strip(),
        teammate_player=form.get('teammate_player', 'Player2').strip() if match_format == '2v2' else None,
        opponents=enemies or None, friendly_fire=form.get('friendly_fire', False),
        learning=form.get('learning', PROFILE_PRESETS[form['profile']]['learning']),
        steps=int(form.get('steps', 0)), max_seconds=float(form.get('max_seconds', 0)),
        checkpoint_interval=int(form.get('checkpoint_interval', 1000)),
        model_root=form.get('model_root', 'models/sessions').strip(),
        initialize_from=form.get('initialize_from', '').strip() or None,
        device=form.get('device', 'auto'))


def options_to_form(options):
    form = options.to_dict()
    form.update(form.pop('profile'))
    form['opponents'] = ', '.join(form['opponents'])
    form['teammate_player'] = form['teammate_player'] or 'Player2'
    form['initialize_from'] = form['initialize_from'] or ''
    return form


class SessionProcess:
    """Own exactly one child and its stop file, output queue, and rotating log."""

    def __init__(self, repo_path, config_path, *, runs_root=None, command_builder=None):
        self.repo = Path(repo_path).resolve()
        self.config = Path(config_path).resolve()
        self.runs_root = Path(runs_root) if runs_root else self.repo / 'runs' / 'sessions'
        self.command_builder = command_builder
        self.process = None
        self.reader = None
        self.run_dir = self.options_path = self.stop_path = None
        self.output = queue.Queue(maxsize=1000)
        self.check = self.stop_requested = self.forced = False

    def start(self, options, *, check=False):
        if self.process is not None and self.poll() is None:
            raise RuntimeError('A session process is already running')
        if not isinstance(options, SessionOptions):
            raise TypeError('Expected validated SessionOptions')
        self.run_dir = self.runs_root / uuid.uuid4().hex
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.options_path, self.stop_path = self.run_dir / 'options.json', self.run_dir / 'stop.request'
        self.options_path.write_text(json.dumps(options.to_dict(), indent=2) + '\n', encoding='utf-8')
        self.check, self.stop_requested, self.forced = check, False, False
        command = [sys.executable, '-u', str(self.repo / 'agent.py'), 'session',
                   '--session-file', str(self.options_path), '--config', str(self.config),
                   '--stop-file', str(self.stop_path)]
        if check:
            command.append('--check')
        if self.command_builder:
            command = self.command_builder(self.options_path, self.config, self.stop_path, check)
        self.output = queue.Queue(maxsize=1000)
        handler = RotatingFileHandler(self.run_dir / 'session.log', maxBytes=5 * 1024 * 1024,
                                      backupCount=2, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger = logging.Logger(f'session.{self.run_dir.name}')
        logger.addHandler(handler)
        try:
            self.process = subprocess.Popen(command, cwd=self.repo, shell=False,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                            errors='replace', bufsize=1)
        except Exception:
            handler.close()
            raise
        self.reader = threading.Thread(target=self._read_output, args=(self.process, self.output, logger, handler),
                                       daemon=True)
        self.reader.start()
        return self.run_dir

    @staticmethod
    def _read_output(process, output, logger, handler):
        try:
            with process.stdout:
                for line in iter(lambda: process.stdout.readline(8192), ''):
                    logger.info(line.rstrip('\r\n'))
                    try:
                        output.put_nowait(line)
                    except queue.Full:
                        # The complete output remains in the bounded rotating log files.
                        try:
                            output.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            output.put_nowait(line)
                        except queue.Full:
                            pass
        finally:
            handler.close()

    def poll(self):
        return self.process.poll() if self.process is not None else None

    def request_stop(self):
        if self.process is not None and self.poll() is None:
            self.stop_path.touch(exist_ok=True)
            self.stop_requested = True

    def force_stop(self):
        if self.process is not None and self.poll() is None:
            self.process.kill()
            self.forced = True

    def drain_output(self, limit=100):
        lines = []
        for _ in range(limit):
            try:
                lines.append(self.output.get_nowait())
            except queue.Empty:
                break
        return lines


def open_menu(config_path='config.json'):
    """Open the Tk menu. Desktop imports are deliberately local to this function."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    repo = Path(__file__).resolve().parent
    config = Path(config_path).resolve()
    controller = SessionProcess(repo, config)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f'No usable desktop display: {error}') from error
    root.title('Brawlhalla learning agent')
    root.geometry('900x850')
    root.minsize(760, 700)
    panel = ttk.Frame(root, padding=12)
    panel.pack(fill='both', expand=True)
    panel.columnconfigure(1, weight=1)
    ttk.Label(panel, text='Windows game-side provider required; not installed by this menu.',
              wraplength=820, foreground='#a04000').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
    labels = {preset['label']: name for name, preset in PROFILE_PRESETS.items()}
    selected = tk.StringVar()
    ttk.Label(panel, text='Mode').grid(row=1, column=0, sticky='w')
    mode = ttk.Combobox(panel, textvariable=selected, values=list(labels), state='readonly')
    mode.grid(row=1, column=1, columnspan=2, sticky='ew')
    description = tk.StringVar()
    ttk.Label(panel, textvariable=description, wraplength=820).grid(row=2, column=0, columnspan=3, sticky='w', pady=8)
    initial = options_to_form(default_options())
    boolean_fields = {'learning', 'friendly_fire'}
    fields = {name: (tk.BooleanVar(value=value) if name in boolean_fields else tk.StringVar(value=value))
              for name, value in initial.items() if name not in {'profile', 'online', 'opponent_kind'}}
    widgets = {}
    captions = [('match_format', 'Bot match format'), ('learning', 'Update policy weights'),
                ('steps', 'Steps (0 = continuous)'), ('max_seconds', 'Time limit in seconds (0 = unlimited)'),
                ('checkpoint_interval', 'Checkpoint every N steps'), ('self_player', 'AI player key'),
                ('teammate_player', 'Teammate key (2v2)'), ('opponents', 'Opponent keys (comma separated)'),
                ('friendly_fire', 'Friendly fire (2v2)'), ('device', 'Device (cuda = AMD ROCm)'),
                ('model_root', 'Model folder'), ('initialize_from', 'Initialize from checkpoint (optional)')]
    for row, (name, caption) in enumerate(captions, 3):
        ttk.Label(panel, text=caption).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=3)
        if name in boolean_fields:
            widget = ttk.Checkbutton(panel, variable=fields[name])
        elif name in {'device', 'match_format'}:
            widget = ttk.Combobox(panel, textvariable=fields[name], state='readonly',
                                  values=('auto', 'cpu', 'cuda') if name == 'device' else ('1v1', '2v2'))
        else:
            widget = ttk.Entry(panel, textvariable=fields[name])
        widget.grid(row=row, column=1, sticky='ew', pady=3)
        widgets[name] = widget
        if name in {'model_root', 'initialize_from'}:
            def browse(field=name):
                value = (filedialog.askdirectory(parent=root) if field == 'model_root' else
                         filedialog.askopenfilename(parent=root, filetypes=[('Model checkpoints', '*.zip'), ('All files', '*')]))
                if value:
                    fields[field].set(value)
            button = ttk.Button(panel, text='Browse…', command=browse)
            button.grid(row=row, column=2, padx=(6, 0))
            widgets[name + '_browse'] = button
    status = tk.StringVar(value='Idle — configure a session or check the provider connection.')
    ttk.Label(panel, textvariable=status, wraplength=820).grid(row=16, column=0, columnspan=3, sticky='w', pady=8)
    actions = ttk.Frame(panel)
    actions.grid(row=15, column=0, columnspan=3, sticky='ew', pady=10)
    log = tk.Text(panel, height=15, state='disabled', wrap='word')
    log.grid(row=17, column=0, columnspan=3, sticky='nsew')
    panel.rowconfigure(17, weight=1)
    running = closing = False

    def current_options():
        return build_options({'profile': labels[selected.get()], **{name: value.get() for name, value in fields.items()}})

    def refresh_fields():
        teams = fields['match_format'].get() == '2v2'
        for name, widget in widgets.items():
            disabled = running or (name in {'teammate_player', 'friendly_fire'} and not teams)
            disabled |= name == 'match_format' and labels[selected.get()] != 'bot_training'
            widget.configure(state='disabled' if disabled else 'readonly' if name in {'device', 'match_format'} else 'normal')
        mode.configure(state='disabled' if running else 'readonly')
        for button in (start_button, check_button, load_button, save_button):
            button.configure(state='disabled' if running else 'normal')
        stop_button.configure(state='normal' if running else 'disabled')

    def set_mode(_event=None):
        preset = PROFILE_PRESETS[labels[selected.get()]]
        description.set(preset['description'])
        fields['learning'].set(preset['learning'])
        fields['match_format'].set(preset['match_format'])
        fields['friendly_fire'].set(False)
        fields['opponents'].set('Player3, Player4' if preset['match_format'] == '2v2' else 'Player2')
        refresh_fields()

    def change_format(_event=None):
        fields['opponents'].set('Player3, Player4' if fields['match_format'].get() == '2v2' else 'Player2')
        if fields['match_format'].get() == '1v1':
            fields['friendly_fire'].set(False)
        refresh_fields()

    def settings(save):
        try:
            if save:
                options = current_options()
                path = filedialog.asksaveasfilename(parent=root, defaultextension='.json',
                                                    filetypes=[('Session settings', '*.json')], confirmoverwrite=True)
                if path:
                    Path(path).write_text(json.dumps(options.to_dict(), indent=2) + '\n', encoding='utf-8')
            else:
                path = filedialog.askopenfilename(parent=root, filetypes=[('Session settings', '*.json')])
                if not path:
                    return
                options = SessionOptions.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
                values = options_to_form(options)
                selected.set(PROFILE_PRESETS[values['profile']]['label'])
                for name, variable in fields.items():
                    variable.set(values[name])
                description.set(PROFILE_PRESETS[values['profile']]['description'])
                refresh_fields()
        except (ValueError, TypeError, KeyError, OSError) as error:
            messagebox.showerror('Session settings', str(error), parent=root)

    def launch(check=False):
        nonlocal running
        try:
            options = current_options()
            folder = controller.start(options, check=check)
            running = True
            status.set('Checking provider connection; no match will be started.' if check else
                       f'Running {"learning" if options.learning else "evaluation"} process; see provider status in the log.')
            append_log(f'Run files: {folder}\n')
            refresh_fields()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            messagebox.showerror('Cannot start session', str(error), parent=root)

    def stop():
        controller.request_stop()
        status.set('Stopping — waiting for the session runner to save and exit…')
        stop_button.configure(state='disabled')
        stopping_process = controller.process
        root.after(10000, lambda: force_button.configure(state='normal')
                   if running and controller.process is stopping_process else None)

    def force():
        if running and messagebox.askyesno('Force end session',
                'End this session process now? Unsaved learning progress may be lost.', parent=root):
            controller.force_stop()

    def append_log(text):
        log.configure(state='normal')
        log.insert('end', text)
        excess = int(log.index('end-1c').split('.')[0]) - 500
        if excess > 0:
            log.delete('1.0', f'{excess + 1}.0')
        log.see('end')
        log.configure(state='disabled')

    def tick():
        nonlocal running
        append_log(''.join(controller.drain_output()))
        if running and controller.poll() is not None:
            code = controller.poll()
            running = False
            status.set('Stopped forcibly; unsaved progress may be lost.' if controller.forced else
                       'Stopped.' if controller.stop_requested else
                       f'Failed (exit {code}); see the log.' if code else
                       'Provider check passed; no match was started.' if controller.check else 'Session finished.')
            force_button.configure(state='disabled')
            refresh_fields()
            if closing:
                root.destroy()
                return
        root.after(100, tick)

    def close():
        nonlocal closing
        if running:
            closing = True
            stop()
        else:
            root.destroy()

    start_button = ttk.Button(actions, text='Start session', command=launch)
    check_button = ttk.Button(actions, text='Check connection', command=lambda: launch(True))
    stop_button = ttk.Button(actions, text='Stop and save', command=stop, state='disabled')
    force_button = ttk.Button(actions, text='Force end', command=force, state='disabled')
    load_button = ttk.Button(actions, text='Load settings…', command=lambda: settings(False))
    save_button = ttk.Button(actions, text='Save settings…', command=lambda: settings(True))
    for button in (start_button, check_button, stop_button, force_button, load_button, save_button):
        button.pack(side='left', padx=(0, 5))
    mode.bind('<<ComboboxSelected>>', set_mode)
    widgets['match_format'].bind('<<ComboboxSelected>>', change_format)
    selected.set(PROFILE_PRESETS['bot_training']['label'])
    set_mode()
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(100, tick)
    root.mainloop()
    return 0
