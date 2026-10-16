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

from session_profiles import PROFILE_PRESETS, SessionOptions, default_options, default_teammate_kind
from guidance_io import write_guidance

TEAMMATE_LABELS = {'none': 'No teammate', 'random': 'Random teammate',
                   'regular': 'Regular teammate (you)', 'ai': 'AI controls both', 'bot': 'Game bot'}


def build_options(form):
    """Convert editable text fields into the runner's validated session settings."""
    enemies = form.get('opponents', '')
    if isinstance(enemies, str):
        enemies = [value.strip() for value in enemies.split(',') if value.strip()]
    match_format = form.get('match_format') or PROFILE_PRESETS[form['profile']]['match_format']
    teammate_kind = form.get('teammate_kind') or default_teammate_kind(form['profile'], match_format)
    teammate_kind = {label: kind for kind, label in TEAMMATE_LABELS.items()}.get(teammate_kind, teammate_kind)
    account_id = form.get('teammate_account_id')
    if match_format == '1v1':
        teammate_kind, account_id = 'none', None
    elif account_id is not None:
        if isinstance(account_id, str):
            account_id = int(account_id.strip()) if account_id.strip() else None
        elif type(account_id) is not int:
            raise ValueError('Teammate account ID must be a nonnegative integer')
    return default_options(
        form['profile'], match_format=match_format,
        self_player=form.get('self_player', 'Player1').strip(),
        teammate_player=form.get('teammate_player', 'Player2').strip() if match_format == '2v2' else None,
        teammate_kind=teammate_kind, teammate_account_id=account_id,
        opponents=enemies or None, friendly_fire=form.get('friendly_fire', False),
        learning=form.get('learning', PROFILE_PRESETS[form['profile']]['learning']),
        steps=int(form.get('steps', 0)), max_seconds=float(form.get('max_seconds', 0)),
        checkpoint_interval=int(form.get('checkpoint_interval', 1000)),
        model_root=form.get('model_root', 'models/sessions').strip(),
        initialize_from=form.get('initialize_from', '').strip() or None,
        device=form.get('device', 'auto'), legend=(form.get('legend') or '').strip() or None,
        guidance=form.get('guidance'))


def options_to_form(options):
    form = options.to_dict()
    form.update(form.pop('profile'))
    form['opponents'] = ', '.join(form['opponents'])
    form['teammate_player'] = form['teammate_player'] or 'Player2'
    form['teammate_account_id'] = '' if form['teammate_account_id'] is None else str(form['teammate_account_id'])
    form['initialize_from'] = form['initialize_from'] or ''
    form['legend'] = form['legend'] or ''
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
        self.run_dir = self.options_path = self.stop_path = self.guidance_path = None
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
        self.guidance_path = self.run_dir / 'guidance.json'
        write_guidance(self.guidance_path, options.guidance)
        self.options_path.write_text(json.dumps(options.to_dict(), indent=2) + '\n', encoding='utf-8')
        self.check, self.stop_requested, self.forced = check, False, False
        command = [sys.executable, '-u', str(self.repo / 'agent.py'), 'session',
                   '--session-file', str(self.options_path), '--config', str(self.config),
                   '--stop-file', str(self.stop_path), '--guidance-file', str(self.guidance_path)]
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

    def update_guidance(self, guidance):
        if self.process is None or self.poll() is not None or self.check or self.stop_requested:
            raise RuntimeError('No active combat session can receive this suggestion')
        write_guidance(self.guidance_path, guidance)

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


class LanguageCommandTask:
    """Run model requests off the UI thread; deliver results through a queue."""

    def __init__(self, text, options, config_path):
        self.options = options
        self.output = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, args=(text, config_path), daemon=True)
        self.thread.start()

    def _run(self, text, config_path):
        try:
            from agent import load_config
            from language_commands import interpret_command
            result = interpret_command(text, self.options, load_config(config_path))
            self.output.put((result, None))
        except Exception as error:
            self.output.put((None, str(error)))

    def poll(self):
        try:
            return self.output.get_nowait()
        except queue.Empty:
            return None


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
    available_height = max(700, root.winfo_screenheight() - 80)
    root.geometry(f'900x{min(950, available_height)}')
    root.minsize(760, min(800, available_height))
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
              for name, value in initial.items() if name not in {'profile', 'online', 'opponent_kind', 'guidance'}}
    guidance = initial['guidance']
    widgets = {}
    captions = [('match_format', 'Bot match format'), ('learning', 'Update policy weights'),
                ('steps', 'Steps (0 = continuous)'), ('max_seconds', 'Time limit in seconds (0 = unlimited)'),
                ('checkpoint_interval', 'Checkpoint every N steps'), ('self_player', 'AI player key'),
                ('teammate_kind', 'Teammate arrangement (2v2)'), ('teammate_account_id', 'Regular teammate account ID (optional)'),
                ('teammate_player', 'Teammate key (2v2)'), ('opponents', 'Opponent keys (comma separated)'),
                ('friendly_fire', 'Friendly fire (2v2)'), ('device', 'Device (cuda = AMD ROCm)'),
                ('legend', 'Primary AI legend (optional)'),
                ('model_root', 'Model folder'), ('initialize_from', 'Initialize from checkpoint (optional)')]
    for row, (name, caption) in enumerate(captions, 3):
        ttk.Label(panel, text=caption).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=3)
        if name in boolean_fields:
            widget = ttk.Checkbutton(panel, variable=fields[name])
        elif name in {'device', 'match_format', 'teammate_kind'}:
            values = {'device': ('auto', 'cpu', 'cuda'), 'match_format': ('1v1', '2v2'),
                      'teammate_kind': tuple(label for kind, label in TEAMMATE_LABELS.items() if kind != 'none')}
            widget = ttk.Combobox(panel, textvariable=fields[name], state='readonly',
                                  values=values[name])
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
    actions_row = 3 + len(captions)
    ttk.Label(panel, textvariable=status, wraplength=820).grid(row=actions_row + 1, column=0, columnspan=3, sticky='w', pady=8)
    actions = ttk.Frame(panel)
    actions.grid(row=actions_row, column=0, columnspan=3, sticky='ew', pady=10)
    log = tk.Text(panel, height=15, state='disabled', wrap='word')
    log.grid(row=actions_row + 2, column=0, columnspan=3, sticky='nsew')
    panel.rowconfigure(actions_row + 2, weight=1)
    running = closing = False
    command_task = command_process = None
    chat_window = chat_log = chat_entry = send_button = None

    def current_options():
        return build_options({'profile': labels[selected.get()], 'guidance': guidance,
                              **{name: value.get() for name, value in fields.items()}})

    def apply_options(options):
        nonlocal guidance
        values = options_to_form(options)
        guidance = values['guidance']
        selected.set(PROFILE_PRESETS[values['profile']]['label'])
        for name, variable in fields.items():
            variable.set(TEAMMATE_LABELS[values[name]] if name == 'teammate_kind' else values[name])
        change_teammate()

    def refresh_fields():
        teams = fields['match_format'].get() == '2v2'
        for name, widget in widgets.items():
            disabled = running or (name in {'teammate_player', 'friendly_fire', 'teammate_kind'} and not teams)
            disabled |= name == 'teammate_account_id' and (not teams or fields['teammate_kind'].get() != TEAMMATE_LABELS['regular'])
            disabled |= name == 'match_format' and labels[selected.get()] != 'bot_training'
            widget.configure(state='disabled' if disabled else 'readonly' if name in {'device', 'match_format', 'teammate_kind'} else 'normal')
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
        fields['teammate_kind'].set(TEAMMATE_LABELS[default_teammate_kind(labels[selected.get()], preset['match_format'])])
        fields['teammate_account_id'].set('')
        fields['opponents'].set('Player3, Player4' if preset['match_format'] == '2v2' else 'Player2')
        change_teammate()

    def change_format(_event=None):
        fields['opponents'].set('Player3, Player4' if fields['match_format'].get() == '2v2' else 'Player2')
        if fields['match_format'].get() == '1v1':
            fields['friendly_fire'].set(False)
        fields['teammate_kind'].set(TEAMMATE_LABELS[default_teammate_kind(labels[selected.get()], fields['match_format'].get())])
        fields['teammate_account_id'].set('')
        change_teammate()

    def change_teammate(_event=None):
        kind = fields['teammate_kind'].get()
        if kind != TEAMMATE_LABELS['regular']:
            fields['teammate_account_id'].set('')
        hints = {TEAMMATE_LABELS['ai']: 'AI controls both players through independent inputs, such as a keyboard and a secondary controller.',
                 TEAMMATE_LABELS['regular']: 'You can keep your keyboard while the AI uses the secondary controller slot. Enter your account ID, or pin the first observed teammate.',
                 TEAMMATE_LABELS['random']: 'The teammate may be a different person each match.',
                 TEAMMATE_LABELS['bot']: 'The game controls the teammate bot.'}
        description.set(PROFILE_PRESETS[labels[selected.get()]]['description'] + ' ' + hints.get(kind, ''))
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
                apply_options(options)
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

    def chat_append(text):
        if chat_log is not None and chat_log.winfo_exists():
            chat_log.configure(state='normal')
            chat_log.insert('end', text + '\n\n')
            excess = int(chat_log.index('end-1c').split('.')[0]) - 300
            if excess > 0:
                chat_log.delete('1.0', f'{excess + 1}.0')
            chat_log.see('end')
            chat_log.configure(state='disabled')

    def submit_command(_event=None):
        nonlocal command_task, command_process
        if command_task is not None:
            return 'break'
        text = chat_entry.get().strip()
        if not text:
            return 'break'
        try:
            options = current_options()
            command_process = controller.process
            command_task = LanguageCommandTask(text, options, config)
            chat_append('You: ' + text)
            chat_entry.delete(0, 'end')
            send_button.configure(state='disabled')
            status.set('Interpreting your command with the local language model…')
        except (ValueError, TypeError, KeyError, OSError) as error:
            chat_append('Command not applied: ' + str(error))
        return 'break'

    def show_chat():
        nonlocal chat_window, chat_log, chat_entry, send_button
        if chat_window is not None and chat_window.winfo_exists():
            chat_window.lift()
            return
        chat_window = tk.Toplevel(root)
        chat_window.title('Talk to the agent')
        chat_window.geometry('720x500')
        chat_window.minsize(500, 350)
        ttk.Label(chat_window, text='Try “play as Hattori”, “keep more distance”, or “practise down light then jump light”. '
                  'Tactical suggestions can update a running session. Stop before changing its setup. '
                  'Responses improve through training; drills do not establish true game combos.',
                  wraplength=680, padding=12).pack(fill='x')
        chat_log = tk.Text(chat_window, height=12, state='disabled', wrap='word')
        chat_log.pack(fill='both', expand=True, padx=12)
        entry_panel = ttk.Frame(chat_window, padding=12)
        entry_panel.pack(fill='x')
        chat_entry = ttk.Entry(entry_panel)
        chat_entry.pack(side='left', fill='x', expand=True)
        chat_entry.bind('<Return>', submit_command)
        send_button = ttk.Button(entry_panel, text='Send', command=submit_command,
                                 state='disabled' if command_task else 'normal')
        send_button.pack(side='left', padx=(8, 0))
        chat_append('Run Ollama locally with the model selected in config.json. '
                    'The default model can be installed with: ollama pull qwen2.5:3b')
        chat_append('Current tactical goals: ' + json.dumps(guidance))
        chat_entry.focus_set()

    def finish_command(result, error):
        nonlocal command_task
        source = command_task.options
        command_task = None
        if send_button is not None and send_button.winfo_exists():
            send_button.configure(state='normal')
        try:
            if error:
                raise RuntimeError(error)
            if result.needs_clarification:
                chat_append('Agent: ' + result.message)
                status.set('Command needs clarification; no settings changed.')
                return
            if source != current_options() or (running and controller.process is not command_process):
                raise ValueError('Session settings changed while the model was replying. Send your command again.')
            if running:
                before, after = source.to_dict(), result.options.to_dict()
                before.pop('guidance')
                after.pop('guidance')
                if before != after:
                    raise ValueError('Stop the session before changing mode, legend, or other setup settings. No changes applied.')
                if result.changed:
                    controller.update_guidance(result.options.guidance)
            apply_options(result.options)
            message = result.message + (' Tactical goals were sent for the next decision.' if running and result.changed else
                                        ' Settings apply when you start a session.' if result.changed else '')
            chat_append('Agent: ' + message)
            append_log(message + '\n')
            status.set(message)
        except (ValueError, TypeError, RuntimeError, KeyError, OSError) as failure:
            chat_append('Command not applied: ' + str(failure))
            status.set('Command not applied: ' + str(failure))

    def tick():
        nonlocal running
        append_log(''.join(controller.drain_output()))
        if command_task is not None:
            response = command_task.poll()
            if response is not None:
                finish_command(*response)
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
    chat_button = ttk.Button(actions, text='Talk to agent…', command=show_chat)
    for button in (start_button, check_button, stop_button, force_button, load_button, save_button, chat_button):
        button.pack(side='left', padx=(0, 5))
    mode.bind('<<ComboboxSelected>>', set_mode)
    widgets['match_format'].bind('<<ComboboxSelected>>', change_format)
    widgets['teammate_kind'].bind('<<ComboboxSelected>>', change_teammate)
    selected.set(PROFILE_PRESETS['bot_training']['label'])
    set_mode()
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(100, tick)
    root.mainloop()
    return 0
