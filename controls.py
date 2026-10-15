"""Platform-independent action mapping and thread-safe human feedback."""
from threading import Lock

ACTIONS = (
    (), ('left',), ('right',), ('jump',), ('down',),
    ('light',), ('heavy',), ('dodge',),
    ('left', 'jump'), ('right', 'jump'),
    ('left', 'light'), ('right', 'light'),
    ('jump', 'light'), ('down', 'light'),
    ('left', 'heavy'), ('right', 'heavy'),
)

class Feedback:
    def __init__(self):
        self.lock = Lock()
        self.reward = 0.0
        self.stop = False
        self.done = False

    def add(self, value):
        with self.lock:
            self.reward += value

    def consume(self):
        with self.lock:
            reward, done = self.reward, self.done
            self.reward, self.done = 0.0, False
            return reward, done
