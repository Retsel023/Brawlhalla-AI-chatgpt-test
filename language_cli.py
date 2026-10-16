"""Translate one local-language command into saved, validated session settings."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from agent import load_config
from guidance_io import write_guidance, write_json_atomic
from session_profiles import SessionOptions, default_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='quoted instruction, such as "keep more distance"')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--session-file', type=Path, help='existing session settings; otherwise bot training defaults')
    parser.add_argument('--output', type=Path, help='save resulting session settings to this file')
    parser.add_argument('--guidance-file', type=Path, help='write only tactical goals for a running session')
    args = parser.parse_args(argv)
    try:
        from language_commands import interpret_command
        options = (SessionOptions.from_dict(json.loads(args.session_file.read_text(encoding='utf-8')))
                   if args.session_file else default_options())
        if args.guidance_file is not None and args.guidance_file.exists():
            from guidance_settings import validate_guidance
            current = validate_guidance(json.loads(args.guidance_file.read_text(encoding='utf-8')))
            options = replace(options, guidance=current)
        result = interpret_command(args.text, options, load_config(args.config))
        if result.needs_clarification:
            print(result.message, file=sys.stderr)
            return 2
        if args.guidance_file:
            before, after = options.to_dict(), result.options.to_dict()
            before.pop('guidance')
            after.pop('guidance')
            if before != after:
                raise ValueError('Live commands can change tactical goals only. Stop the session before changing its setup.')
            write_guidance(args.guidance_file, result.options.guidance)
        if args.output:
            write_json_atomic(args.output, result.options.to_dict())
        print(result.message)
        print(json.dumps(result.options.to_dict(), indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as error:
        print(f'Command failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
