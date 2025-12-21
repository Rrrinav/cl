# C# CL

> Command line args parsing library

## Bugs

- We are not checking repeated scalar
- We are not handling commas properly
- We are not handling short falg concatenation
- We are not handling -ve numbers because '-' minus sign can show a new option too.
- Long single letter name may collide with short names.
- No bound checks on arity
- opt->type could be corrupted in switch case

## TODOs

- [ ] Implement sub-commands/commands grouping
- [ ] Testing
- [ ] Environment variable fallback (infrastructure exists but not implemented)
- [ ] Mutual exclusivity constraints (--foo and --bar can't both be set)
- [ ] Dependency constraints (--foo requires --bar to also be set)
- [ ] Custom help formatting/theming options
- [ ] Automatic short flag generation from long names
- [ ] Version flag handling (--version, -V)
- [ ] Usage examples in help output
- [ ] Grouped options in help (e.g., "Input Options:", "Output Options:")
- [ ] Custom error messages per option
- [ ] Did-you-mean suggestions for typos (--helo -> did you mean --help?)
- [ ] Option aliasing (multiple names for same option)
- [ ] Deprecation warnings for options
- [ ] Response files (@file.txt to read args from file)
- [ ] Parse from string vector (not just argc/argv)
- [ ] Better error messages with context (show where in argv the error occurred)
- [ ] Negation flags (--no-color to disable --color)
- [ ] Count flags (-vvv for verbosity level)
- [ ] Benchmark suite
